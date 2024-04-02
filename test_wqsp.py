import os
import sys
import shutil
import json
import fire
import gradio as gr
import requests
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from commonsenseqa_prompter import CommonsensePrompter
from transformers.generation import utils
from datasets import load_dataset
import os
import random
import nltk
from fuzzywuzzy import fuzz


def init_weight_KG(model, path, config):
    for idx, layer in enumerate(model):
        layer.init()
    for idx, layer in enumerate(model):
        if idx + 1 in config.layer_insertion:
            layer.KG_infuded_module.init(path)
    return model

def main(
    is_KG: bool = False,
    base_model: str = "llama2_7B/",
    lora_weights: str = "./llama2-lora",
    prompt_template: str = "commonsenseQA"  # The prompt template to use, will default to alpaca.
):
    kg = 'cn'
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    prompter = CommonsensePrompter(tokenizer, kg, prompt_template)
    is_prompt = False
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_8bit,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float32,
    )
    shutil.copyfile(base_model + "/utils.py", os.path.abspath(sys.modules[utils.__module__].__file__), follow_symlinks=True)
    shutil.copyfile(base_model + "/modeling_llama.py", os.path.abspath(sys.modules[LlamaForCausalLM.__module__].__file__), follow_symlinks=True)
    #model = model.cuda()
    if is_KG:
        kg_infused_weight = lora_weights
        kg_embed_path = "./data/kgs/conceptnet/ent.npy"
        model.base_model.model.model.layers = init_weight_KG(model.base_model.model.model.layers, kg_embed_path, model.config)
        print(model)
        model.base_model.model.model.layers = load_KG_module(model.base_model.model.model.layers, kg_infused_weight, model.config)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    #if not load_8bit:
    #   model.half()  # seems to fix bugs for some users.

    model.eval()
    #model = model.cuda()
    def evaluate(
        input_ids,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        # 必须强迫出现batch_size
        input_ids = input_ids.unsqueeze(0).cuda()
        #labels = labels.unsqueeze(0).cuda()
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            use_cache = True,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        try:
            r1 = prompter.get_response(output)
        except:
            r1 = "nop"
        return r1, output

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        words_ents_list, words_subtoken_map = prompter.get_mapping_ids(prompt, result["input_ids"], tokenizer)
        try:
            result["words_ents_list"] = torch.nn.utils.rnn.pad_sequence(words_ents_list, batch_first=True, padding_value=-1)
            result["words_subtoken_map"] = torch.nn.utils.rnn.pad_sequence(words_subtoken_map, batch_first=True, padding_value=-1)
        except:
            result["words_ents_list"] = []
            result["words_subtoken_map"] = []
        #print(result)
        return result

    def generate_and_tokenize_prompt(data_point):
        #print(data_point)
        prompt, answer = prompter.generate_input_llama_format(
            data_point["question"]["choices"],
            data_point["question"]["stem"],
            data_point["answerKey"],
            #data_point["convert_prompt"]
        )
        tokenized_full_prompt = tokenize(prompt)
        
        tokenized_full_prompt["input_ids"] = torch.IntTensor(tokenized_full_prompt["input_ids"])
        #print(tokenized_full_prompt)
        return (tokenized_full_prompt, answer)
    
    def generate_and_tokenize(data_point):
        #print(data_point)
        tokenized_full_prompt = tokenize(data_point)
        
        tokenized_full_prompt["input_ids"] = torch.IntTensor(tokenized_full_prompt["input_ids"])
        #print(tokenized_full_prompt)
        return tokenized_full_prompt
    
    # testing code for readme
    num = 0
    acc = 0
    nop = 0
    
    file = open('/data1/xdluo/alpaca-lora-main/data/wqsp/test.json', 'r', encoding='utf-8')
    lines = file.read()
    lines = json.loads(lines)
    file.close()
    txt_res = []
    for line in tqdm(lines["Questions"]):
        #data_point = json.loads(line)
        data_point = line
        prompt_key = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'''                
        prompt_key += "\n\n### Input:"
        prompt_key += data_point["RawQuestion"]
        if is_prompt and len(data_point["knowledge"]) != 0:
            #prompt_key += " The relevant passage is: "
            #prompt_key += data_point["ctxs"][0]["text"]
            prompt_key += " The relevant knowledge is: "
            prompt_key += data_point["knowledge"][0]
        #prompt_key += line
        knowledge = None
        #if len(data_point['knowledges']) != 0:
        #    prompt_key += (" " + data_point['knowledges'][0].replace("</s><s>", "") + " ")
        #    knowledge = data_point['knowledges'][0].replace("</s><s>", "")
        #prompt_key += " Please explain it in detail."
        #prompt_key += " Please directly give the answer."
        end_t = "\n\n### Response:\n"
        prompt_key += end_t
        answer = []
        for ans in data_point["Parses"][0]["Answers"]:
            an = ans["EntityName"]
            if an == None:
                continue
            #an = ans["text"].lower()
            answer.append(an.lower())
        if len(answer) == 0:
            continue
        tokenized_full_prompt = generate_and_tokenize(prompt_key)
        
        kwargs = {
            "words_ents_list": [tokenized_full_prompt["words_ents_list"]],
            "words_subtoken_map": [tokenized_full_prompt["words_subtoken_map"]]
        }
        #print("the answer is : {}".format(answer))
        result, output = evaluate(tokenized_full_prompt["input_ids"], temperature=0.1, top_p=0.75, 
                    top_k=40, num_beams=4, max_new_tokens=256, stream_output=False, **kwargs)
        t = grammarTree_parse(result, answer)
        print("question is {}, answer is {}, result is {}, acc is {}".format(data_point["RawQuestion"], answer, result, t))
        txt_res.append("question is {}, answer is {}, knowledge is {}, result is {}, acc is {}".format(data_point["RawQuestion"], answer, knowledge, result, t))
        acc += t
        num += 1
    #f=open("llama2-wqsp-wn.txt","w")
    #for line in txt_res:
    #    f.write(line+'\n')
    #f.close()
    print(acc)
    print(num)
    print(acc / num)
    
    

def grammarTree_parse(result, answers):
    lowersetence=result.lower()
    if "I'm sorry" in result:
        return 0
    text = nltk.word_tokenize(lowersetence)
    sentence=nltk.pos_tag(text)
    #grammar = "NP:{<JJ|NN|NNS.*><POS|IN.*><NN|NNS.*>}"
    grammar = r"""
                NP:{<JJ|NN><POS|IN>?<NN>+}
                PP:{<NN|NNS|NNP|NNPS>}

                """
    cp = nltk.RegexpParser(grammar) #生成规则
    result = cp.parse(sentence) #进行分块

    substring=[]
    finalstring= []
    for subtree in result.subtrees():
        if ((subtree.label() == 'NP')|(subtree.label()=='PP')):
            substring.append(subtree)
    for each in substring:
        length=len(each)
        #for i in (0,length-1):
            #print(each[i])
        final = ''
        for i in range(0,length):
            final += each[i][0] + ' '
        finalstring.append(final)
        
    for st in finalstring:
        #st = st[0]
        for ans in answers:
            if fuzz.ratio(ans.lower(), st) > 50:
                return 1
    return 0


def load_KG_module(model, path, config):
    for idx, layer in enumerate(model):
        if idx + 1 in config.layer_insertion:
            tmp = path + r"/KG_retrieve_{}.bin".format(idx)
            buffer = torch.load(tmp, map_location=torch.device("cuda"))
            layer.KG_infuded_module.load_state_dict(buffer, strict=False)
    #path += r"KG_retrieve.pth"
    #buffer = torch.load(path, map_location=torch.device("cuda"))
    #model.load_state_dict(buffer, strict=False)
    return model

def generate_prompts_from_KG(tokens):
    result_list = []
    for token in tokens:
        token = token.split()
        token = "_".join(token)
        obj = requests.get('http://api.conceptnet.io/c/en/'+token).json()
        try:
            total = len(obj['edges'])
            i = 0
            count = 0
            while i < total and count < 5:
                max_edge = obj['edges'][i]
                start = max_edge["start"]
                end = max_edge["end"]
                relation = max_edge["rel"]
                if start["language"] != "en" or end["language"] != "en":
                    i += 1
                    continue
                result_list += generate_prompt(start, end, relation)
                i += 1
                count += 1
        except:
            pass
    return result_list

def generate_prompt(start, end, relation):
    result = []
    prompt_triple = '''
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:Generate short, reasonable sentences based on the given entities that contains the given entity.\n\n\n### Input: entity: {}, {}; relation: {}\n\n\n### Response:\n
    '''
    prompt_triple = prompt_triple.format(start["label"], end["label"], relation["label"])
    result.append(prompt_triple)
    return result

if __name__ == "__main__":
    fire.Fire(main)