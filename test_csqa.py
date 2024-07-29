import os
import sys
import json
import fire
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import shutil
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from commonsenseqa_prompter import CommonsensePrompter
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
    dataset: str = "siqa",
    prompt_template: str = "commonsenseQA", 
):
    shutil.copyfile(base_model + "/modeling_llama.py", os.path.abspath(sys.modules[LlamaForCausalLM.__module__].__file__), follow_symlinks=True)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    prompter = CommonsensePrompter(tokenizer, "cn", prompt_template)
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
    #model = model.cuda()
    if is_KG:
        kg_embed_path = "./data/kgs/conceptnet/ent.npy"
        kg_infused_weight = lora_weights
        model.base_model.model.model.layers = init_weight_KG(model.base_model.model.model.layers, kg_embed_path, model.config)
        #print(model)
        model.base_model.model.model.layers = load_KG_module(model.base_model.model.model.layers, kg_infused_weight, model.config)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    #if not load_8bit:
    #   model.half()  # seems to fix bugs for some users.

    model.eval()
    #if torch.__version__ >= "2" and sys.platform != "win32":
    #    model = torch.compile(model)
    #model = model.cuda()
    def evaluate(
        input_ids,
        labels, 
        **kwargs,
    ):
        # 必须强迫出现batch_size
        input_ids = input_ids.unsqueeze(0).cuda()
        labels = labels.unsqueeze(0).cuda()
        # Without streaming
        with torch.no_grad():

            generation_output = model(
                input_ids=input_ids,
                labels=labels, 
                words_ents_list=kwargs.get("words_ents_list"),
                words_subtoken_map=kwargs.get("words_subtoken_map"),
            ).loss.item()
            
        return generation_output

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
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 512
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
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
    
    def generate_prompt_for_GPT3(data_point):
        prompt, answer = prompter.generate_GPT3_prompt(
            data_point["cands"],
            data_point["query"],
            data_point["answer"],
            data_point["knowledges"] 
        )
        '''
        full_prompt = data_point["convert_prompt"]
        try:
            full_prompt = full_prompt.replace("These information may be useful:", "")
        except:
            pass
        '''
        full_prompts = [prompt + choice for choice in data_point["cands"]]

        tokenized_user_prompt = tokenize(prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        result_list = tokenize_prompt(full_prompts, user_prompt_len)
        return result_list, answer

    def generate_prompt_for_KG(data_point):
        #print(data_point)
        prompt = prompter.generate_input_llama_format(
            data_point["question"],
        )
        answer = data_point["answer"]
        #prompt = data_point["convert_prompt"]
        
        '''
        try:
            prompt = prompt.replace("These information may be useful:", "")
        except:
            pass
        '''
        full_prompts = [prompt + choice for choice in data_point["choices"]]

        tokenized_user_prompt = tokenize(prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        result_list = tokenize_prompt(full_prompts, user_prompt_len)
        return result_list, answer

    def tokenize_prompt(full_prompts, user_prompt_len):
        result_list = []

        for full_prompt in full_prompts:
            tokenized_full_prompt = tokenize(full_prompt)
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            
            tokenized_full_prompt["labels"] = torch.LongTensor(tokenized_full_prompt["labels"])
            tokenized_full_prompt["input_ids"] = torch.IntTensor(tokenized_full_prompt["input_ids"])
            result_list.append(tokenized_full_prompt)
        return result_list
        #print(tokenized_full_prompt)
    # testing code for readme
    num = 0
    acc = 0
    nop = 0
    is_GPT3 = False
    prob_answers = []
    if dataset == 'csqa':
        file = open('./data/commonsenseqa/devc.json', 'r', encoding='utf-8')
    elif dataset == 'siqa':
        file = open('./data/SIQA/devc.json', 'r', encoding='utf-8')
    else:
        raise Exception("Error dataset name!") 
    lines = file.read()
    lines = json.loads(lines)
    file.close()
    print(type(lines))
    for line in tqdm(lines):
        data_point = line
        num += 1
        th = 0
        min_num = None
        # Decide whether to use KG prompt or GPT3 knowledge
        if is_GPT3:
            tokenized_full_prompts, answer = generate_prompt_for_GPT3(data_point)
        else:
            tokenized_full_prompts, answer = generate_prompt_for_KG(data_point)
        score = []
        for i, tokenized_full_prompt in enumerate(tokenized_full_prompts):
            kwargs = {
                "words_ents_list": [tokenized_full_prompt["words_ents_list"]],
                "words_subtoken_map": [tokenized_full_prompt["words_subtoken_map"]]
            }
            #print("the answer is : {}".format(answer))
            output = evaluate(tokenized_full_prompt["input_ids"], tokenized_full_prompt["labels"], **kwargs)
            score.append(-output)
            if min_num == None:
                min_num = output
            elif min_num > output:
                min_num = output
                th = i
        my_result = None
        prob_answer = -1
        scores = torch.tensor(score)
        probs = torch.softmax(scores, dim=0)
        index = -1
        if is_GPT3:
            my_result = data_point["cands"][th]
            index = data_point["cands"].index(answer)
        else:
            my_result = data_point["choices"][th]
            idx = data_point["choices"].index(answer)
            prob_answer = probs[idx].item()
        prob_answers.append(prob_answer)
        print("result and answer are {}, {}".format(my_result, answer))
        if my_result == answer:
            acc += 1
        '''      
        if result in answer or answer in result:
            acc += 1
        if result == 'nop':
            nop += 1
        '''
        #else:

            #print(output)
            #print("error: answer is {}, result is {}".format(answer, result))
    prob_answers = np.array(prob_answers)
    prob_answers = np.mean(prob_answers)
    print(prob_answers)
    print("acc is")
    print(acc / num)
    print(acc)
    print(nop)
    print(num)


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


if __name__ == "__main__":
    fire.Fire(main)