import os
import sys
import json
import fire
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
import numpy as np
import transformers
import shutil
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from commonsenseqa_prompter import CommonsensePrompter
import os


def init_weight_KG(model, path):
    for idx, layer in enumerate(model):
        layer.init()
    for idx, layer in enumerate(model):
        if idx + 1 in config.layer_insertion:
            layer.KG_infuded_module.init(path)
    return model

def main(
    is_KG: bool = False,
    base_model: str = "llama2_7B/",
    save_path: str = "bbh_lora.txt",
    kg_infused_weight = "/llama2-lora",
    lora_weights: str = "",
    prompt_template: str = "commonsenseQA"  # The prompt template to use, will default to alpaca.
):
    kg_infused_weight = "/llama2-lora"
    lora_weights = kg_infused_weight
    shutil.copyfile(base_model + "/modeling_llama.py", os.path.abspath(sys.modules[LlamaForCausalLM.__module__].__file__), follow_symlinks=True)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    prompter = CommonsensePrompter(tokenizer, "cn", prompt_template)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_8bit,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code = True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float32,
    )
    if is_KG:
        kg_embed_path = "/data1/xdluo/alpaca-lora-main/data/kgs/conceptnet/ent.npy"
        model.base_model.model.model.layers = init_weight_KG(model.base_model.model.model.layers, kg_embed_path)
        print(model)
        model.base_model.model.model.layers = load_KG_module(model.base_model.model.model.layers, kg_infused_weight, model.config)
    #model = model.cuda()
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
                return_dict = True
                #generation_config=generation_config,
                #return_dict_in_generate=True,
                #output_scores=True,
                #max_new_tokens=max_new_tokens,
            ).loss.item()
        return generation_output

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=2000,
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
    
    def generate_and_tokenize(prompt, input):
        '''
        full_prompts: size：[candidate_num * knowledge_num]
        user_prompt_len: size: [candidate_num * knowledge_num]
        '''
        answers = input.split("Options:\n")[1].split('\n')
        full_prompts = [prompt + answer[4:] for answer in answers]

        tokenized_user_prompt = tokenize(prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        result_list = tokenize_prompt(full_prompts, user_prompt_len)
        return result_list
        #print(tokenized_full_prompt)
    
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

    # testing code for readme
    label_root = r'./data/bbh/'
    txt_res = []
    score_sum = 0
    acc_sum = 0
    for _, _, fs in os.walk(label_root):
        for f in tqdm(fs):
            FILE = label_root + f
            num = 0
            acc = 0
            final_score = 0
            file = open(FILE, 'r', encoding='utf-8')
            lines = file.read()
            lines = json.loads(lines)
            file.close()
            final_data = []
            print(FILE)
            #try:
            for line in tqdm(lines["examples"]):
                #data_point = json.loads(line)
                data_point = line
                th = 0
                num += 1
                min_num = None
                prompt_key = '''Below is an instruction that describes a task, paired with an input that provides further context. \n\n'''             
                prompt_key += "\n\n### Input:"
                prompt_key += data_point["input"]
                prompt_key += " Please directly give the answer."
                end_t = "\n\n### Response:\n"
                prompt_key += end_t
                answer = data_point["target"]
                index = int(ord(answer[1]) - ord('A'))
                answers = data_point["input"].split("Options:\n")[1].split('\n')
                answer = answers[index][4:]
                tokenized_full_prompts = generate_and_tokenize(prompt_key, data_point["input"])
                scores = []
                for i, tokenized_full_prompt in enumerate(tokenized_full_prompts):
                    kwargs = {
                        "words_ents_list": [tokenized_full_prompt["words_ents_list"]],
                        "words_subtoken_map": [tokenized_full_prompt["words_subtoken_map"]]
                    }
                    #print("the answer is : {}".format(answer))
                    output = evaluate(tokenized_full_prompt["input_ids"], tokenized_full_prompt["labels"], **kwargs)
                    scores.append(-output)
                    if min_num == None:
                        min_num = output
                    elif min_num > output:
                        min_num = output
                        th = i
                scores = torch.tensor(scores)
                probs = torch.softmax(scores, dim=0)
                my_result = None
                final_score += probs[index].item()
                my_result = answers[th][4:]
                print("result, answer and score are {}, {}, {}".format(my_result, answer, probs[index].item()))
                if my_result == answer:
                    acc += 1
            print(acc)
            print(num)
            print(acc / num)
            score_sum += final_score / num
            acc_sum += acc / num
            txt_res.append("dataset is {}, acc is {}, score is {}".format(FILE, acc / num, final_score / num))
            f=open(save_path,"w")
            for line in txt_res:
                f.write(line+'\n')
            f.write(str(score_sum / 15) +'\n')
            f.write(str(acc_sum / 15) +'\n')
            f.close()


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
