import argparse
import os
import sys
import shutil 
import json
import fire
import gradio as gr
import requests
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from commonsenseqa_prompter import CommonsensePrompter
from transformers.generation import utils
from rouge import Rouge
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
            max_length=5000,
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

    
    def generate_and_tokenize(data_point):
        #print(data_point)
        tokenized_full_prompt = tokenize(data_point)
        
        tokenized_full_prompt["input_ids"] = torch.IntTensor(tokenized_full_prompt["input_ids"])
        #print(tokenized_full_prompt)
        return tokenized_full_prompt
    
    # testing code for readme
    from nltk.translate.bleu_score import sentence_bleu
    file = open('./data/truthfulqa/truthfulqa.json', 'r', encoding='utf-8')
    lines = file.read()
    lines = json.loads(lines)
    file.close()
    f1_score = 0
    rouge_1 = 0
    rouge_2 = 0
    bleu_score = 0
    rouge = Rouge()
    l = 0
    for line in tqdm(lines):
        prompt_key = '''Below is an instruction that describes a task, paired with an input that provides further context. \n\n'''                
        prompt_key += "\n\n### Input:"
        prompt_key += line["question"]
        prompt_key += "Please answer the question simply."
        end_t = "\n\n### Response:\n"
        prompt_key += end_t
        tokenized_full_prompt = generate_and_tokenize(prompt_key)
        
        kwargs = {
            "words_ents_list": [tokenized_full_prompt["words_ents_list"]],
            "words_subtoken_map": [tokenized_full_prompt["words_subtoken_map"]]
        }
        result, _ = evaluate(tokenized_full_prompt["input_ids"], temperature=0.1, top_p=0.75, 
                    top_k=40, num_beams=4, max_new_tokens=256, stream_output=False, **kwargs)
        result = result.replace("</s>", "")
        r = rouge.get_scores([line["answer"]], [result])[0]
        f1 = r["rouge-l"]["f"]
        f1_score += f1
        f2 = r["rouge-1"]["f"]
        rouge_1 += f2
        f3 = r["rouge-2"]["f"]
        rouge_2 += f3
        answer = line["answer"].split()
        res = result.split()
        bleu = sentence_bleu([answer], res)
        bleu_score += bleu
        print(f"result is :{result}")
        print(f"answer is : " + line["answer"])
        print(f"bleu is {bleu}, rougel is {f1}")
        #print(bleu, f1)
        l += 1
    f1_score /= l
    rouge_1 /= l
    rouge_2 /= l
    bleu_score /= l
    print(f1_score)
    print(rouge_1)
    print(rouge_2)
    print(bleu_score)
    

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