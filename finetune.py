import os
import sys
from typing import List

import fire
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import transformers
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass, field
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import shutil
from typing import Optional, Dict, Sequence
from peft import (
    AdaLoraConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from commonsenseqa_prompter import CommonsensePrompter
from utils.prompter import Prompter



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels, words_ents_list, words_subtoken_map= tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "words_ents_list", "words_subtoken_map"))
        input_ids = [torch.IntTensor(input_id) for input_id in input_ids]
        words_ents_list = [torch.IntTensor(ent) for ent in words_ents_list]
        words_subtoken_map = [torch.IntTensor(ent) for ent in words_subtoken_map]
        labels = [torch.LongTensor(label) for label in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            words_ents_list = words_ents_list,
            words_subtoken_map = words_subtoken_map
        )


def compute_metrics(pred):
    #print("---------------------------")
    predict_res = torch.Tensor(pred.predictions[0]) # size：[验证集样本量, label的token长度, vocab大小]
    #print(predict_res.size())
    pred_ids = predict_res.argmax(dim=2)
 
    ## 2.处理 pred.label_ids
    labels_actual = torch.LongTensor(pred.label_ids)
    
    ## 3.计算accuracy
    total_num = labels_actual.shape[0]
    acc = torch.sum(torch.all(torch.eq(pred_ids, labels_actual), dim=1))/total_num
    return {'accuracy': acc}


def train(
    # model/data params
    base_model: str = './llama2_7B',  # the only required argument
    data_path: str = "alpaca_data_cleaned.json",
    output_dir: str = "",
    embedding_path: str = "./data/kgs/conceptnet/ent.npy",
    # training hyperparams
    batch_size: int =128,
    kg_path: str = "./data/kgs/conceptnet/concept.txt",
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 1221,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    #device_map = "balanced"
    device_map={"": "mps"},
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print("--------------------------")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    shutil.copyfile(base_model + "/modeling_llama.py", os.path.abspath(sys.modules[LlamaForCausalLM.__module__].__file__), follow_symlinks=True)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=True,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    #tokenizer.padding_side = "left"  # Allow batched inference
    prompter = Prompter(tokenizer, kg_path, prompt_template_name)
    if lora_r == 32:
        warmup_steps = 400
    else:
        warmup_steps = 200
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
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

    def generate_and_tokenize_prompt(data_point):
        #print(data_point)
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_user_prompt["input_ids"] = torch.IntTensor(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            tokenized_user_prompt["labels"] = torch.IntTensor(tokenized_user_prompt["labels"])
        tokenized_full_prompt["input_ids"] = torch.IntTensor(tokenized_full_prompt["input_ids"])
        tokenized_full_prompt["labels"] = torch.IntTensor(tokenized_full_prompt["labels"])
        #print(tokenized_full_prompt)
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    print(model)
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=8)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt, num_proc=8)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=8)
        val_data = None
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        #compute_metrics = compute_metrics,
        args=transformers.TrainingArguments(
            remove_unused_columns = False,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps, 
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            #fp16=True,
            logging_steps=50,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=5000000 if val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        #data_collator=transformers.DataCollatorForSeq2Seq(
        #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #),
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )
    #model.config.use_cache = True

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=(resume_from_checkpoint or False))

    model.save_pretrained(output_dir)
    #save_KG_module(model.base_model.model.model.layers, output_dir)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
