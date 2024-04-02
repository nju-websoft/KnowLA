"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union
import torch
from KGRetriever import WordnetRetriever
from conceptnet_retriever import Conceptnet_retriever

class Prompter(object):
    __slots__ = ("template", "_verbose", "prefix_length", "retrievers")

    def __init__(self, tokenizer, filepath, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = "/data1/xdluo/alpaca-lora-main/templates/alpaca.json"
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        result = tokenizer(
            self.template["prefix_no_input"],
            truncation=True,
            max_length=1000,
            padding=False,
            return_tensors=None,
        )
        self.retrievers = WordnetRetriever("/data1/xdluo/alpaca-lora-main/data/kgs/")
        #self.retrievers = Conceptnet_retriever()
        #self.retrievers.init(filepath)
        self.prefix_length = len(result["input_ids"])


        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

    def get_mapping_ids(self, text, text_tokenized, tokenizer):
        
        text = text.split("### Instruction:")[1].strip().split(self.template["response_split"])[0]
        text_tokenized = tokenizer.convert_ids_to_tokens(text_tokenized)
        #print("pure text is {}".format(text_tokenized))
        args_dict = dict()
        args_dict['max_ents'] = 3
        args_dict['do_lower_case'] = True
        args_dict['no_stopwords'] = True
        args_dict['ignore_length'] = 0
        args_dict['is_filter'] = True
        args_dict['is_lemma'] = True
        args_dict['is_clean'] = True
        args_dict['is_morphy'] = True
        #example = "street"

        words_ents_list, wdnet_name_dict, tokens = \
        self.retrievers.lookup_concept_ids(text, None, **args_dict)
        #print(wdnet_name_dict)

        #for i, token in enumerate(tokens):
        #    tokens[i] = tokenizer.tokenize(token)[0][1: ]
        tmp = tokens
        remove_index = 0
        words_subtoken_map = []
        word_subtoken_map = []
        j = self.prefix_length
    
        for i, token in enumerate(text_tokenized[j: ]):
                if text_tokenized[j + i].startswith("▁"):
                    text_tokenized[j + i] = text_tokenized[j + i][1: ]
        #print(text_tokenized[j: ])

        while j < len(text_tokenized):
            if remove_index < len(tokens) and (text_tokenized[j] == tokens[remove_index] or 
                                              (j + 1 < len(text_tokenized) and tokens[remove_index].startswith(text_tokenized[j] + text_tokenized[j + 1]))):
                word_subtoken_map.append(j)
                tokens[remove_index] = tokens[remove_index].replace(text_tokenized[j], "")
                j += 1
                while j < len(text_tokenized) and tokens[remove_index].startswith(text_tokenized[j]):
                    tokens[remove_index] = tokens[remove_index].replace(text_tokenized[j], "")
                    word_subtoken_map.append(j)
                    j += 1
                words_subtoken_map.append(torch.IntTensor(word_subtoken_map))
                remove_index += 1
                word_subtoken_map = []
            else:
                j += 1
        #print(words_subtoken_map)
        #assert len(words_subtoken_map) == len(tokens)
        if len(words_subtoken_map) != len(tokens):
            return [], []
        return words_ents_list, words_subtoken_map