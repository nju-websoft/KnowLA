"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union
from KGRetriever import KGRetriever, WordnetRetriever
from conceptnet_retriever import Conceptnet_retriever
import torch
import requests
import random
from nltk.corpus import wordnet

def get_atomic_dict():
    import csv
    path_list = ['train.tsv', 'dev.tsv', 'test.tsv']
    atomic_dict = dict()
    prediate = {'AtLocation':' is at the location of ', 'ObjectUse':' is used to ', 'HasSubEvent':' has the subevent of ', 'xNeed':' need ', 'CapableOf':' is capable to '}
    for subpath in path_list:
        with open('/data1/xdluo/alpaca-lora-main/atomic/' + subpath) as f:
            tsvreader = csv.reader(f, delimiter='\t')
            for line in tsvreader:
                if not line[1] in prediate.keys():
                    continue
                result_list = atomic_dict.get(line[0], [])
                result_list.append(line[0] + prediate[line[1]] + line[2])
                atomic_dict[line[0]] = result_list
    return atomic_dict

class CommonsensePrompter(object):
    __slots__ = ("template", "_verbose", "prefix_length", "retrievers", "tokens", "atomic_dict", "kg")

    def __init__(self, tokenizer, kg, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        result = tokenizer(
            "",
            truncation=True,
            max_length=2000,
            padding=False,
            return_tensors=None,
        )
        self.kg = kg
        if kg == 'wn':
        #self.atomic_dict = get_atomic_dict()
            self.retrievers = WordnetRetriever("./data/kgs/")
        elif kg == 'wi':
            self.retrievers = Conceptnet_retriever()
            self.retrievers.init("./data/kgs/wikidata/wiki_entity_name.txt")
        elif kg == 'cn':
            self.retrievers = Conceptnet_retriever()
            self.retrievers.init("./data/kgs/conceptnet/concept.txt")
        self.prefix_length = len(result["input_ids"])
        self.tokens = None
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        choices,
        question,
    ) -> str:
        res = '''Below is an instruction that describes a task, paired with an input that provides further context. Choose a correct answer that appears in the candidate answers.'''
        res += "\n\n### Input:"
        res += question
        end_t = "\n\n### Response:\n"
        res += end_t
        
        #print(res)
        return res
    
    def generate_KG_prompt(self, text):
        result = dict()
        full_prompt, _, result_list = self.generate_triple_from_atomic(
            text["question"]["stem"]
        )
        result['knowledge'] = result_list
        #print(result)
        return result
    
    def generate_wordnet(self, a, question, b):
        tokens = clean_text(question)
        template = "{} and {} are synonymous."
        result_list = []
        for token in tokens:
            synonyms = []
            for syn in wordnet.synsets(token):
                for lm in syn.lemmas():
                    tmp = lm.name().replace("-", " ")
                    tmp = tmp.replace("_", " ")
                    synonyms.append(tmp)
            new_token = token.replace("_", " ")
            synonyms = list(set(synonyms))
            for syn in synonyms[:4]:
                if syn == new_token:
                    continue
                result_list.append(template.format(new_token, syn))
        return None, None, result_list

    def generate_triple_from_KG(self, text):
        result = dict()
        full_prompt, _, result_list = self.generate_prompt_prediction(
            text["question"]["choices"],
            text["question"]["stem"],
            text["answerKey"],
        )
        a = result.split('\n')[: 5]
        a = [tmp.strip()[3: ] for tmp in a]
        #result['convert_prompt'] = full_prompt
        result['knowledge'] = result_list
        #print(result)
        return result
    
    def generate_triple_from_atomic(self, text):
        result_list = []
        from rake_nltk import Rake
        r = Rake()
        r.extract_keywords_from_text(text)
        ts = r.get_ranked_phrases()
        for word in ts:
            prompt = self.atomic_dict.get(word, None)
            if prompt is not None:
                result_list += prompt
        #print(result)
        return None, None, result_list

    def generate_csqa_prompt(
        self,
        choices,
        question,
        answer = None,
        knowledge = None,
        h = 1
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = '''Below is an instruction that describes a task, paired with an input that provides further context. Choose a correct answer that appears in the question.'''
        res += "\n\n### Input:"
        res += question
        res += " {know}"
        answer_text = ""
        #tokens = []
        # tokens = clean_text(question)
        #print(tokens)
        for i, choice in enumerate(choices):
            #print(choice)
            ans = choice["label"]
            #tokens.append(convert_choice(choice["text"]))
            if answer is not None and ans == answer:
                answer_text = choice["text"]

        #example = "street"
        #print(tokens)
        end_t = "\n\n### Response:\n"
        res += end_t
        if knowledge is None:
            knowledge = []
        for i in range(len(knowledge)):
            try:
                knowledge[i] = knowledge[i].replace("</s><s>", "")
            except:
                pass
        if h == 1:
            knowledge.insert(0, "")
            res = [res.format(know = know) for know in knowledge]
        else:
            tmp = []
            v = int(len(knowledge) / h)
            tmp.append(res.format(know = ""))
            for i in range(0, v * h, h):
                tmp.append(res.format(know = ' '.join(knowledge[i:i+h])))
            res = tmp 
        '''
        result_list = generate_prompts_from_KG(tokens)
        if len(result_list) != 0:
            res += " These information may be useful: "
            for prompt in result_list:
                res += prompt
                res += ';'
        '''
        #print(res)
        return res, answer_text


    def generate_siqa_prompt(
        self,
        choices,
        question,
        answer = None,
        knowledge = None,
        h = 1
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        #res = '''Below is an instruction that describes a task, paired with an input that provides further context. Choose a correct answer that appears in the question.'''
        res = '''Below is an instruction that describes a task, paired with an input that provides further context. Choose a correct answer that appears in the candidate answers.'''
        res += "\n\n### Input:"
        res += question
        res += " {know}"
        #res += " Candidate answers are"
        answer_text = ""
        #tokens = []
        # tokens = clean_text(question)
        #print(tokens)
        end_t = "\n\n### Response:\n"
        res += end_t
        if knowledge is None:
            knowledge = []

        if h == 1:
            #knowledge.insert(0, "")
            ori = res.format(know = "")
            if len(knowledge) == 0 or knowledge == question.lower():
                res = [ori]
            else:
                res = [res.format(know = "These information may be useful: " + know) for know in knowledge[:1]]
                #res = [res.format(know = know) for know in knowledge[:1]]
                res.insert(0, ori)
        # Conceptnet是knowledge[: 2], wordnet不需要
        else:
            tmp = []
            v = int(len(knowledge[:2]) / h)
            tmp.append(res.format(know = ""))
            for i in range(0, v * h, h):
                tmp.append(res.format(know = "These information may be useful: " + ' '.join(knowledge[i:i+h])))
            res = tmp 
        print(res)
        return res, answer
    
    def generate_input_llama_format(
        self,
        question,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = '''Below is an instruction that describes a task, paired with an input that provides further context. Choose a correct answer that appears in the candidate answers.'''
        res += "\n\n### Input:"
        res += question
        end_t = "\n\n### Response:\n"
        res += end_t
        return res
    
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()

    def get_mapping_ids(self, text, text_tokenized, tokenizer):
        text = text.split("### Input:")[1].strip().split(self.template["response_split"])[0]
        #text = text.strip().split(self.template["response_split"])[0]
        text_tokenized = tokenizer.convert_ids_to_tokens(text_tokenized)
        #print("pure text is {}".format(text_tokenized))
        args_dict = dict()
        args_dict['max_ents'] = 5
        args_dict['do_lower_case'] = True
        args_dict['no_stopwords'] = True
        args_dict['ignore_length'] = 0
        args_dict['is_filter'] = True
        args_dict['is_lemma'] = True
        args_dict['is_clean'] = True
        args_dict['is_morphy'] = True
        #example = "street"
        if self.kg == 'wn':
            words_ents_list, wdnet_name_dict, tokens = \
            self.retrievers.lookup_concept_ids(text, None, **args_dict)
        else:
            words_ents_list, wdnet_name_dict, tokens = \
            self.retrievers.lookup_concept_ids(text, None, None)
        #print(wdnet_name_dict)
        #for i, token in enumerate(tokens):
        #    tokens[i] = tokenizer.tokenize(token)[0][1: ]
        tmp = tokens
        self.tokens = tokens
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
        #assert len(words_subtoken_map) == len(tokens)
        if len(words_subtoken_map) != len(tokens):
            pass
            #print("-------------------------------------------------------------")
            #print(words_subtoken_map)
            #print(tokens)
        return words_ents_list, words_subtoken_map
    
def generate_prompts_from_KG(tokens):
    result_list = []
    for token in tokens:
        obj = requests.get('http://api.conceptnet.io/c/en/'+token).json()
        try:
            total = len(obj['edges'])
            for i in range(total):
                max_edge = obj['edges'][i]
                max_weight = obj['edges'][i]['weight']
                result_list += generate_prompt(max_edge)
        except:
            pass
    return result_list

def clean_text(text):
    from rake_nltk import Rake
    r = Rake()
    r.extract_keywords_from_text(text)
    ts = r.get_ranked_phrases()
    for th, t in enumerate(ts):
        t = t.split()
        ts[th] = "_".join(t)
    return ts

def convert_choice(choice):
    choice = choice.split()
    choice = "_".join(choice)
    return choice

def generate_prompt(edge):
    result = []
    if edge is None or edge["surfaceText"] is None:
        return []
    edge = edge["surfaceText"]
    if "translation" in edge:
        return []
    edge = edge.replace('[', '')
    edge = edge.replace(']', '')
    result.append(tmp)
    return result
    
if __name__ == "__main__":
    token = "perjury"
    for syn in wordnet.synsets("animal"):
        print(syn)
        for lm in syn.lemmas():
            print(lm)
        #    tmp = lm.name().replace("-", " ")
        #    tmp = tmp.replace("_", " ")
        #    print(tmp)
    #print(r.get_ranked_phrases_with_scores())