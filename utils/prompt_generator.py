import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import heapq
import unicodedata
import numpy as np
import string
import logging
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import numpy
import nltk
from transformers import LlamaForCausalLM, LlamaTokenizer
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import sys
sys.path.append("/data1/xdluo/contriever/")
from src.contriever import Contriever

conceptnet = {'gloss': 'is', 'IsA':'is a', 'Synonym':'is a synonym of', 'CausesDesire':'causes desire', 'DefinedAs':'is defined as', 
              'MadeOf':'is made of', 'Causes':'causes', 'HasProperty':'has property of', 'dbpedia':'in dbpedia is', 
              'MannerOf':'has manner of', 'HasContext':'has context of', 'SimilarTo':'is similar to', 'CapableOf':'is capable to', 
              'UsedFor':'is used to', 'NotDesires':'not desires', 'HasFirstSubevent':'has first subevent', 'InstanceOf':'has instance of', 
              'EtymologicallyDerivedFrom':'is etymologically derived from', 'ReceivesAction':'receives action of', 'Desires':'desires', 
              'HasSubevent':'has subevent of', 'LocatedNear':'is located near', 'DerivedFrom':'is derived from', 'MotivatedByGoal':'is motivated by goal of', 
              'HasA':'has a', 'HasLastSubevent':'has last subevent of', 'AtLocation':'is at location of', 'PartOf':'has part of', 'DistinctFrom':'is distinct from', 
              'HasPrerequisite':'has prerequisite of', 'NotCapableOf':'is not capable to', 'SymbolOf':'has symbol of', 'Antonym':'has antonym of', 'RelatedTo':'is related to', 
              'NotHasProperty':'not has property of', 'FormOf':'is form of', 'CreatedBy':'is created by', 'Entails':'entails', 'EtymologicallyRelatedTo':'is etymologically related to'}

tokenizer = AutoTokenizer.from_pretrained('/data1/xdluo/mpnet')
model = AutoModel.from_pretrained('/data1/xdluo/mpnet')
#tokenizer = AutoTokenizer.from_pretrained('/data1/xdluo/contriever/model')
#model = AutoModel.from_pretrained('/data1/xdluo/contriever/model')
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def filter_prompts(text, knowledge):
    #result = dict()
    sentences = []
    if len(knowledge) == 0:
        return text, []
    sentences.append(text)
    sentences += knowledge
    index = get_embeddings(sentences)
    tmp = [knowledge[i] for i in index]
    #print(result)
    return tmp, index

def save_embeds(encoded_input, th):
    device = torch.device("cpu")
    with torch.no_grad():
            model_output = model(**encoded_input.to(device))
        # Perform pooling
    print("finish")
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    numpy.save(f"/data1/xdluo/alpaca-lora-main/utils/wikidata/{th}.npy", sentence_embeddings.cpu().data.numpy())
    
def save_as_embeds(sentences):
    print(model.device)
    th = 17
    for i in tqdm(range(750000, len(sentences), 50000)):
        encoded_input = tokenizer(sentences[i:i+50000], padding=True, truncation=True, return_tensors='pt')
        save_embeds(encoded_input, th)
        th += 1


def get_index(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    device = torch.device("cuda")
    with torch.no_grad():
        model_output = model(**encoded_input.to(device))
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    candidates = torch.tensor(numpy.load(r"/data1/xdluo/alpaca-lora-main/utils/concept.npy"))
    socres = torch.matmul(sentence_embeddings, candidates.T).tolist()
    max_number = heapq.nlargest(5, socres) 
    max_index = []
    for t in max_number:
        index = socres.index(t)
        max_index.append(index)
        socres[index] = 0
    return max_index

def get_embeddings(sentences):
    device = torch.device("cpu")
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input.to(device))
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).cpu()
    ori_embedding = sentence_embeddings[0]
    candidates_embeddings = sentence_embeddings[1:]
    socres = torch.matmul(ori_embedding, candidates_embeddings.T).tolist()
    max_number = heapq.nlargest(5, socres) 
    max_index = []
    for t in max_number:
        index = socres.index(t)
        if socres[index] < 0.3:
            break
        max_index.append(index)
        socres[index] = 0
    return max_index


def generate(tokenizer, model, prompt):
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
            r1 = output.split("### Response:")[-1].strip()
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
    tokenized_full_prompt = generate_and_tokenize(prompt)
    kwargs = {
        "words_ents_list": [],
        "words_subtoken_map": []
    }
    #print("the answer is : {}".format(answer))
    result, _ = evaluate(tokenized_full_prompt["input_ids"], temperature=0.1, top_p=0.75, 
                top_k=40, num_beams=4, max_new_tokens=256, stream_output=False, **kwargs)
    return result
        

class prompt_generator(object):
    '''
        Two ways to index triples
    '''
    def init(self, path1, path2, mode):
        self.entity_name_path = path1
        self.triples_path = path2
        self.mode = mode
        #self.filepath = "/data1/xdluo/alpaca-lora-main/data/kgs/conceptnet/ent_name.txt"
        self.concept_list = []
        self.concept2tri = dict()
        base_model = "/data1/xdluo/llama2_7B/"
        kg_infused_weight = "/data1/xdluo/k-lora/llama2-lora"
        #kg_infused_weight = "/data1/xdluo/alpaca-lora-main/lora-alpaca-kg-wn"
        lora_weights = kg_infused_weight
        '''
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
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
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        '''
        self.stopwords = set(stopwords.words('english'))
        self.read_text()

    def read_text(self):
        f=open(self.entity_name_path)
        concepts = f.readlines()
        f.close()  # 关
        relation_set = set()
        id = 0
        for concept in concepts:
            self.concept_list.append(' '.join(concept.split('_')))
            id += 1
        #save_as_embeds(self.concept_list)
        f=open(self.triples_path)
        triples = f.readlines()  
        f.close()  # 关
        id = 0
        for triple in triples:
            if self.mode == "wn":
                triple = triple.strip().split('\t')
            else:
                triple = triple.strip().split(' ')
            relation_set.add(triple[1])
            t = [' '.join(triple[0].split('_')), ' '.join(conceptnet.get(triple[1], triple[1]).split('_')), ' '.join(triple[2].split('_'))]
            tmp = self.concept2tri.get(triple[0].lower(), [])
            tmp.append(t)
            self.concept2tri[triple[0].lower()] = tmp
            if triple[0] != triple[2]:
                tmp = self.concept2tri.get(triple[2].lower(), [])
                tmp.append(t)
                self.concept2tri[triple[2].lower()] = tmp

    def get_knowledge(self, text):
        result = dict()
        #print(result)
        from rake_nltk import Rake
        r = Rake()
        r.extract_keywords_from_text(text["RawQuestion"].lower())
        ts = r.get_ranked_phrases()
        tmp, triple_list = self.generate_triples(ts)
        knowledge, index = filter_prompts(text["RawQuestion"].lower(), tmp)
        #return knowledge
        #return knowledge
        if len(index) == 0:
            knowledge = []
        else:
            knowledge = [tmp[i] for i in index]
            #knowledge = self.generate_text_llama(knowledge)
        return knowledge

    def get_knowledge_text(self, text):
        result = dict()
        #print(result)
        knowledge, index = filter_prompts(text["question"].lower(), [knowledge.lower() for knowledge in text["knowledge"]])
        print(knowledge)
        return knowledge
    
    def generate_triples(self, words):
        result_list = []
        triple_list = []
        for word in words:
            word = '_'.join(word.split())
            triples = self.concept2tri.get(word, None)
            if triples is None:
                continue
            if len(triples) > 5:
                triples = triples[:5]
            triple_list += triples
            for triple in triples:
                prompt = ' '.join(triple)
                result_list.append(prompt)
        return result_list, triple_list
    
    def generate_text_llama(self, results_list):
        result = []
        prompt_triple = '''Please convert this triple into a single short sentence. Do not insert any other information or commentary. {}, {}, {}\n\n\n### Response:\n'''
        for triple in results_list:
            print(triple)
            tmp = prompt_triple.format(triple[0], triple[1], triple[2])
            output = generate(self.tokenizer, self.model, tmp).replace("</s>", "")
            result.append(output)
        return result

    
def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

if __name__ == "__main__":
    import json
    t = prompt_generator()
    name_list = [r'/data1/xdluo/alpaca-lora-main/data/kgs/conceptnet/ent_name.txt', r'/data1/xdluo/alpaca-lora-main/data/kgs/wikidata/wiki_entity_name.txt', 
                 r'/data1/xdluo/alpaca-lora-main/data/kgs/wordnet/wordnet_entity_name.txt']
    triple_list = [r'/data1/xdluo/alpaca-lora-main/data/kgs/conceptnet/train_text.txt', r'/data1/xdluo/alpaca-lora-main/data/kgs/wikidata/wiki_triples_text.txt',
                  r'/data1/xdluo/alpaca-lora-main/data/kgs/wordnet/wordnet_text.txt' ]
    tag = ["cn", "wi", "wn"]
    #t.init(r'/data1/xdluo/alpaca-lora-main/data/kgs/conceptnet/ent_name.txt', r'/data1/xdluo/alpaca-lora-main/data/kgs/conceptnet/train_text.txt', 'symbol')
    
    #t.get_knowledge('A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?')
    #t.init(r'/data1/xdluo/alpaca-lora-main/data/kgs/wordnet/wordnet_entity_name.txt', r'/data1/xdluo/alpaca-lora-main/data/kgs/wordnet/wordnet_text.txt', 'symbol')
    #data_files = {"train": "/data1/xdluo/alpaca-lora-main/data/SIQA/devc.json"
    for i in range(1, 2):
        t.init(name_list[i], triple_list[i], tag[i])
        result = []
        file = open(f'/data1/xdluo/alpaca-lora-main/data/wqsp/test.json', 'r', encoding='utf-8')
        #file = open(f'/data1/xdluo/contriever/output/wikipedia-dev.json', 'r', encoding='utf-8')
        #file = open(f'/data1/xdluo/alpaca-lora-main/data/commonsenseqa/test.jsonl', 'r', encoding='utf-8')
        #file = open('/data1/xdluo/alpaca-lora-main/data/SIQA/devc.json', 'r', encoding='utf-8')
        #lines = file.readlines()
        lines = file.read()
        lines = json.loads(lines)
        file.close()
        #json_file_path = f'/data1/xdluo/alpaca-lora-main/data/commonsenseqa/csqa_{tag[i]}_filtered_contriever.json'
        json_file_path = f'/data1/xdluo/alpaca-lora-main/data/wqsp/test2.json'
        json_file = open(json_file_path, mode='w')
        for line in tqdm(lines["Questions"]):
            #line = json.loads(line)
            print(line)
            knowledge = t.get_knowledge(line)
            line.update({'knowledge': knowledge})
            result.append(line)
        json.dump(result, json_file, indent=4)