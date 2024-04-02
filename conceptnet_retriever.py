import unicodedata
import numpy as np
import string
import logging
import nltk
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from nltk.corpus import wordnet
import os
from nltk.corpus import wordnet as wn
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords

class Conceptnet_retriever(object):
    def init(self, filepath):
        self.filepath = filepath
        self.concept2id = dict()
        self.stopwords = set(stopwords.words('english'))
        self.read_txt()

    def read_txt(self):
        f=open(self.filepath)
        concepts = f.readlines()  
        f.close()  # 关
        id = 0
        for concept in concepts:
            self.concept2id[concept.strip()] = id
            id += 1
    
    def lookup_concept_ids(self, text, no, args):
        #ents =  word_tokenize(text)   #分词
        ents =  text.split(" ")   #分词
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  
        tokens = []
        words_ents_lists = []
        for ent in ents:
            ent_name = []
            #if ent == "John" or ent == "probably" or en
            has_ent = False
            ent = ent.strip()
            if ent == "":
                continue
            if ent in interpunctuations:
                continue
            if ent.lower() in self.stopwords:
                continue 
            if ent in set(string.punctuation):
                #print('{} is punctuation, skipped!'.format(retrieve_token))
                continue
            words_ents_list =  [-1] * 5
            id_ent = self.concept2id.get(ent.lower(), -1)
            if id_ent == -1:
                continue
            #words_ents_list[0] = id_ent
            synonyms = []
            for syn in wordnet.synsets(ent):
                for lm in syn.lemmas():
                    synonyms.append(lm.name().lower())
            tmp = synonyms
            synonyms = list(set(tmp))
            #synonyms.sort(key = tmp.index) 
            synonyms.insert(0, ent.lower())
            i = 0
            j = 0
            while i < len(synonyms):
                id_ent = self.concept2id.get(synonyms[i].lower(), -1)
                if id_ent == -1:
                    i += 1
                    continue
                ent_name.append(synonyms[i].lower())
                i += 1
                words_ents_list[j] = id_ent
                has_ent = True
                j += 1
                if j >= 5:
                    break
            if has_ent:
                words_ents_lists.append(torch.IntTensor(words_ents_list))
                tokens.append(ent)
        return words_ents_lists, None, tokens 
    
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
    for syn in wordnet.synsets("table_tennis"):
        print(syn)
        for lm in syn.lemmas():
            print(lm.name())