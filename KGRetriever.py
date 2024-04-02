import unicodedata
import numpy as np
import string
import logging
import nltk
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from nltk.corpus import wordnet as wn
import pickle
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
POS_LIST = ['n', 'v', 'a', 'r']


class KGRetriever(object):
    def __int__(self):
        self.filepath = ""
        self.max_concept_length = 0
        self.name = "general_kg_retriever"
        self.concept_embedding_mat = [[]]

    def to_dict(self):
        output = dict()
        output['name'] = self.__dict__['name']
        output['max_concept_length'] = self.__dict__['max_concept_length']
        output['concept_vocab_size'] = self.get_concept_vocab_size()
        output['concept_embed_size'] = self.get_concept_embed_size()
        output['file_path'] = self.__dict__['filepath']
        return output

    def get_embedding_mat(self):
        return self.concept_embedding_mat
    def get_concept_embed_size(self):
        return self.concept_embedding_mat.shape[1]
    def get_concept_vocab_size(self):
        return self.concept_embedding_mat.shape[0]
    def get_concept_max_length(self):
        return self.max_concept_length
    def update_max_concept_length(self, num):
        self.max_concept_length = max(self.max_concept_length, num)

    def lookup_concept_ids(self, tokenization_info, **kwargs):
        raise NotImplementedError

    def id2concept_check(self, entity_id):
        return self.id2concept[entity_id]

def read_concept_embedding(embedding_path):
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    fin.close()
    return id2concept, concept2id, embedding_mat

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




# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright 2019 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class WordnetRetriever(KGRetriever):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.name = "wordnet"
        self.max_concept_length = 0

        concept_embedding_path = os.path.join(filepath, "wn_concept2vec.txt")
        self.id2concept, self.concept2id, self.concept_embedding_mat = read_concept_embedding(
            concept_embedding_path)

        self.offset_to_wn18name_dict = {}
        fin = open(os.path.join(filepath, 'wordnet-mlj12-definitions.txt'))
        for line in fin:
            info = line.strip().split('\t')
            offset_str, synset_name = info[0], info[1]
            self.offset_to_wn18name_dict[offset_str] = synset_name
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        #logger.info('Finish loading wn18 definition file.')

        self.pos = POS_LIST

        self.wn18_dir = os.path.join(self.filepath, "wn18/text")
        self.wn18_path = os.path.join(self.wn18_dir, "full.txt")

        self.synset_name_set_path = os.path.join(self.wn18_dir, "synset_name.txt")
        if not os.path.exists(self.synset_name_set_path):
            self.synset_name_set = self.create_synset_name_set()
        else:
            with open(self.synset_name_set_path, "rb") as fp:
                self.synset_name_set = set(pickle.load(fp))

        repeated_id_path = os.path.join(self.filepath, "repeated_id.npy")

        self.repeated_id = np.load(repeated_id_path, allow_pickle='TRUE').item()

        self.conceptids2synset = {}

    def create_entire_wn18_graph(self):
        wn18_full = open(self.wn18_path, 'a')

        wn18_train = open(os.path.join(self.wn18_dir, "train.txt"), 'r')
        for line in wn18_train.readlines():
            wn18_full.writelines(line, )
        wn18_train.close()

        wn18_valid = open(os.path.join(self.wn18_dir, "valid.txt"), 'r')
        for line in wn18_valid.readlines():
            wn18_full.writelines(line, )
        wn18_valid.close()

        wn18_test = open(os.path.join(self.wn18_dir, "test.txt"), 'r')
        for line in wn18_test.readlines():
            wn18_full.writelines(line, )

        wn18_test.close()
        wn18_full.close()

    def create_synset_name_set(self):
        synset_name_set = set()
        if not os.path.exists(self.wn18_path):
            self.create_entire_wn18_graph()
        wn18_full = open(os.path.join(self.wn18_dir, "full.txt"), 'r')

        for line in wn18_full.readlines():
            src, relation, dst = line.strip().split("\t")
            if src not in synset_name_set:
                synset_name_set.add(src)
            if dst not in synset_name_set:
                synset_name_set.add(dst)
        wn18_full.close()

        synset_name_list = list(synset_name_set)
        with open(self.synset_name_set_path, 'wb') as fp:
            pickle.dump(synset_name_list, fp)
        return synset_name_set

    def lookup_concept_ids_single(self, text, ori_to_tok_map, tok_num, tolower, no_stopwords, ignore_length, tokenizer, is_filter,
                                  is_lemma, is_clean, is_morphy, query=True, query_size=0):

        words = text.split(" ")
        word_to_ori_map = []
        is_begin = True
        conceptids2synset = {}

        for i, c in enumerate(text):
            if is_begin:
                word_to_ori_map.append([i])
            if c == " ":
                is_begin = True
            else:
                if is_begin:
                    is_begin = False
                else:
                    word_to_ori_map[-1].append(i)
        #print(word_to_ori_map)
        # logger.info("text: {}".format(words))
        # # remove stop followed by word (uncomment when using copa dataset)
        # if words[-1][-1] == "." or words[-1][-1] == ",":
        #     words[-1] = words[-1][:-1]
        words_ent_list = []
        token = []
        for i, word in enumerate(words):
            word_ent_list = [-1] * tok_num
            retrieve_token = run_strip_accents(word.lower()) if tolower else word
            if retrieve_token in set(string.punctuation):
                #print('{} is punctuation, skipped!'.format(retrieve_token))
                continue
            if no_stopwords and retrieve_token in self.stopwords:
                #print('{} is stopword, skipped!'.format(retrieve_token))
                continue
            if ignore_length > 0 and len(retrieve_token) <= ignore_length:
                #print('{} is too short, skipped!'.format(retrieve_token))
                continue

            try:
                synsets = wn.synsets(retrieve_token)
                #print(synsets)
            except:
                #print("{} can't work in nltk".format(retrieve_token))
                synsets = []
            wn18synset_names = []
            if is_morphy:
                # logger.info("morphy match")
                morphy_set = self.get_morphy(retrieve_token)
                if retrieve_token not in morphy_set:
                    # logger.info("{} not in morphy_set{}".format(retrieve_token, morphy_set))
                    morphy_set.add(retrieve_token)
            else:
                # logger.info("exact match")
                morphy_set = None

            th = 0
            has_ents = False
            for synset in synsets:
                if is_filter and not self.is_in_full_wn18(synset):
                    continue

                if not is_lemma and not self.is_center_entity(synset, retrieve_token, morphy_set, is_morphy):
                    continue

                offset_str = str(synset.offset()).zfill(8)
                if offset_str in self.offset_to_wn18name_dict:
                    full_synset_name = self.offset_to_wn18name_dict[offset_str]

                    if is_clean and self.is_repeated(self.concept2id[full_synset_name]):
                        continue
                    if self.concept2id[full_synset_name] in conceptids2synset and conceptids2synset[self.concept2id[full_synset_name]] != synset:
                        #print("different wn object {} {} map to the same id {}".format
                        #               (conceptids2synset[self.concept2id[full_synset_name]], synset, self.concept2id[full_synset_name]))
                        if self.concept2id[full_synset_name] not in self.repeated_id:
                            self.repeated_id[self.concept2id[full_synset_name]] = [str(conceptids2synset[self.concept2id[full_synset_name]]), str(synset)]

                    wn18synset_names.append(full_synset_name)
                    has_ents = True
                    if th == 0:
                        token.append(word)
                    if th < tok_num:
                        word_ent_list[th] = (int(self.concept2id[full_synset_name]) - 1)
                    conceptids2synset[self.concept2id[full_synset_name]] = synset
                    th += 1
            if has_ents:
                words_ent_list.append(torch.IntTensor(word_ent_list))
            #print(conceptids2synset)
        #print(words_ent_list)
        #print(token)
        assert len(token) == len(words_ent_list)
        return words_ent_list, conceptids2synset, token

    def lookup_concept_ids(self, example, tokenizer, **kwargs):
        """
            :param tokenization_info:
            :param tokenizer_type:
            :return:

            find the concepts in wordnet, and add the ids to the corresponding tokens.
            """
        max_ents = 5
        do_lower_case = kwargs.pop("do_lower_case", False)
        no_stopwords = kwargs.pop("no_stopwords", False)
        ignore_length = kwargs.pop("ignore_length", 0)
        is_filter = kwargs.pop("is_filter")
        is_lemma = kwargs.pop("is_lemma")
        is_clean = kwargs.pop("is_clean")
        is_morphy = kwargs.pop("is_morphy")

        # tolower = not do_lower_case
        tolower = True

        doc_text = example

        words_ent_list, doc_conceptids2synset, token = \
            self.lookup_concept_ids_single(doc_text, None, max_ents, tolower,
                                           no_stopwords, ignore_length, tokenizer, is_filter=is_filter, is_lemma=is_lemma,
                                           is_clean=is_clean, is_morphy=is_morphy, query=False)


        return words_ent_list, doc_conceptids2synset, token

    def is_center_entity(self, entity, word, morphy_set, is_morphy):
        if len(str(entity).split("'")) == 3:
            tmp = str(entity).split("'")[1]
        else:
            tmp = str(entity).replace("')", "('").split("('")[1]

        # if is_filter and not self.is_in_full_wn18(tmp):
        #     return False

        tmp = tmp.split(".")
        if len(tmp) == 3:
            if is_morphy:
                return tmp[0] in morphy_set
            else:
                return tmp[0] == word
        else:
            tmp2 = ""
            for i, substring in enumerate(tmp):
                if i >= len(tmp)-2:
                    break
                tmp2 += substring
            if is_morphy:
                return tmp2 in morphy_set
            else:
                return tmp2 == word

    def is_in_full_wn18(self, synset_name):
        if len(str(synset_name).split("'")) == 3:
            tmp = str(synset_name).split("'")[1]
        else:
            tmp = str(synset_name).replace("')", "('").split("('")[1]

        return tmp in self.synset_name_set

    def get_morphy(self, lemma, check_exceptions=True):
        morphy_list = [form
                        for p in self.pos
                        for form in wn._morphy(lemma, p, check_exceptions)]
        return set(morphy_list)

    def is_repeated(self, entity_id):
        return entity_id in self.repeated_id

if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained("/data1/xdluo/llama2_7B")
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" 
    retrievers = WordnetRetriever("/data1/xdluo/alpaca-lora-main/data/kgs/")
    args_dict = dict()
    args_dict['do_lower_case'] = True
    args_dict['no_stopwords'] = True
    args_dict['ignore_length'] = 0
    args_dict['is_filter'] = True
    args_dict['is_lemma'] = False
    args_dict['is_clean'] = True
    args_dict['is_morphy'] = True
    example = "Where would you expect to find a pizzeria while shopping? chicago, street, little italy, food court and capital cities."
    #example = "street"
    query_kg_concept_ids, doc_kg_concept_ids, max_concept_length, query_conceptids2synset, doc_conceptids2synset = \
    retrievers.lookup_concept_ids(example, tokenizer, **args_dict)
