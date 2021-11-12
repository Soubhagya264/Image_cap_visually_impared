
import numpy as np
from numpy import array
import string
from PIL import Image
import argparse
import pandas as pd
import os
import glob
from tqdm import tqdm
import logging

def load_description(text):
    logging.info("load description function called  and started loading description")
    mapping = dict()
    for line in text.split("\n"):
        token = line.split("\t")
        if len(line) < 2:  
            continue
        img_id = token[0].split('.')[0] 
        img_des = token[1]              
        if img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_des)
    logging.info(" Ended loading description")     
    return mapping

def clean_description(desc):
    logging.info("started cleaning the description")
    for key, des_list in desc.items():
        for i in range(len(des_list)):
            caption = des_list[i]
            caption = [ch for ch in caption if ch not in string.punctuation]
            caption = ''.join(caption)
            caption = caption.split(' ')
            caption = [word.lower() for word in caption if len(word)>1 and word.isalpha()]
            caption = ' '.join(caption)
            des_list[i] = caption
    logging.info("Description cleaned")
def to_vocab(desc):
    logging.info("Extracting Vocabulary from the sentences")
    words = set()
    for key in desc.keys():
        for line in desc[key]:
            words.update(line.split())
    logging.info("vocabulary Extracted")        
    return words  

def prepare_train_images(images,img,train_images):
    logging.info("started prepairing  train_img") 
    train_img=[]
    for im in img:
        if (im[len(images):] in train_images):
            train_img.append(im)
           
    logging.info("prepared  train_img")        
    return train_img        

def load_clean_descriptions(des, dataset):
    logging.info("started cleaning description for training images")
    dataset_des = dict()
    for key, des_list in des.items():
        if key+'.jpg' in dataset:
            if key not in dataset_des:
                dataset_des[key] = list()
            for line in des_list:
                desc = 'startseq ' + line + ' endseq'
                dataset_des[key].append(desc)
    logging.info("description cleaned for training images")            
    return dataset_des        

def train_captions(train_descriptions):
    logging.info("Extracting Captions")
    all_train_captions=[]
    for key, val in train_descriptions.items():
        for caption in val:
            all_train_captions.append(caption)
    logging.info("Captions Extracted")        
    return all_train_captions     

def find_vocab_size_max_len(vocab,all_train_captions):
        logging.info("started finding vocab size and maximum length")
        vocabulary = vocab
        threshold = 10
        word_counts = {}
        for cap in all_train_captions:
                for word in cap.split(' '):
                    word_counts[word] = word_counts.get(word, 0) + 1
        vocab = [word for word in word_counts if word_counts[word] >= threshold]         
        ixtoword = {}
        wordtoix = {}
            
        ix = 1
        for word in vocab:
                wordtoix[word] = ix
                ixtoword[ix] = word
                ix += 1
        max_length = max(len(des.split()) for des in all_train_captions)
        vocab_size = len(ixtoword) + 1
        logging.info("finding vocab_size max_len is completed")
        return vocab_size ,max_length,wordtoix       