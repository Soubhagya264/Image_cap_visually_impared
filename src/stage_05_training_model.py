from utils.all_utils import read_yaml,create_directory
from utils.callbacks import get_callbacks
from utils.data_management import data_generator
import argparse
import pandas as pd
import os
import shutil
import numpy as np
import ast
from tqdm import tqdm
import logging
from pickle import dump, load

logging_str="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')

def train_model(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    artifacts=config['artifacts']
    
    train_features_path=os.path.join(artifacts["DATA_DIR"],artifacts["PICKLE"],artifacts["TRAIN_FEATURES_FILE_NAME"])
    train_features=load(open(train_features_path,"rb"))
    file=open("prepaired_data/vocab_size_dir/vocab_size.txt","r")
    vocab_size=int(file.read())
    file.close()

    file=open("prepaired_data/max_length_dir/max_length.txt","r")
    max_length=int(file.read())
    file.close()
    
    file=open("prepaired_data/wordtoix_dir/wordtoix.txt","r")
    wordtoix=ast.literal_eval(file.read())  
    file.close()
    
    file=open("prepaired_data/train_descriptions_dir/train_descriptions.txt","r")
    train_descriptions=ast.literal_eval(file.read())  
    file.close()
    
    X1,X2,y=data_generator(train_descriptions,train_features, wordtoix, max_length,vocab_size)  
     
    
    glove_dir=os.path.join(artifacts["DATA_DIR"],artifacts["GLOVE_DIR"],artifacts["GLOVE_FILE"])
    f=open(glove_dir,encoding="utf-8")
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_dim = 200
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args=args.parse_args()
    try:
        logging.info("\n >>>>>>>>>> stage five started")
        train_model(config_path=parsed_args.config,params_path=parsed_args.params) 
        logging.info("stage five completed !! model trained \n >>>>>>>>>>>>")  
    except Exception as e:
        logging.exception(e)