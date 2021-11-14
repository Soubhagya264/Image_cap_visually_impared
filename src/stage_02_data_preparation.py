import numpy as np
from numpy import array
import string
from PIL import Image
from utils.all_utils import read_yaml,create_directory
from utils.text_img_data_preparation import load_description,clean_description,to_vocab,prepare_train_images,load_clean_descriptions,train_captions,find_vocab_size_max_len 
import shutil
import argparse
import pandas as pd
import os
import glob
from tqdm import tqdm
import logging
logging_str="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')


      

def data_preparation(config_path,params_path):
    config=read_yaml(config_path)
    param=read_yaml(params_path)
    artifacts=config["artifacts"]
    token_path=os.path.join(artifacts["DATA_DIR"],artifacts["TEXT_DATA"],artifacts["TOKEN_PATH_NAME"])
    text = open(token_path, 'r', encoding = 'utf-8').read()
    descriptions = load_description(text)
    clean_description(descriptions)
    vocab = to_vocab(descriptions)
    img_path=os.path.join(artifacts["DATA_DIR"],artifacts["IMAGE_DATA"])
    img=glob.glob(img_path+'/'+'*.jpg')
    
    train_path=os.path.join(artifacts["DATA_DIR"],artifacts["TEXT_DATA"],artifacts["TRAIN_PATH"])
    test_path=os.path.join(artifacts["DATA_DIR"],artifacts["TEXT_DATA"],artifacts["TEST_PATH"])
    train_images = open(train_path, 'r', encoding = 'utf-8').read().split("\n")
    test_images = open(test_path, 'r', encoding = 'utf-8').read().split("\n")
    
    # train_img=prepare_train_images(train_path+'/',img,train_images)
    train_img = [] 
    test_img=[] 
    for im in img:
        if(im[len(img_path+'/'):] in train_images):
            train_img.append(im)
    for im in img:
        if(im[len(img_path+'/'):] in test_images):
            test_img.append(im)        
    train_descriptions = load_clean_descriptions(descriptions, train_images)
    train_caption=train_captions(train_descriptions)
    vocab_size ,max_length,wordtoix=find_vocab_size_max_len(vocab,train_caption)
    lis_of_data=[vocab_size,max_length,wordtoix,train_img,test_img]
    
    prepaired_data_dirs=config["prepaired_data_dir"]
    for local_dir in (prepaired_data_dirs):
        create_directory([local_dir])
        
    lis_of_data_name=['vocab_size','max_length','wordtoix','train_img','test_img']     
    for local_dir,data_path,data in zip(prepaired_data_dirs,lis_of_data_name,lis_of_data):
        file1 = open(local_dir+'/'+data_path+'.txt','w')  
        logging.info(f"started writing the prepared {data_path} in the {data_path+'.txt'} file ")    
        file1.write(str(data))
        logging.info(f"prepared data  written succesfully in the {data_path+'.txt'} file and stored in the {local_dir} dir ") 
        file1.close()
        
    
        
        
  
            
    

    
    
  


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args=args.parse_args()
    try:
        logging.info("\n >>>>>>>>>> stage two started")
        data_preparation(config_path=parsed_args.config,params_path=parsed_args.params) 
        logging.info("stage two completed !! all the img_text_data prepartion done and stored \n >>>>>>>>>>>>")  
    except Exception as e:
        logging.exception(e)
