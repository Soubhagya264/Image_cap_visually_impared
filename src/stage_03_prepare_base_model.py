from utils.all_utils import read_yaml,create_directory
from utils.models import get_inception_v3_model ,prepare_model,preprocess
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import ast
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging
import io
import numpy as np

logging_str="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')

def encode(image,model):
    image = preprocess(image)
    vec = model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

def prepare_base_model(config_path,params_path):
    config=read_yaml(config_path)
    param=read_yaml(params_path)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    base_model_dir=artifacts["BASE_MODEL_DIR"]
    base_model_name=artifacts["BASE_MODEL_NAME"]
    base_model_dir_path=os.path.join(artifacts_dir,base_model_dir)
    create_directory([base_model_dir_path])
    base_model_path=os.path.join(base_model_dir_path,base_model_name)
    base_model=get_inception_v3_model(model_path=base_model_path)
    model=prepare_model(base_model)
    images=os.path.join(artifacts["DATA_DIR"],artifacts["TEXT_DATA"],artifacts["TRAIN_PATH"])
    file=open("prepaired_data/train_img/train_img.txt","r")
    train_img=[]
    train_img.append(file.read())
    file.close()
    train_img=ast.literal_eval(train_img[0])
    print(len(train_img))
    encoding_train = {}
    logging.info("encoding start")
    for img in train_img:
        encoding_train[img[len(images+'/'):]] = encode(img,model)
    logging.info("encoding done")    
    print(encoding_train)    
    
    
    
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args=args.parse_args()
    try:
        logging.info("\n >>>>>>>>>> stage three started")
        prepare_base_model(config_path=parsed_args.config,params_path=parsed_args.params) 
        logging.info("stage three completed !! base model is created \n >>>>>>>>>>>>")  
    except Exception as e:
        logging.exception(e)