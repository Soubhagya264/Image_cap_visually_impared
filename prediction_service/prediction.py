import argparse
import pandas as pd
import os
import shutil
from googletrans import Translator
from gtts import gTTS
import base64
import numpy as np
import ast
from tqdm import tqdm
import logging
from pickle import dump, load
import ast
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import yaml
import os
import json
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

args=argparse.ArgumentParser()
args.add_argument("--config","-c",default="config/config.yaml")
args.add_argument("--params","-p",default="params.yaml")
parsed_args=args.parse_args()


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content=yaml.safe_load(yaml_file)
        logging.info(f"Yaml file :{path_to_yaml} lodded sucessfully")
    return content

def prediction_model(config_path):
    config = read_yaml(config_path)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    return trained_model_dir
    
model_dir=prediction_model(config_path=parsed_args.config)         
model=load_model(os.path.join(model_dir+'\model_149.h5'))  

def extract_features(image):
        def preprocess_img(img_path):
            # inception v3 excepts img in 299 * 299 * 3
            img = load_img(img_path, target_size = (299, 299))
            x = img_to_array(img)
            x = np.expand_dims(x, axis = 0)
            x = preprocess_input(x)
            return x

        base_model = InceptionV3(weights = 'imagenet')
        model = Model(base_model.input, base_model.layers[-2].output)
        image = preprocess_img(image)
        vec = model.predict(image)
        vec = np.reshape(vec, (vec.shape[1]))
        return vec
    
def imageSearch(photo):
        file=open("prepaired_data/max_length_dir/max_length.txt","r")
        max_length=int(file.read())
        file.close()
        
        file=open("prepaired_data/wordtoix_dir/wordtoix.txt","r")
        wordtoix=ast.literal_eval(file.read())  
        file.close()
        
        file=open("prepaired_data/ixtoword_dir/ixtoword.txt","r")
        ixtoword=ast.literal_eval(file.read())  
        file.close()
        in_text = 'startseq'
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final




def prediction_model(config_path):
    config = read_yaml(config_path)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    return trained_model_dir

def get_audio_file(config_path):
    config = read_yaml(config_path)
    artifacts=config["artifacts"]
    Mp3_dir=artifacts['MP3_DIR']
    return Mp3_dir
    # create_directory([os.path.join(loc)])
def create_directory(dirs:list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True) 
        logging.info(f"directory is created at {dir_path}") 
        
def translate(data):
        
    translator=Translator()
    out=translator.translate(data,dest='hi')
    return out.text



def text2Speech(data,loc):
    my_text = data
    tts = gTTS(text=my_text, lang='en', slow=False)
    
    tts.save(os.path.join(loc,"sound.mp3"))
    with open(loc+"\sound.mp3", "rb") as file:
        my_string = base64.b64encode(file.read())
    return my_string

    

  
def predict(data):
        photo = extract_features(data)
        photo=photo.reshape((1,2048))
        cap=imageSearch(photo) 
        translated=translate(cap)
        loc=get_audio_file(config_path=parsed_args.config)
        create_directory([os.path.join(loc)])
        string=text2Speech(translated,loc)
        return cap ,string  






# if __name__=="__main__":
#     args=argparse.ArgumentParser()
#     args.add_argument("--config","-c",default="config/config.yaml")
#     args.add_argument("--params","-p",default="params.yaml")
#     parsed_args=args.parse_args()
#     try:
#         logging.info("\n >>>>>>>>>> Prediction started")
#         model_dir=prediction_model(config_path=parsed_args.config) 
        
#         model=load_model(os.path.join(model_dir+'\model_149.h5'))
#         def predict_file(data):
#             photo = extract_features(data)
#             photo=photo.reshape((1,2048))
#             cap=imageSearch(model,photo)
#             translated=translate(cap)
#             loc=get_audio_file(config_path=parsed_args.config)
#             create_directory([os.path.join(loc)])
#             text2Speech(translated,loc)
            
#             return cap
        
        
        
#         logging.info("Prediction Done \n >>>>>>>>>>>>")  
#     except Exception as e:
#         logging.exception(e)