from utils.all_utils import read_yaml,create_directory
from utils.callbacks import get_callbacks
from utils.models import get_unique_path_to_save_model
from utils.data_management import data_generator
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding,TimeDistributed,Dense,RepeatVector,Activation,Flatten,Reshape,concatenate,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers.wrappers import Bidirectional
from tensorflow.keras.layers.merge import add
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
import argparse
import pandas as pd
import os
import shutil
import io
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
    
    logging.info(f"Extracted Input Features")
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
    epoch=params["EPOCHS"]
    size=params["SIZE"]
    embedding_dim=params["EMBEDDING_DIM"]
    optimizer=params["OPTIMIZER"]
    dropout_ratio=params["DROPOUT_RATIO"] 
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    logging.info(f"Embedding matrix created")        
    
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    callback_dir_path  = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)
    
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str=stream.getvalue()
        return summary_str  
    
    
    
    logging.info(f"training started")       
    ip1 = Input(shape = (2048, ))
    fe1 = Dropout(0.2)(ip1)
    fe2 = Dense(size, activation = 'relu')(fe1)
    ip2 = Input(shape = (max_length, ))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero = True)(ip2)
    se2 = Dropout(dropout_ratio)(se1)
    se3 = LSTM(size)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(size, activation = 'relu')(decoder1)
    outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
    model = Model(inputs = [ip1, ip2], outputs = outputs)        
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    logging.info(f" full model summary \n {_log_model_summary(model)}")
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
    # model.fit([X1, X2], y, epochs = epoch, batch_size = size,callbacks=callbacks)    
    logging.info(f"training completed")
    
    

    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directory([trained_model_dir])
    model_file_path = get_unique_path_to_save_model(trained_model_dir)
    model.save(model_file_path)
    logging.info(f"trained model is saved at: {model_file_path}")
    
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