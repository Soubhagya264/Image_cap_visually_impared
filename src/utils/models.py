import tensorflow as tf
import os
import logging
from keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.backend import flatten
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from keras.models import Model
from utils.all_utils import get_time_stamp 

def get_inception_v3_model(model_path):
    model=InceptionV3(weights='imagenet')
    model.save(model_path)
    logging.info(f"inception_v3 model saved at: {model_path}")
    return model

def prepare_model(base_model):
    model = Model(base_model.input, base_model.layers[-2].output)
    return model

def preprocess(image_path):
    img=image.load_img(image_path,target_size=(299,299))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    return x

def get_unique_path_to_save_model(trained_model_dir, model_name="model"):
    timestamp = get_time_stamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path  
