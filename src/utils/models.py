import tensorflow as tf
import os
import logging
from keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.backend import flatten
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from utils.all_utils import get_time_stamp 

def get_inception_v3_model(model_path):
    model=InceptionV3(weights='imagenet')
    model.save(model_path)
    logging.info(f"inception_v3 model saved at: {model_path}")
    return model

# def prepare_model(model,CLASSES,freeze_all,freeze_till,learning_rate):
#     if freeze_all:
#         for layer in model.layers:
#             layer.trainable=False
#     elif freeze_till is not None and freeze_till > 0:
#             for layer in model.layers[:-freeze_till]:
#                 layer.trainable=False
#     flatten_in=tf.keras.layers.Flatten()(model.output)
#     prediction=tf.keras.layers.Dense(
#         units=CLASSES,activation='softmax'
#     )(flatten_in)            
#     full_model=tf.keras.Model(
#         inputs=model.input,
#         outputs=prediction
#     ) 
#     full_model.compile(
#         optimizer=tf.keras.optimizers.SGD(
#             learning_rate=learning_rate
#         ),
#         loss=tf.keras.losses.CategoricalCrossentropy(),
#         metrics=["accuracy"]
#     )   
    
#     logging.info(f"custom model is compiled and ready to trained")  
#     return full_model 
# def preprocess(image_path):
#     img=image.load_img(image_path,target_size=(299,299))
#     x=image.img_to_array(img)
#     x=np.expand_dims(x,axis=0)
#     x=preprocess_input(x)
#     return x