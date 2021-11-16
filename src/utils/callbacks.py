import os
import time
import tensorflow as tf
import joblib
import logging
from utils.all_utils import get_time_stamp



def create_and_save_checkpoint_callback(callbacks_dir,checkpoint_dir):
    checkpoint_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_best_only=True
    )

    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)
    logging.info(f"tensorboard callback is being saved at {ckpt_callback_filepath}")

def create_and_save_tensorboard_callback(Callbacks_dir,tensorboard_log_dir):
    unique_name=get_time_stamp("tb_logs")
    tb_running_log_dir=os.path.join(tensorboard_log_dir,unique_name)
    tb_callbacks=tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    tb_callbacks_filepath=os.path.join(Callbacks_dir,"tb_cb.cb")
    joblib.dump(tb_callbacks,tb_callbacks_filepath)
    logging.info(f"tensorboard callback is being saved at {tb_callbacks_filepath}")
    
    
def get_callbacks(callback_dir_path):

    callback_path = [
        os.path.join(callback_dir_path, bin_file) for bin_file in os.listdir(callback_dir_path) if bin_file.endswith(".cb")
    ]

    callbacks = [
        joblib.load(path) for path in callback_path
    ]

    logging.info(f"saved callbacks are loaded from {callback_dir_path}")

    return callbacks