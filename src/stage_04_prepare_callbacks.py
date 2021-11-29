from utils.all_utils import read_yaml,create_directory
from utils.callbacks import create_and_save_tensorboard_callback,create_and_save_checkpoint_callback
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging
logging_str="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')


def prepare_callbacks(config_path):
    config=read_yaml(config_path)
    
    
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    tensorboard_log_dir=os.path.join(artifacts_dir,artifacts["TENSORBOARD_ROOT_LOG_DIR"])
    checkpoint_dir=os.path.join(artifacts_dir,artifacts["CHECKPOINT_DIR"])
    Callbacks_dir=os.path.join(artifacts_dir,artifacts["CALLBACKS_DIR"])
    
    create_directory(
        [tensorboard_log_dir,
        checkpoint_dir,
        Callbacks_dir]
    )
    
    create_and_save_tensorboard_callback(Callbacks_dir,tensorboard_log_dir)
        
    create_and_save_checkpoint_callback(Callbacks_dir,checkpoint_dir)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    
    parsed_args=args.parse_args()
    try:
        logging.info("\n >>>>>>>>>> stage four started")
        prepare_callbacks(config_path=parsed_args.config) 
        logging.info("stage four completed !! callbacks are prepaired \n >>>>>>>>>>>>")  
    except Exception as e:
        logging.exception(e)