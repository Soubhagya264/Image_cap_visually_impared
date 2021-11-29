from utils.all_utils import read_yaml,create_directory
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

def copy_file(source_path,local_dir):
    lis_of_file=os.listdir(source_path)
    n=len(lis_of_file)
    for file in tqdm(lis_of_file,total=n,desc=f"copying file from {source_path} to {local_dir}",colour="green"):
        src=os.path.join(source_path,file)
        des=os.path.join(local_dir,file)
        shutil.copy(src,des)


def get_data(config_path):
    config=read_yaml(config_path)
    
    source_download_path=config["source_download_path"]
    local_data_dirs=config["local_data_dirs"]
    print(local_data_dirs)
    print(source_download_path)
    
    for source_path,local_dir in tqdm(zip(source_download_path,local_data_dirs),total=4,desc="list of folders",colour="red"):
        create_directory([local_dir])
        copy_file(source_path,local_dir)
    
    
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    parsed_args=args.parse_args()
    try:
        logging.info("\n >>>>>>>>>> stage one started")
        get_data(config_path=parsed_args.config) 
        logging.info("stage one completed !! all the data stored in local \n >>>>>>>>>>>>")  
    except Exception as e:
        logging.exception(e)
    