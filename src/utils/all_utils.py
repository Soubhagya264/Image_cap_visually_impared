import yaml
import os
import json
import logging
import time
def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content=yaml.safe_load(yaml_file)
        logging.info(f"Yaml file :{path_to_yaml} lodded sucessfully")
    return content

def create_directory(dirs:list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True) 
        logging.info(f"directory is created at {dir_path}") 
        
def save_local_df(data,data_path):
    data.to_csv(data_path,index=False)
    logging.info("Data is saved at",data_path)
    
def save_reports(report: dict, report_path: str, indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    logging.info(f"reports are saved at {report_path}")
    
def get_time_stamp(name):
    time_stamp=time.asctime().replace(" ","_").replace(":","_")
    unique_name=f"{name}_at_{time_stamp}"
    return unique_name