from flask import Flask, render_template, request, jsonify
from src.utils.all_utils import read_yaml,create_directory
import os
import numpy as np
from prediction_service import prediction
import argparse
import logging
logging_str="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')

from werkzeug.utils import secure_filename

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


def get_folder_name(config_path):
    config=read_yaml(config_path)
    artifacts=config['artifacts']
    location=artifacts["LOCATION"] 
    return location
    
    

@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        try:
            if request.form:
                logging.info(f'request from from')
                file = request.files['file1']
                #file.save(os.path.abspath("images"),secure_filename(file.filename))
                response = prediction.predict(file)
                logging.info(f'predicted from request')
                return render_template("index.html", response=response)
            elif request.json:
                logging.info(f'request from api{request.json}')
                print(request.get_json)
                response = prediction.predict(request.get_json)
                print(response)
                logging.info(f'predicted api request')
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args=args.parse_args()
    loc=get_folder_name(config_path=parsed_args.config)
    print()
    try:
        logging.info("\n >>>>>>>>>> app.py running ")
        loc=get_folder_name(config_path=parsed_args.config) 
        create_directory([os.path.join(loc)])
        print(os.path.abspath("images"))
        logging.info("completed \n >>>>>>>>>>>>")  
    except Exception as e:
        logging.exception(e)