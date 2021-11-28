# 
# 
from flask import Flask, json, render_template, request, jsonify,send_from_directory
from wsgiref import simple_server
from src.utils.all_utils import read_yaml,create_directory
import os
from flask import Response
from flask_cors import CORS, cross_origin
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
UPLOAD_FOLDER= 'C:/Users/HP/Major_Project/Image_Captioning_for_visually_impaired/images'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
MP3_folder='C:/Users/HP/Major_Project/Image_Captioning_for_visually_impaired/speechs'
    

@app.route('/speechs/<path:filename>')
def download_file(filename):
    return send_from_directory('C:/Users/HP/Major_Project/Image_Captioning_for_visually_impaired/speechs/', filename)

# def get_folder_name(config_path):
#     config=read_yaml(config_path)
#     artifacts=config['artifacts']
#     location=artifacts["LOCATION"] 
#     return location
        
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index2.html')    
    
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        
        # if request.method=="POST":
                print(request.files)
                logging.info("getting file")
                file = request.files['file']
                print(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
                path=os.path.abspath("images")
                
                img_path = path+"\\"+os.listdir(path)[0] 
                
                path=os.path.abspath("speechs")
                
                if os.listdir(path):
                    sound_path = path+"\\"+os.listdir(path)[0]
                    os.remove(sound_path)
               
                res,string = prediction.predict(img_path)
                print(res,string)
                os.remove(img_path)
                logging.info("predicted responses")
                return {"data" : string.decode("utf-8")}       
        
    except Exception as e:
        logging.info(e)     
    except ValueError:
        logging.info("value error")
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    




port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args=args.parse_args()
    
      
