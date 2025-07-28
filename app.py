from flask import Flask,render_template,request,jsonify,send_file
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.constant import *


from src.pipeline.predict_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainingPipeline


app=Flask(__name__)


@app.route("/")
def home():
    return "Welcome to Sensory Faulty Detection"

@app.route("/train")
def train_route():
    try:
        TrainingPipeline().run_pipeline()
        return "training completed"

    except Exception as e:
        raise CustomException(e,sys)
    

@app.route("/predict",methods=['POST','GET'])
def upload():    
    try:
        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
           
            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_prediction_pipeline()


            logging.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)




        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e,sys)
   




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)
