import os 
import sys
import numpy as np
import pandas as pd

import shutil 
import pickle   #trained model is saved in this format

from src.constant import *
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomException
 

from dataclasses import dataclass
from flask import request


@dataclass

class PredictionPipelineConfig:
    prediction_dir_name="predictions"
    prediction_file_name="prediction_file.csv"
    prediction_file_path=os.path.join(prediction_dir_name,prediction_file_name)
    trained_model_file_path=os.path.join(artifact_folder,"model.pkl")
    preprocessed_file_path=os.path.join(artifact_folder,"preprocessor.pkl")



class PredictionPipeline:
    def __init__(self,request=request):
        self.request=request
        self.prediction_pipeline_config=PredictionPipelineConfig()
        self.utils=MainUtils()


    def save_input_files(self) -> str:
        try:
            prediction_input_dir = "prediction_artifacts"
            os.makedirs(prediction_input_dir, exist_ok=True)

            input_csv_file = self.request.files.get('file')
            if input_csv_file is None or input_csv_file.filename == '':
                raise CustomException("No file uploaded or filename is empty.", sys)

            pred_input_file_path = os.path.join(prediction_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_input_file_path)

            return pred_input_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self,features):

        try:
            model=self.utils.load_object(
                file_path=self.prediction_pipeline_config.trained_model_file_path
            )

            preprocessor=self.utils.load_object(
                file_path=self.prediction_pipeline_config.preprocessed_file_path
            )

            transformed_x=preprocessor.transform(features)
            pred=model.predict(transformed_x)

            return pred


        except Exception as e:
            raise CustomException(e,sys)
        


    def get_predict_dataframe(self,input_csv_file_path):

        try:
            prediction_column_name=TARGET_COLUMN
            df=pd.read_csv(input_csv_file_path)

            df=df.drop(columns="Unnamed 0") if "Unnamed 0" in df.columns else df


            predictions=self.predict(df)
            df[prediction_column_name]=[pred for pred in predictions]


            mapping={0:"bad",1:"good"}
            df[prediction_column_name]=df[prediction_column_name].map(mapping)


            #save the ouput in the folder
            os.makedirs(self.prediction_pipeline_config.prediction_dir_name,exist_ok=True)

            df.to_csv(self.prediction_pipeline_config.prediction_file_path,index=False)

            logging.info("predictios completed and saved ")

        except Exception as e:
            raise CustomException(e,sys)


    def run_prediction_pipeline(self):
        try:
            
            pred_input_file_path=self.save_input_files()
            self.get_predict_dataframe(pred_input_file_path)

            return self.prediction_pipeline_config

        except Exception as e:
            raise CustomException(e,sys)    




        

    



