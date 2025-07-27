import os
import sys

from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transforamtion import DataTransforamtion
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            return DataIngestion().initaite_data_ingestion()
        except Exception as e:
            raise CustomException(e,sys)
        


        
    def start_data_transformtion(self,feature_store_file_path):
        try:
            return DataTransforamtion(feature_store_file_path).initiate_data_tranforamtion()
        except Exception as e:
            raise CustomException(e,sys)
        



    def start_train_model(self,trainArr,testArr):
        try:
            return ModelTrainer().initiate_train_model(trainArr,testArr)
        except Exception as e:
            raise CustomException(e,sys)
     

    def run_pipeline(self):
        try:
            feature_store_file_path=self.start_data_ingestion()
            train_arr,test_arr,preprocessor_path=self.start_data_transformtion(feature_store_file_path)
            trained_model_file_path=self.start_train_model(train_arr,test_arr)
            print(f"training completed and saved at {trained_model_file_path}")

            
        except Exception as e:
            raise CustomException(e,sys)
            