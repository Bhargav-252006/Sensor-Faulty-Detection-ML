import os 
import sys
import numpy as np
import pandas as pd

from src.constant import *
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomException

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBClassifier
from dataclasses import dataclass
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    arti_dir=os.path.join(artifact_folder)
    trained_model_file_path=os.path.join(arti_dir,"model.pkl")
    expected_accuracy=0.45
    model_config_file_path=os.path.join("config","model.yaml")



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.utils=MainUtils()
        self.models={
            'svc':SVC(),
            'RandomForestClassifier':RandomForestClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'XGBClassifier':XGBClassifier()
        }


    def evalvate_models(self,x_tr,x_te,y_tr,y_te,models):

        report_accuracy={}
        try:
            for name,model in models.items():
                logging.info(f"Training and evaluating model: {name}")
                ins=model
                ins.fit(x_tr,y_tr)
                y_pre=ins.predict(x_te)
                acc=accuracy_score(y_te,y_pre)
                report_accuracy[name]=acc
                logging.info(f"  {name} Accuracy: {acc:.4f}")

            return report_accuracy
        except Exception as e:
            raise CustomException(e,sys)
            





    def get_best_model(self,x_tr,x_te,y_tr,y_te):
        try:
            report=self.evalvate_models(x_tr,x_te,y_tr,y_te,self.models)
            best_acc=max(report.values())
            best_name = max(report, key=report.get)#search about this
            best_model_obj=self.models[best_name]
            return (best_model_obj,best_name)


        except Exception as e:
            raise CustomException(e,sys)
        





    def finetune_best_model(self,best_model_object,best_model_name,x_tr,y_tr):

        try:
            model_param_grid=self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            
            gridSearch=GridSearchCV(best_model_object,param_grid=model_param_grid,verbose=1,n_jobs=-1,cv=5)
            gridSearch.fit(x_tr,y_tr)


            best_parameters=gridSearch.best_params_# return a dict 
            print(f"best paraments are: {best_parameters}")


            fine_tune_model=best_model_object.set_params(**best_parameters)#here the model is fitted with that best paramets
            return fine_tune_model

        except Exception as e:
            raise CustomException(e,sys)
        




    def initiate_train_model(self,trainArr,testArr):

        try :

            logging.info("splliting the train array and test array")
            x_tr,y_tr=trainArr[:,:-1],trainArr[:,-1]
            x_te,y_te=testArr[:,:-1],testArr[:,-1]

            logging.info("extarcting the model config file path")
            report_accuracy=self.evalvate_models(x_tr,x_te,y_tr,y_te,self.models)



            best_model_obj,best_name=self.get_best_model(report_accuracy)

            fine_tune_model=self.finetune_best_model(best_model_obj,best_name,x_tr,y_tr)


            fine_pre=fine_tune_model.predict(x_te)

            fine_acc=accuracy_score(y_te,fine_pre)
            logging.info(f"Fine-tuned model ({best_name}) accuracy on test set: {fine_acc:.4f}")

            if fine_acc<self.model_trainer_config.expected_accuracy:
                raise CustomException("no model found with expected accuracy",sys)
            
            logging.info(f"saving the best fine tunned model{self.model_trainer_config.trained_model_file_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path),exist_ok=True)
            

            self.utils.save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=fine_tune_model)
            
            return self.model_trainer_config.trained_model_file_path
        
        

        except Exception as e:
            raise CustomException(e,sys)




    



