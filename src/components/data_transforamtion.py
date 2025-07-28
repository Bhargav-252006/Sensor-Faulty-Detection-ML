import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    artifact_dir:str=os.path.join(artifact_folder)#from constants artifacts folder
    # where train test prerocessor are stored
    transformed_train_file_path=os.path.join(artifact_dir,"train.npy")
    transformed_test_file_path=os.path.join(artifact_dir,"test.npy")
    transformed_obj_file_path=os.path.join(artifact_dir,"preprocessor.pkl")



class DataTransforamtion:
    def __init__(self,feature_store_file_path):
        self.feature_store_file_path=feature_store_file_path#where is csv stored
        self.data_transformation_config=DataTransformationConfig()
        self.utils=MainUtils()



    #get the data from csv
    def getData(self,feature_store_file_path:str)->pd.DataFrame:
        try:
            df=pd.read_csv(feature_store_file_path)
            df.rename(columns={"Good/Bad":TARGET_COLUMN}, inplace=True)
            return df

        except Exception as e:
            raise CustomException(e,sys)
        







    #for the imputer ans scaler object
    def get_transformer_object(self):
        try:
            imp=("imputer",SimpleImputer(strategy="constant",fill_value=0))
            sca=("scaler",RobustScaler())

            #build both in one pkl 
            preprocessor=Pipeline(
                steps=[imp,sca]
            )

            return preprocessor
        


        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_tranforamtion(self):

        logging.info("initated data tranformation method in data tarnsformation class")

        try:
            df=self.getData(self.feature_store_file_path)
            X=df.drop(columns=[TARGET_COLUMN],axis=1)
            y = np.where(df[TARGET_COLUMN] == -1, 0, df[TARGET_COLUMN])#replace -1 with 0 and 1 remains same

            x_tr,x_te,y_tr,y_te=train_test_split(X,y,test_size=0.2)

            preprocessor=self.get_transformer_object()

            x_tr_transformed=preprocessor.fit_transform(x_tr)
            x_te_transformed=preprocessor.transform(x_te)# only transform for test data


            preprocessor_path=self.data_transformation_config.transformed_obj_file_path#path of preprocessor
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)#create the dir for saving the preprocessor
            self.utils.save_object(file_path=preprocessor_path,obj=preprocessor)

            train_arr=np.c_[x_tr_transformed,y_tr]
            test_arr=np.c_[x_te_transformed,y_te]

            return (train_arr,test_arr,preprocessor_path)
        except Exception as e:
            raise CustomException(e,sys)
        


