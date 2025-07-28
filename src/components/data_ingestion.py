import sys 
import os
import numpy as np
import pandas as pd
from pymongo.mongo_client import MongoClient #for mongo connecting
from zipfile import Path# since in data ingestio output is csv path

#src files
from src.constant import * # for all constant 
from src.exception import CustomException# if any error
from src.logger import logging# for maintaing track record
from src.utils.main_utils import MainUtils# for using helpers

from dataclasses import dataclass# for avoiding constructor function writing


@dataclass
class DataIngestionConfig:
    artifact_dir:str=os.path.join(artifact_folder)

class DataIngestion:

    
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.utils=MainUtils()




    def export_collection_as_df(self,db,collection):
        try :
            client=MongoClient(MONGO_DB_URL)
            coll=client[db][collection]
            df=pd.DataFrame(list(coll.find()))
            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"],axis=1,inplace=True)
            df.replace({"na":np.nan},inplace=True)
            df.drop(columns=["Unnamed: 0"],axis=1,inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        



    def export_data_into_feature_store_filepath(self)->pd.DataFrame:
        try:
            #for showing log info
            logging.info(f"exporting info from monogo db")


            #store its location
            raw_file=self.data_ingestion_config.artifact_dir


            #save in artifacts folder
            os.makedirs(raw_file,exist_ok=True)

            #get data
            sensor_data=self.export_collection_as_df(MONGO_DATABASE_NAME,MONGO_COLLECTION_NAME)

            logging.info(f"saving data in file path {raw_file}")


            #this is path where data store
            feature_store_file_path=os.path.join(raw_file,"wafer_fault.csv")

            #convert to csv
            sensor_data.to_csv(feature_store_file_path,index=False)
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e,sys)
    




    # for initaiting the data ingestion
    def initaite_data_ingestion(self):
        logging.info("entered intiated data ingestion class")
        try :
            feature_store_file_path=self.export_data_into_feature_store_filepath()

            logging.info("got data from mongo db")
            logging.info("exited data ingestion")


            return feature_store_file_path
        

        except Exception as e:
            raise CustomException(e,sys)
            




