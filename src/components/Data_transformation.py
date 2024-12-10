
import os 
import sys 
import pandas as pd 
import numpy as np

from dataclasses import dataclass 
from src.logger import logging 
from src.exception import CustomException 
from src.utils.utils import save_object

from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder


@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_preprocessor_object(self, train_df : pd.DataFrame) -> ColumnTransformer:

        try:
            logging.info("Creating preprocessor object")

            # numerical_columns = train_df.select_dtypes(exclude=["object"]).columns
            # categorical_columns = train_df.select_dtypes(include=["object"]).columns

            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            logging.info(f"Numberical columns : {numerical_columns}")
            logging.info(f"Categorical columns : {categorical_columns}")

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())]
                )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))]
                )

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info("Preprocessor object created")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    def initialize_data_transformation(self, train_path : str, test_path : str):
        try:
            logging.info("Data Transforrmation started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test data completed")


            # Training data 
            target_column_name = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_column_train_df = train_df[target_column_name]

            # Test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_column_test_df = test_df[target_column_name]

            logging.info(f"Train data before transformation \n {train_df}")
            logging.info(f"Test data before transformation \n {test_df}")
           

            # Data transformationw with preprocessor
            preprocessor_obj = self.get_data_preprocessor_object(input_feature_train_df)
            input_feature_train_transform_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_transform_arr = preprocessor_obj.transform(input_feature_test_df)

            # logging.info(f'Input Train Data After Transformation: \n {input_feature_train_transform_arr}')
            # logging.info(f'Input Test Data After Transformation:\n {input_feature_test_transform_arr}')

            train_arr = np.c_[input_feature_train_transform_arr, np.array(target_column_train_df)]
            test_arr = np.c_[input_feature_test_transform_arr, np.array(target_column_test_df)]

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessor_obj
            )

            logging.info(f"Saved preprocessing object.")
            logging.info("Data transformation completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)