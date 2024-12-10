import os 
import sys 
import pandas as pd 
import numpy as np 

from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass  
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR 
from src.utils.utils import save_object, evaluate_models 
# from catboost import CatBoostRegressor 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_arr, test_arr ):
        try:
            logging.info("Model training started")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001], 
                    'n_estimators': [8,16,32,64,128,256]
                } 
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model Report: {model_report}")

            ## To get best model score from report
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f'Best Model: {best_model}')
            logging.info(f"R2 score of best model is {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)