import os, sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")# saving pkl file 
    
 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    
    
    def get_data_transformer_obj(self): #crate pkl features for conversion
        # this function is responcible for data transformation 
        try : 
            
                        
            numerical_columns= ['Age', 'Flight_Distance', 'Inflight_wifi_service',
       'Departure_Arrival_time_convenient', 'Ease_of_Online_booking',
       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',
       'Inflight_entertainment', 'On_board_service', 'Leg_room_service',
       'Baggage_handling', 'Checkin_service', 'Inflight_service',
       'Cleanliness', 'Departure_Delay_in_Minutes',
       'Arrival_Delay_in_Minutes']
            
            categorical_columns = ['Gender', 'Customer_Type', 'Type_of_Travel', 'Class'] #, 'satisfaction']
            
            #create num column changing pipeline
            #1. handling missing values
            num_pipeline= Pipeline(
                steps= [
                ("imputer", SimpleImputer(strategy="median")),
               #doing standard scaling ,pipeline should act on train dataset(fit.transform(trsin data) 
               # and for test just transform data and not fit_transform#)
                ("scaler",StandardScaler())]
            )
            
            logging.info("Numerical  column encoding completed")
            
            cat_pipeline= Pipeline(
                #handle missing vals /converting vals to num 
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))]              
            )
                
            logging.info("Categorical column encoding completed")
            
                       
            
            #Combine both pipelines, for that use column transformer
            preprocessor= ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline, categorical_columns),
            ])
            logging.info("Column Transfer completed")
            
            return preprocessor
        
        
        except Exception as e :
            raise CustomException(e,sys)
    
    def delete_columns(self, columns_to_delete, df):
        try:
            df = df.drop(columns_to_delete, axis=1)
            return df
        except Exception as e:
            logging.info("Error occurred while deleting columns")
            raise CustomException(e, sys)     
        
         
    def remove_outliers(self,col,df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lowwer_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lowwer_limit), col] = lowwer_limit

            return df

        except Exception as e:
            logging.info("Outliers handling code")
            raise CustomException(e, sys)     
        
      
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test csv file reading done ")
            
            
            # Columns to delete
            columns_to_delete = ['Unnamed:_0'] #,'Gate location','Departure Delay in Minutes', 'Arrival Delay in Minutes']

            # Delete columns from train_data and test_data
            train_df = self.delete_columns(columns_to_delete, train_df)
            test_df = self.delete_columns(columns_to_delete, test_df)
                  
            logging.info("Column Deletion from train and test data is done")
            
            #for col in numerical_columns
            numerical_features = [ 'Age', 'Flight_Distance', 'Inflight_wifi_service',
       'Departure_Arrival_time_convenient', 'Ease_of_Online_booking',
       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',
       'Inflight_entertainment', 'On_board_service', 'Leg_room_service',
       'Baggage_handling', 'Checkin_service', 'Inflight_service',
       'Cleanliness', 'Departure_Delay_in_Minutes',
       'Arrival_Delay_in_Minutes' ]
        
            for col in numerical_features:
                self.remove_outliers(col = col, df = train_df)
                self.remove_outliers(col = col, df = test_df)
            logging.info("Outliers capped on our train data")
            
            
        
            logging.info("Outliers capped on our train data")
                 
            
            logging.info("Getting preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()
                        
            target_col = 'satisfaction'
            
            #Converting categorical target feature to numeric
            # Map values for the 'satisfaction' column
            satisfaction_mapping = {
                'satisfied': 1,
                'neutral or dissatisfied': 0
            }
            train_df['satisfaction'] = train_df['satisfaction'].map(satisfaction_mapping)
            test_df['satisfaction'] = test_df['satisfaction'].map(satisfaction_mapping)

                      
            
            logging.info("Splitting train data into dependent and independent features")
            input_feature_train_df= train_df.drop(target_col,axis=1)
            target_feature_train_df= train_df[target_col]
            
            
            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_df= test_df.drop(target_col,axis=1)
            target_feature_test_df= test_df[target_col]
            
                       
            logging.info("Applying preprocessing obj on training dataframe")
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            
            logging.info("Applying preprocessing obj on testing dataframe")
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
            
            
            
            
            
            
            
            logging.info("Creating arrays of training and testing dataframe")
            train_arr=np.c_[ input_feature_train_arr, np.array(target_feature_train_df) ]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            
            
            #print(train_arr)
            #print("*"*30)
            #print(test_arr)
            
            logging.info("Saved preprocessing object.")
            
            save_object (
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )
            #print(test_arr)
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )  
              
            
            
        except Exception as e:
            raise CustomException(e,sys)