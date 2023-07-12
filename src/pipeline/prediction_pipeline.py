#create prediction pipeline
# create function to load objects
# create custom class based on dataset
# create a function to convert dataframe 
# into dictionary


import os, sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    # write helper function to load .pkl file in model_path in utils.py
    # function to load .pkl files
    def predict(self, features):
        preprocessor_path= os.path.join("artifacts","preprocessor.pkl")
        model_path= os.path.join ("artifacts/model_trainer","model.pkl")
        
        processor = load_object(preprocessor_path)
        model = load_object (model_path)
        
        scaled = processor.transform (features)
        pred = model.predict(scaled)
        
        return pred
        
class AirlineClass:
    def __init__(self,
                 Age :int,
                 Flight_Distance:int, 
                 Inflight_wifi_service:int,
                 Departure_Arrival_time_convenient:int, 
                 Ease_of_Online_booking:int,
                 Gate_location:int, 
                 Food_and_drink:int,
                 Online_boarding:int,
                Seat_comfort:int,
                Inflight_entertainment:int,
                On_board_service:int,
                Leg_room_service:int,
                Baggage_handling:int,
                Checkin_service:int,
                Inflight_service:int,
                Cleanliness:int,
                Departure_Delay_in_Minutes:int,
                Arrival_Delay_in_Minutes:int):  
        
        self.Age = Age
        self.Flight_Distance =  Flight_Distance
        self.Inflight_wifi_service = Inflight_wifi_service
        self.Departure_Arrival_time_convenient =  Departure_Arrival_time_convenient
        self.Ease_of_Online_booking = Ease_of_Online_booking
        self.Gate_location = Gate_location 
        self.Food_and_drink = Food_and_drink
        self.Online_boarding = Online_boarding 
        self.Seat_comfort = Seat_comfort
        self.Inflight_entertainment = Inflight_entertainment
        self.On_board_service = On_board_service
        self.Leg_room_service = Leg_room_service
        self.Baggage_handling = Baggage_handling
        self.Checkin_service =  Checkin_service
        self.Inflight_service = Inflight_service
        self.Cleanliness = Cleanliness
        self.Departure_Delay_in_Minutes = Departure_Delay_in_Minutes
        self.Arrival_Delay_in_Minutes =   Arrival_Delay_in_Minutes    
    
    def get_data_into_DataFrame(self):
        try:
            
            custom_input ={
                'Age':[self.Age],
                'Flight_Distance' :[self.Flight_Distance],
                'Inflight_wifi_service':[self.Inflight_wifi_service],
                'Departure_Arrival_time_convenient':[self.Departure_Arrival_time_convenient],
                'Ease_of_Online_booking':[self.Ease_of_Online_booking],
                'Gate_location':[self.Gate_location],
                'Food_and_drink':[self.Food_and_drink],
                'Online_boarding':[self.Online_boarding],
                'Seat_comfort':[self.Seat_comfort],
                'Inflight_entertainment':[self.Inflight_entertainment],
                'On_board_service':[self.On_board_service],
                'Leg_room_service':[self.Leg_room_service],
                'Baggage_handling':[self.Baggage_handling],
                'Checkin_service' :[self.Checkin_service],
                'Inflight_service':[self.Inflight_service],
                'Cleanliness':[self.Cleanliness],
                'Departure_Delay_in_Minutes':[self.Departure_Delay_in_Minutes],
                'Arrival_Delay_in_Minutes':[self.Arrival_Delay_in_Minutes]             
            }
            
            data=pd.DataFrame(custom_input)
            
            return data
        
        except Exception as e:
            raise CustomException (e,sys)