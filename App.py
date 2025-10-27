import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

import pickle


with open('Clean_Dataset_Model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler = StandardScaler()

st.title("Clean_Dataset_Model")

airline=st.selectbox("Enter Airlines name",['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo','Air_India'])

source=st.selectbox("Enter Source city",['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])

destination=st.selectbox("Enter Destination city",['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])

departure=st.selectbox("Enter Departure time",['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night','Late_Night'])

arrival=st.selectbox("Enter Arrival time",['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening','Late_Night'])

class_type=st.selectbox("Enter type of class",["Economy","Business"])

dur=st.number_input("Enter duration for flight")

days=st.number_input("Enter days left")

stop=st.selectbox("Enter number of stops",["zero","one","two or more"])

# input_df = pd.DataFrame([[dur,days,airline,source,departure, stop,arrival,destination,class_type]],columns=['duration', 'days_left', 'airline_AirAsia', 'airline_Air_India','airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet','airline_Vistara', 'source_city_Bangalore', 'source_city_Chennai','source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata','source_city_Mumbai', 'departure_time_Afternoon','departure_time_Early_Morning', 'departure_time_Evening','departure_time_Late_Night', 'departure_time_Morning','departure_time_Night', 'stops_one', 'stops_two_or_more', 'stops_zero','arrival_time_Afternoon', 'arrival_time_Early_Morning','arrival_time_Evening', 'arrival_time_Late_Night','arrival_time_Morning', 'arrival_time_Night','destination_city_Bangalore', 'destination_city_Chennai','destination_city_Delhi', 'destination_city_Hyderabad','destination_city_Kolkata', 'destination_city_Mumbai', 'class_Business','class_Economy'])
input_df = pd.DataFrame({
    'duration': [dur],
    'days_left': [days],
    'airline': [airline],
    'source_city': [source],
    'departure_time': [departure],
    'stops': [stop],
    'arrival_time': [arrival],
    'destination_city': [destination],
    'class': [class_type]
})

input_df= pd.get_dummies(input_df)

column=['duration', 'days_left', 'airline_AirAsia', 'airline_Air_India',
       'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet',
       'airline_Vistara', 'source_city_Bangalore', 'source_city_Chennai',
       'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata',
       'source_city_Mumbai', 'departure_time_Afternoon',
       'departure_time_Early_Morning', 'departure_time_Evening',
       'departure_time_Late_Night', 'departure_time_Morning',
       'departure_time_Night', 'stops_one', 'stops_two_or_more', 'stops_zero',
       'arrival_time_Afternoon', 'arrival_time_Early_Morning',
       'arrival_time_Evening', 'arrival_time_Late_Night',
       'arrival_time_Morning', 'arrival_time_Night',
       'destination_city_Bangalore', 'destination_city_Chennai',
       'destination_city_Delhi', 'destination_city_Hyderabad',
       'destination_city_Kolkata', 'destination_city_Mumbai', 'class_Business',
       'class_Economy']

input_df = input_df.reindex(columns=column, fill_value=0)
input_df= scaler.fit_transform(input_df)

if st.button("Predict"):
    price = model.predict(input_df)
    st.write("Predicted Price of flight: ",price)
   
