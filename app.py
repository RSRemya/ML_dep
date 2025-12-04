#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import pandas as pd
import streamlit as st


with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

def preprocess_and_predict(features):
    input_data = pd.DataFrame([features])
    input_data = pd.get_dummies(input_data, columns = ['sex'], drop_first=True)
    required_columns = model.feature_names_in_
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col]=0
    input_data = input_data[required_columns]
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:,1]
    return prediction[0], probability[0]


st.title("Heart Disease Prediction App")


age = st.number_input("Age",min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["Male","Female"])
cp = st.selectbox("Chest Pain Type", options=[0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure",min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level",min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar", options=[0,1])
restecg = st.selectbox("Resting ECG", options=[0,1,2])
thalach = st.number_input("Max. HR",min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0,1])
oldpeak = st.number_input("ST Depresion",min_value=0.0, max_value=6.0, value=1., step =0.1)
slope = st.selectbox("Slope of ST", options=[0,1,2])
ca = st.selectbox("N.o. of Major Blood Vessels", options=[0,1,2,3,4])
thal = st.selectbox("Thalessemia", options=[0,1,2,3])

features = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach, "exang":exang, "oldpeak":oldpeak, "slope":slope,"ca":ca, "thal":thal
}

if st.button("Predict"):
    prediction, probability = preprocess_and_predict(features)
    if prediction==1:
        st.error(f"There is high chance of heart disease with a probability of {probability}")
    if prediction==0:
        st.success(f"There is low chance of heart disease with a probability of {probability}")


# In[ ]:




