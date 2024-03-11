#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:59:27 2024

@author: rainpriest
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/home/rainpriest/machinelearning/trained_model.sav','rb'))

def heart_disease_prediction(input_data):
    input_data_as_float = [float(value) for value in input_data]
    
    #change input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data_as_float)

    #reshape numpy array as we are predicting for only instance
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction  = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]==0):
      return 'the person does not have a heart disease'
    else:
      return 'the person has heart disease'

def main():
    st.title('Heart disease prediction')
    
    #getting data from user 
   # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    age=st.text_input('Age of the Person')
    sex=st.text_input('sex(1=male,0=female)')
    cp=st.text_input('The chest pain experienced(Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)')
    trestbps=st.text_input('Resting Blood Pressure')
    chol=st.text_input('Serum Cholestoral in mg/dl')
    fbs=st.text_input('(Fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)')
    restecg=st.text_input('Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes criteria)')
    thalach=st.text_input('Maximum Heart Rate achieved')
    exang=st.text_input('Exercise induced angina (1 = yes; 0 = no)')
    oldpeak=st.text_input('ST depression induced by exercise relative to rest')
    slope=st.text_input('Slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)')
    ca=st.text_input('Major vessels(0-3) colored by flourosopy')
    thal=st.text_input('thal:1 = normal; 2 = fixed defect; 3 = reversable defect')
    
    diagnosis = ''
    
    if st.button('Test Result'):
        diagnosis = heart_disease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
    