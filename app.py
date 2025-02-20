import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

diabetes_model_path = r"diabetes_model.sav"
diabetes_model = pickle.load(open(diabetes_model_path,'rb'))

st.title ('Diabetes Prediction using ML')


st.title('Diabetes Prediction using ML')

col1,col2,col3=st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col1:
    SkinThickness = st.text_input('SkinThickness')

with col2:
    Insulin = st.text_input('Insulin')

with col3:
    BMI = st.text_input('BMI')

with col1:
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')

with col2:
    Age = st.text_input('Age')

with col3:
    BloodPresure = st.text_input('BloodPresure')

#prediction results
diab_diagnosis =''

if st.button('Diabetes test button'):
    try:
        user_input = [float(Pregnancies), float(Glucose), float(BloodPresure),float(SkinThickness), float(Insulin),
                      float(BMI), float(DiabetesPedigreeFunction), float(Age)]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = "The patient is likely to have diabetes."
        else:
            diab_diagnosis = "The patient is not likely to have diabetes."

        st.success(diab_diagnosis)

    except Exception as e:
        st.error(f"An error occurred: {e}")
if st.button('Show Model Acurracy'):
 

 test_data=pd.read_csv(r"C:\\Users\\sowja\\OneDrive\\Desktop\\project1\\diabetes.csv")

 x_test=test_data.drop(columns=["Outcome"])
 y_test=test_data["Outcome"]

 y_pred=diabetes_model.predict(x_test)

 accuracy=accuracy_score(y_test,y_pred)

 st.write(f"Model Accuracy on test Data:{accuracy * 100:.2f}%")
