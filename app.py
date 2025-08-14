import numpy as np
import joblib
import streamlit as st

# Loading the saved model

model=joblib.load("trained_models/model.pkl")
scaler=joblib.load("trained_models/scaler.pkl")

# Make a prediction funtion

def diabetes_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

    standardized_data=scaler.transform(input_data_reshape)
    prediction=model.predict(standardized_data)
    # print(prediction)

    if(prediction[0]==1):
      return "The person is diabetic."
    else:
        return "The person is not diabetic."


# Create input fields for the user
def main():
    # Title and description for the web app
    st.title("Diabetes Prediction Web App 🩺")
    st.write("Please enter the patient's data below:")



    #Take the input data from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    # Pregnancies=st.number_input('Number of Pregnancies',min_value=0,max_value=17,value=0)
    # Glucose=st.number_input('Glucose Level (mg/dL)',min_value=0,value=120)
    # BloodPressure=st.number_input('Blood Pressure (mm Hg)',min_value=0,value=70)
    # SkinThickness=st.number_input('Skin Thickness (mm)',min_value=0,value=20)
    # Insulin=st.number_input('Insulin Level (mu U/ml)',min_value=0,value=80)
    # BMI=st.number_input('BMI (Body Mass Index)',min_value=0.0,value=25.0,format="%.2f")
    # DiabetesPedigreeFunction=st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.4, format="%.3f")
    # Age=st.number_input('Age (years)', min_value=0, value=30)
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17, value=0)
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, value=120)
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, value=70)
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, value=20)
    
    with col2:
        Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, value=80)
        BMI = st.number_input('BMI (Body Mass Index)', min_value=0.0, value=25.0, format="%.2f")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.4, format="%.3f")
        Age = st.number_input('Age (years)', min_value=0, value=30)

    
    
    # code for prediction
    diagnosis=''
    
    # create a button for prediction
    if st.button("Diabetes Test Results"):
        # Gather all inputs into a list
        input_data = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age]
        
         # Call the prediction function
        diagnosis = diabetes_prediction(input_data)
        
         # Display the result to the user using st.success
        st.success(diagnosis)
        

    
#  Run the main function
if __name__ == '__main__':
    main()


