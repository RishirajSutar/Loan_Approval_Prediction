import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained Logistic Regression model from a pickle file

model=pickle.load(open('model.pkl','rb'))

# Create a Streamlit web app
st.title('Predict the Chances of your Loan Approval')
st.write(':red[_Enter the Required Details_]')

# Create input fields for user input
Married = st.selectbox('Marital Status' , ['married' , 'not married'])
Credit_History = st.selectbox('Credit_History' , ['Good Credit_History' , 'Bad Credit_History'])
Applicants_Income = st.number_input('Applicants_Income in 1000s' )
LoanAmount = st.number_input('LoanAmount in 100s')

if Married == 'married':
    Married = 1
else:
    Married = 0

if Credit_History == 'Good Credit_History':
    Credit_History = 1
else:
    Credit_History = 0

#Applicants_Income = float(Applicants_Income)
#LoanAmount = float(LoanAmount)

# Create a button to perform prediction
if st.button('Predict'):
    # Create a pandas DataFrame with user input
    user_data = pd.DataFrame({
        'Married': [Married],
        'Credit_History': [Credit_History],
        'CoapplicantIncome': [Applicants_Income],
        'LoanAmount': [LoanAmount]
    })

    prediction = 100 * (model.predict_proba(user_data))
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    st.write("<center><p style='font-size: 20px;'>Chances of your Loan Approval is {} %</p></center>".format(output),
             unsafe_allow_html=True)
