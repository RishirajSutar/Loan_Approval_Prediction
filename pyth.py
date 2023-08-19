#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import pandas as pd
pip install sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np

data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
data['Education'] = data['Education'].replace(['Graduate', 'Not Graduate'] , [1.0 , 0.0])
mode_value = data['Education'].mode()[0]

# Replace the null values with the mode value
data['Education'].fillna(mode_value, inplace=True)

data['ApplicantIncome'] = data['ApplicantIncome'].astype(float)
data = data.drop('Loan_ID' , axis = 1)

data['Gender'] = data['Gender'].replace(['Male', 'Female'] , [1.0 , 0.0])
data['Gender'].unique()

mode_value = data['Gender'].mode()[0]

# Replace the null values with the mode value
data['Gender'].fillna(mode_value, inplace=True)


data['Married'] = data['Married'].replace(['No', 'Yes'] , [0.0 , 1.0])

mode_value = data['Married'].mode()[0]

# Replace the null values with the mode value
data['Married'].fillna(mode_value, inplace=True)


data['Self_Employed'] = data['Self_Employed'].replace(['No', 'Yes'] , [0.0 , 1.0])

mode_value = data['Self_Employed'].mode()[0]

# Replace the null values with the mode value
data['Self_Employed'].fillna(mode_value, inplace=True)


data['Property_Area'] = data['Property_Area'].replace(['Urban', 'Rural', 'Semiurban'] , [2.0 , 0.0 , 1.0])

mode_value = data['Property_Area'].mode()[0]

# Replace the null values with the mode value
data['Property_Area'].fillna(mode_value, inplace=True)


data['Loan_Status'] = data['Loan_Status'].replace(['Y', 'N'] , [1.0 , 0.0])

mode_value = data['Loan_Status'].mode()[0]

# Replace the null values with the mode value
data['Loan_Status'].fillna(mode_value, inplace=True)


data['Dependents'] = data['Dependents'].replace(['0', '1', '2', '3+'] , [0.0 , 1.0 , 2.0 , 3.0])

mode_value = data['Dependents'].mode()[0]

# Replace the null values with the mode value
data['Dependents'].fillna(mode_value, inplace=True)


mode_value = data['Credit_History'].mode()[0]

data['Credit_History'].fillna(mode_value , inplace=True)

#imputations in continuous variables

med_value = data['LoanAmount'].median()

data['LoanAmount'].fillna(med_value , inplace=True)


med_value = data['Loan_Amount_Term'].median()

data['Loan_Amount_Term'].fillna(med_value , inplace=True)


#Seperating response and Predictors

y = data['Loan_Status']
X = data.drop('Loan_Status' , axis=1 , inplace=False)

predictors = list(X.columns)


#Splitting the Data into trainig and testing sets


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42 , stratify = y)



X1 = ['Married'  , 'Credit_History' , 'CoapplicantIncome' , 'LoanAmount']

#fitting a logistic Model
logisticRegr = LogisticRegression(max_iter=4000)
logisticRegr.fit(X_train[X1], y_train)

import warnings
import pickle
warnings.filterwarnings("ignore")

pickle.dump(logisticRegr,open('model.pkl','wb'))
