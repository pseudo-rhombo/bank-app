#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py
import streamlit as st
import joblib
import numpy as np

# Load trained Decision Tree model
model = joblib.load("decision_tree_model.pkl")

st.title("Bank Customer Default Prediction")
st.write("Enter customer details to predict whether they are likely to default.")

# User Inputs
age = st.number_input("Age", 18, 70, 30)
income = st.number_input("Income", 200, 5000, 1000)
employment_years = st.number_input("Years Employed", 0, 30, 5)
loan_amount = st.number_input("Loan Amount", 500, 20000, 5000)
credit_history = st.selectbox("Credit History", [0, 1])
account_balance = st.number_input("Account Balance", 0, 10000, 1000)

# Predict Button
if st.button("Predict Default"):
    # Arrange input in same order as model trained
    input_features = np.array([[age, income, employment_years, loan_amount, credit_history, account_balance]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Show result
    if prediction == 1:
        st.error("⚠️ Customer is likely to DEFAULT!")
    else:
        st.success("✅ Customer is unlikely to default.")

