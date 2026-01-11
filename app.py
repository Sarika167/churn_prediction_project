import streamlit as st
import pandas as pd
import joblib

# load model and columns
model = joblib.load("models/churn_model.pkl")
model_cols = joblib.load("models/model_columns.pkl")
