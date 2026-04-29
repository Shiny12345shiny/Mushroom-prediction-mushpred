import numpy as np
import streamlit as st
import pandas as pd
import joblib as jb
from label import encode
st.set_page_config(page_title="Mushroom Prediction")
st.title("Mushroom Prediction App 🍄")
st.markdown("lets find out you mushrooms are edible or poisonous ")
df = pd.read_csv("mushrooms (2).csv")
st.dataframe(df)
data=df.filter(['class', 'cap-surface'])
st.line_chart(data)
uploaded_file = st.file_uploader("Upload your file", type=["csv"])
if uploaded_file:
    df=pd.read_csv(uploaded_file)
    st.dataframe(df)
    kbest = jb.load("kbest.pkl")
    st.header("your file encoding is done")
    
    data=encode(df)
    st.dataframe(data)
    model=jb.load("model.pkl")
    df_selected = kbest.transform(data)
    predict = pd.DataFrame(model.predict(df_selected))
    predict.columns=['Results']
    result=predict.replace({0:"edible",1:"poisonous"})
    st.header("your file prediction is done")
    st.dataframe(result)
    csv=result.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="Download prediction",
    data=csv,
    file_name="prediction.csv",
    mime="text/csv"
    )
    st.markdown("**Note**: The predictions are based on the features provided in the uploaded file. Please ensure that the file format and features match the expected input for accurate predictions.")