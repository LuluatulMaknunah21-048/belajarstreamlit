import streamlit as st
import pandas as pd
from PIL import Image
import pickle
iris=pickle.load(open('naive.pkl','rb'))
#judul web
st.title('PREDIKSI DATA IRIS')
gambar = Image.open('iris.png')
st.image(gambar, width=500)

st.sidebar.header('Parameter inputan')


def inputan():
    sepal_length=st.sidebar.slider('Masukkan nilai sepal length',0.1,9.9,5.4)
    sepal_width=st.sidebar.slider('Masukkan nilai sepal width',0.1,9.9,3.9)
    petal_length=st.sidebar.slider('Masukkan nilai petal length',0.1,9.9,1.7)
    petal_width=st.sidebar.slider('Masukkan nilai petal width',0.1,9.9,0.4)
    data={'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width}
    fitur=pd.DataFrame(data,index=[0])
    return fitur
df=inputan()
button=st.button('prediksi')
if button :
    st.subheader('Parameter Inputan')
    st.write(df)
    prediksi=iris.predict(df)
    for pred in prediksi:
        st.write('Species: ', pred)
