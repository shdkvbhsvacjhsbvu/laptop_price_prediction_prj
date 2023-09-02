import streamlit as st
import pickle
import numpy as np


pipe = pickle.load(open('pipe.pkl','rb'))
d = pickle.load(open('df.pkl','rb'))

st.title("Laptop_price_predicter")
Company = st.selectbox('Brand',d['Company'].unique())
Type = st.selectbox('Type',d['TypeName'].unique())

Ram = st.selectbox('RAM_in_GB',[0,4,8,16,24,32])
weight= st.number_input('enter weight')
touchscreen =st.selectbox('Touchscreen',['YES','NO'])
ips=st.selectbox('IPS',['NO','YES'])
screen_size =st.number_input('Screen Size')
resolution =st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu =st.selectbox('cpu brand',d['cpu_brands'].unique())
Hdd =st.selectbox('HDD_in_GB',[0,128,256,512,1024,2048])
SSD =st.selectbox('SSD_in_GB',[0,8,128,256,512,1024])

GPU =st.selectbox('GPU_brands',d['Gpu_brand'].unique())
os =st.selectbox('os',d['OS'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'YES':
        touchscreen = 1
    else :
        touchscreen = 0

    if ips =='YES':
        ips= 1
    else :
        ips =0
    x_res= int(resolution.split('x')[0])
    y_res= int(resolution.split('x')[1])
    ppi=(x_res**2 + y_res**2)*0.5/screen_size
    query = np.array([Company,Type,Ram,weight,touchscreen,ips,ppi,cpu,Hdd,SSD,GPU,os])

    query = query.reshape(1,12)
    st.title(np.exp(pipe.predict(query)))
