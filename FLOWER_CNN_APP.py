import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model=load_model('model30.keras')

st.title('IMAGE CLASSIFICATION MODEL :')
st.text('USING CNN ARCHITECTURE')

category=['tulips','roses', 'daisy', 'dandelion', 'sunflowers']
st.text('Classifies the Image Belonging to The Following Categories ...')
st.text(" 1.Roses \n 2.daisy \n 3.dandelionv\n 4.sunflowers \n 5.tulips")

height=150
width=150

input_img=st.text_input("ENTER IMAGE PATH :",r'.keras\datasets\flower_photos\valA\roses\898102603_2d5152f09a.jpg')
img=tf.keras.utils.load_img(input_img, target_size=(height,width))
imgarr=tf.keras.utils.array_to_img(img)
img_batch=tf.expand_dims(imgarr,0)
pred=model.predict(img_batch)
score=tf.nn.softmax(pred)
res=category[np.argmax(score)]
st.image(input_img, caption="Input Image", use_column_width=False,width=200)
st.text(f'DETECTED IMAGE NAME :'+ res)
acc=np.max(score)*100
st.text(f'ACCURACY :{acc}')