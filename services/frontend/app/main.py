from PIL import Image
import streamlit as st
import requests
import json

BASE_URL="http://backend:8080/"

st.title("Cat vs. Dog - DenseNet")

st.header('This is a header')

image = st.file_uploader(
    label="Choose an image",
    type=['jpg','jpeg','png']
)

if image is not None:
    import requests

    headers = {
        'accept': 'application/json',
    }

    files = { 'file': (f"{image.name};type={image.type}",
                       image.getvalue()) }

    st.image(Image.open(image))
    
    predict_btn = st.button("Predict")
    if predict_btn:
        files = { "file": image.getvalue(), }
        predict = requests.post(
            url=f"{BASE_URL}predict",
            headers=headers,
            files=files
        )

        prediction = predict.json().get('data').get('prediction')

        st.header(prediction)
