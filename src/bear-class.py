from fastai.learner import load_learner
from fastai.vision.core import PILImage

import streamlit as st
from PIL import Image

import pathlib
import platform

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

path = pathlib.Path().cwd()


def load_image(image):
    '''
    pass: bytesIO image
    return PILImage object
    '''
    return Image.open(image)



learn_inf = load_learner(path.parent/'export.pkl')

def predict_img(pil_img):
    '''
    Pass: PILImage object
    Return: prediction[str], prediction_idx[int], probabilities[tensor]
    '''

    if pil_img is not None:
        return learn_inf.predict(pil_img)

pic = st.file_uploader('Upload an image')

pred = 'n/a'
pred_idx = 1
probs = []

if pic is not None:
    img = load_image(pic)
    st.image(img)

    pil_img = PILImage.create(pic)

    pred, pred_idx, probs = predict_img(pil_img)

if st.button('classify'):
    f'Prediction: {pred}, Probability: {probs[pred_idx]:.04f}'