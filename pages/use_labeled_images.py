import streamlit as st
from torchvision import datasets
import torch

st.header('Use Labeled Images for Prediciton')

@st.cache_resource
def load_crop_damage_model():
    model = torch.load(r'models\best_5.pt')
    return model

@st.cache_resource
def create_dataset_valid(path):
  valid_dataset = datasets.ImageFolder(path)
  # Not transfroming here. It will be transformed when the image is fed
  # to the prediciton function (single image). This was done to maintain consistency
  return valid_dataset

model = load_crop_damage_model()
dataset = create_dataset_valid(r'labeled images')







with st.container(border=True):
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"Main.py")
    with col2:
        if st.button('Use Custom Images',use_container_width = True):
            st.switch_page(r"pages/use_own_images.py")