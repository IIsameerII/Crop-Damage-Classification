import streamlit as st

import torch

st.header('Use Own Image for Prediciton')

@st.cache_resource
def load_crop_damage_model():
    model = torch.load(r'models//best_5.pt')
    return model

model = load_crop_damage_model()














with st.container(border=True):
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"Main.py")
    with col2:
        if st.button('Use Labeled Images',use_container_width = True):
            st.switch_page(r"pages/use_labeled_images.py")