import streamlit as st
import pages.utils.utils_streamlit as utils_st
import torch

st.header('Use Labeled Images for Prediciton')

@st.cache_resource
def load_crop_damage_model():
    model=utils_st.load_model()
    return model

model = load_crop_damage_model()

c




with st.container(border=True):
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"Main.py")
    with col2:
        if st.button('Use Custom Images',use_container_width = True):
            st.switch_page(r"pages/use_own_images.py")