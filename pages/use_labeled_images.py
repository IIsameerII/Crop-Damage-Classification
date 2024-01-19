import streamlit as st

st.header('Use Labeled Images for Prediciton')








with st.container():
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"Main.py")
    with col2:
        if st.button('Use Custom Images',use_container_width = True):
            st.switch_page(r"pages/use_own_images.py")