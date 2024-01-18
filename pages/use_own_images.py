import streamlit as st

st.write('own_images')













with st.container():
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"main_page.py")
    with col2:
        if st.button('Use Labeled Images',use_container_width = True):
            st.switch_page(r"pages/use_labeled_images.py")