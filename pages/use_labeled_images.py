import streamlit as st

st.write('labeled_images')








with st.container():
    st.header('Try our solution for prediction')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home'):
            st.switch_page(r"main_page.py")
    with col2:
        if st.button('Use Custom Images'):
            st.switch_page(r"pages/use_own_images.py")