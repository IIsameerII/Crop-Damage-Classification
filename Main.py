
# Now you can import your_script (or any other module in that directory)

import streamlit as st

st.set_page_config(page_title='Crop Damage Classification',initial_sidebar_state='collapsed')


st.title('Crop Damage Classification')

st.markdown('<div style="text-align: justify;">In the realm of agriculture, crop health and yield are pivotal for the sustenance of food supply chains and the economy. However, crops are highly vulnerable to various threats like drought, nutrient deficiencies, weeds, diseases, pests, and environmental factors. Timely and accurate assessment of crop health is crucial to mitigate these threats. This project, initiated by an insurance company, plays a critical role in revolutionizing how crop damages are assessed and addressed.</div>', unsafe_allow_html=True)
st.write('')
st.subheader('1. Rapid Damage Assessment')
st.markdown('<div style="text-align: justify;">The conventional methods of crop damage assessment involve manual inspections which are time-consuming and often subject to human error. This project aims to significantly decrease the time required to handle insurance cases by utilizing mobile photography for swift and accurate damage classification. Rapid assessments can lead to quicker insurance claim processing, benefiting both farmers and insurance companies.</div>', unsafe_allow_html=True)
st.subheader('2. Enhancing Precision in Damage Identification:')
st.markdown('<div style="text-align: justify;">By classifying crop damage into specific categories like drought impact, nutrient deficiency, weed infestation, and other damages (including diseases, pests, and wind), the project seeks to provide a more nuanced understanding of crop health. This precision is critical for implementing targeted remedial measures and improving crop management practices.</div>', unsafe_allow_html=True)
st.subheader('3. Data-Driven Insights for Farmers')
st.markdown('<div style="text-align: justify;">The project also aims to empower farmers with valuable insights regarding the health of their crops. By analyzing images taken from mobile phones, farmers can receive real-time feedback on their crop\'s condition, enabling them to take immediate and appropriate actions.</div>', unsafe_allow_html=True)
st.subheader('4. Advancing Agricultural Insurance Practices')
st.markdown('<div style="text-align: justify;">For the insurance sector, this project is a step towards modernizing agricultural insurance practices. By automating and improving the accuracy of damage assessment, insurance companies can process claims more efficiently and accurately, reducing the risk of fraud and errors.</div>', unsafe_allow_html=True)
st.subheader('5. Contribution to Sustainable Agriculture')
st.markdown('<div style="text-align: justify;">By providing early detection of crop issues, the project contributes to sustainable agricultural practices. Early intervention can reduce the excessive use of water, fertilizers, and pesticides, promoting environmentally friendly farming.</div>', unsafe_allow_html=True)
st.subheader('6. Economic Impact')
st.markdown('<div style="text-align: justify;">Improved crop damage assessment and management can significantly impact the agricultural economy. It helps in reducing losses, ensuring better yields, and stabilizing the income of farmers, which is vital for the economic health of regions dependent on agriculture.</div>', unsafe_allow_html=True)
st.subheader('7. Scalability and Accessibility')
st.markdown('<div style="text-align: justify;">Utilizing mobile technology for crop damage assessment makes the solution highly scalable and accessible. Farmers in remote areas can benefit from this technology, as it only requires a smartphone, which is increasingly common even in less developed areas.</div>', unsafe_allow_html=True)
st.write('')
st.markdown('<div style="text-align: justify;">This project represents a synergistic approach where technology meets agriculture to create a sustainable, efficient, and farmer-friendly future. It not only aims to streamline insurance processes but also stands as a testament to the potential of technology in transforming agricultural practices for the betterment of farmers, the industry, and the environment.</div>', unsafe_allow_html=True)
st.write('')
st.write('')
with st.container(border=True):
    st.subheader('Try our solution for prediction')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Use Labeled Images',use_container_width = True):
            st.switch_page(r"pages/use_labeled_images.py")
    with col2:
        if st.button('Use Custom Images',use_container_width = True):
            st.switch_page(r"pages/use_own_images.py")

    

st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True) # Template Markdown

