import streamlit as st
import torch
import torchvision

st.set_page_config(page_title='Crop Damage Classification',initial_sidebar_state='collapsed')

st.header('Use Own Image for Prediciton')

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
@st.cache_resource
def load_crop_damage_model():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    class_names = ['Drought','Good (growth)','Nutrient Deficient','Weed','Disease\Pest\Wind']
    automatic_transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
    model = torch.load(r'models//best_5.pt',map_location=device)
    model.eval()
    return model, automatic_transform, device, class_names

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