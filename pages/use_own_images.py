import streamlit as st
import torch
import torchvision
from PIL import Image

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

def predict(model,X,transform,device,class_names):
    # Transfrom the image
    transformed_image = transform(X)
    # Unsqueeze the image
    pred_image = torch.unsqueeze(transformed_image,dim=0)
    pred_image = pred_image.to(device)

    with torch.inference_mode():
        # Get logits for forward pass
        y_logits = model(pred_image)

        # Get pred
        y_pred_prob = torch.argmax(y_logits,dim=1).item()

        # Predicted class
        predicted_class = class_names[y_pred_prob]

    return predicted_class

model, transform, device, class_names = load_crop_damage_model()

# Get the image using file_uploader widget
image = st.file_uploader(label='Upload an image of an agricultural field',
                 accept_multiple_files=False)

with st.spinner("Prediction Running...Please Wait.."):
    if image!=None:

        # Open the image
        image = Image.open(image).convert('RGB')

        # Display the image
        st.image(image)

        # Send the image for prediction
        pred_class = predict(model,
                                 image,
                                 transform,
                                 device,
                                 class_names)

        st.info(f'Predicted Class: {pred_class}')














with st.container(border=True):
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"Main.py")
    with col2:
        if st.button('Use Labeled Images',use_container_width = True):
            st.switch_page(r"pages/use_labeled_images.py")