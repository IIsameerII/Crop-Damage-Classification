import streamlit as st
import torchvision
from torchvision import datasets
import torch

st.set_page_config(page_title='Crop Damage Classification',initial_sidebar_state='collapsed')

st.header('Use Labeled Images for Prediciton')


@st.cache_resource
def load_crop_damage_model():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    class_names = ['Drought','Good (growth)','Nutrient Deficient','Weed','Disease\Pest\Wind']
    automatic_transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
    model = torch.load(r'models//best_5.pt',map_location=device)
    model.eval()
    return model, automatic_transform, device, class_names

@st.cache_resource
def create_dataset_valid(path):
  valid_dataset = datasets.ImageFolder(path)
  # Not transfroming here. It will be transformed when the image is fed
  # to the prediciton function (single image). This was done to maintain consistency.
  return valid_dataset

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
dataset = create_dataset_valid(r'labeled images')

num_images = int(st.slider(label='Number of Predictions',min_value=1,max_value=5,value=3))

if st.button(label='Run Prediction',use_container_width=True):
    with st.spinner("Prediction Running...Please Wait.."):
        for image in range(0,num_images):
            st.markdown("""---""")
            rand_idx = torch.randint(low=0,high=len(dataset),size=[1,1]).item()
            X,y = dataset[rand_idx]
            pred_class = predict(model,
                                 X,
                                 transform,
                                 device,
                                 class_names)

            with st.container():

                # 2 columns
                col1,col2 = st.columns(2)

                # The first column will show the image
                with col1:
                    st.image(X ,caption=f'Ground Truth: {class_names[y]} | Predicted Class: {pred_class}')

                # The second column will show a if the predicted class matches with the ground truth
                with col2:
                    if pred_class == class_names[y]:
                        st.success(f'✅ Predicted Class: "{pred_class}" is True! ')
                    else:
                        st.error(f'❌ Predicted Class: "{pred_class}" is False.')






with st.container(border=True):
    st.subheader('Go to another page')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Home',use_container_width = True):
            st.switch_page(r"Main.py")
    with col2:
        if st.button('Use Custom Images',use_container_width = True):
            st.switch_page(r"pages/use_own_images.py")