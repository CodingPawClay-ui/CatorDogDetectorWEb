import streamlit as st
from fastai.vision.all import *


st.title(":orange[Pet Breed Classifier!]")
st.text("Hello, this is Clayton")

def extract_breed_name(file_path):
    file_name_parts = file_path.split("/")
    file_path_parts = file_name_parts[len(file_name_parts) - 1].split(".")
    file_path_parts = file_path_parts[0].split("_")
    breed_name = ""
    for i in range(len(file_path_parts)):
        if i != len(file_path_parts) - 1:
            breed_name += file_path_parts[i] +"_"


pet_breed_model = load_learner("pet_breed_model.pkl")

uploaded_file = st.file_uploader("Upload your file here", type=["csv", "txt", "png", "jpg"])
if uploaded_file is not None:
    fastai_img = PILImage.create(uploaded_file)
    prediction = pet_breed_model.predict(fastai_img)

    print(prediction)

    img_label = prediction[0]
    st.text(img_label)
    st.image(uploaded_file, caption = "Uploaded Image", use_container_width=True)