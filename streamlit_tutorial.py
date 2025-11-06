import streamlit as st
from streamlit.elements.image import PILImage

# st.title("Cat or Dog Classifier")
# st.text("Hello this is Clayton")



uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    fastai_img = PILImage.create(uploaded_file)
    prediction = cat_vs_dog_model.predict(fastai_img)

    img_label = None
    confidence_level = prediction[2][1]
    if prediction[0] == 'True':
        if confidence_level >= 0.90:
            img_label = f"I am very confident this is a CAT - {confidence_level}:.2%"
        else:
            img_label = (f"I think this is a CAT but I am not sure - {confidence_level}:.2%")
    else:
        confidence_level = prediction[2][0]
        if confidence_level >= 0.90:
            img_label = f"I am very confident this is a DOG - {confidence_level}:.2%"
        else:
            img_label = (f"I think this is a DOG but I am not sure - {confidence_level}:.2%")


    st.text(img_label)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# This is a tensor of probabilities (or scores) for each class, in the same order as the model's
def extract_breed_name(filename):
    return "_".join(filename.split("/")[-1].split(".")[0].split("_")[:-1])
print(extract_breed_name("Abyssinian_1.jpg"))              # "Abyssinian"
print(extract_breed_name("english_cocker_spaniel_35.jpg")) # "english_cocker_spaniel"
print(extract_breed_name("english_setter_178.jpg"))        # "english_setter"

