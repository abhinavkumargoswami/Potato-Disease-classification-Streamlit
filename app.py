
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
st.title('Potato Disease Classifier')
@st.cache(allow_output_mutation=True)
def get_model():
    model = load_model("models/potatoes.h5")
    return model
def predict(img):
    model = get_model()
    class_names = ['Early_blight', 'Late_blight', 'healthy']
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return "CLASS: " + str(predicted_class) + "\n" + "  (" + str(confidence) + "%)"

def main():
    file_uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])   
    if st.button("Classify"):
        if file_uploaded is None:
            st.write("Please upload an image")
        else:
            image = Image.open(file_uploaded)
            with st.spinner('Model working....'):
                predictions = predict(image)
                st.write(predictions)
if __name__ == "__main__":
    main()

