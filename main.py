import streamlit as st
import cv2
import numpy as np
from PIL import Image

age_model_path = 'models/age_net.caffemodel'
age_prototxt_path = 'prototxts/age_deploy.prototxt'
gender_model_path = 'models/gender_net.caffemodel'
gender_prototxt_path = 'prototxts/gender_deploy.prototxt'

age_net = cv2.dnn.readNet(age_model_path, age_prototxt_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_prototxt_path)

face_net = cv2.dnn.readNetFromCaffe('prototxts/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

st.markdown("<h1 style='text-align:center; color:#822FEF; background-color: black; border:1; border-radius:20px'>FAC3wise</h1>", unsafe_allow_html=True)

st.markdown("<h6 style='color:#FBDF78; padding-top:20px;'>Welcome to our FAC3wise app developed by KD Sunday and Zin Phyo Min</h6>", unsafe_allow_html=True)
st.markdown("<h4 style='color:white; padding-top:20px;'>Please upload an image with clear face to predict age and gender:</h4>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])


if uploaded_image:
    image = Image.open(uploaded_image)
    image = np.array(image)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    face_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  
            face_detected = True
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (x, y, x1, y1) = box.astype("int")

            face_roi = image[y:y1, x:x1]

            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            age_net.setInput(blob)
            age_predictions = age_net.forward()
            age = age_labels[np.argmax(age_predictions)]

            gender_net.setInput(blob)
            gender_predictions = gender_net.forward()
            gender = gender_labels[np.argmax(gender_predictions)]

            resized_image = Image.fromarray(face_roi)
            resized_image = resized_image.resize((300, 300), Image.ANTIALIAS)
            st.image(resized_image, caption="Predicted Age and Gender", use_column_width=False)
            st.write(f"Predicted Age: {age}")
            st.write(f"Predicted Gender: {gender}")
            st.write("---")

    if not face_detected:
        st.error("No faces detected in the uploaded image. Please upload an image with clear face.")
