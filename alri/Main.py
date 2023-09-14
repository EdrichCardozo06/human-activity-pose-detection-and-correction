import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
import os

## output conversion function
dict  = {1: 'Downdog', 2: 'Goddess', 3: 'Plank',4: 'Tree',5: 'Warrior2'}
def out_conversion(out_array):
    for i in range(5):
        if out_array[i] == 1.:
            return dict[i+1]

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

st.title("yoga pose detecation")
uppload = st.file_uploader("upload image", type=["jpg", "png"])
if uppload is not None:
    if save_uploaded_file(uppload):
        img = cv2.imread(f'uploads/{uppload.name}')
        st.image(img, use_column_width=True)
        img = cv2.resize(img, (150, 150))
        img = np.array(img)
        img = img.reshape(1, 150, 150, 3)
        model = tf.keras.models.load_model("model.h5")
        out = model.predict(img)[0]
        out = out_conversion(out)
        st.write(out)




## reading the image and resizing for the model

# image = cv2.imread(r'D:\codeplayground\New folder (5)\Copy of Yoga Pose Detection\DATASET\TEST\tree\00000003.jpg')
# print(image)
# image = cv2.resize(image,(150,150))

# image = np.array(image)

# image = image.reshape(1,150,150,3)



# ## loading the saved model.h5 file
# model = tf.keras.models.load_model('model.h5')

# ##  Predicting the one hot encoding as the output
# out_arr = model.predict(image)[0]

# ## converting the one hot encoding to the class output
# print(out_conversion(out_arr))
# #print(f'Predicted class is : {out_conversion(out_arr)}')