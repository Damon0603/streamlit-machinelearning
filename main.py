import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf


# Create Neuralnetwork

# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense
# from tensorflow.keras.utils import to_categorical
#
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#
# X_train = X_train / 255.0
# X_test = X_test / 255.0
#
# # If you are using categorical_crossentropy loss:
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
#
# model = Sequential([
#     Flatten(input_shape=(32, 32, 3)),
#     Dense(1000, activation="relu"),
#     Dense(10, activation="softmax")
# ])
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))
# model.save('cifar10_model.h5')


def main():
    st.title("Cifar10 Web Classifier")
    st.write("Upload any Image that fits into one of the classes")
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image / 255)
        img_array = img_array.reshape((1, 32, 32, 3))

        model = tf.keras.models.load_model("cifar10_model.h5")
        predictions = model.predict(img_array)
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "frog", "horse", "ship", "truck"]
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0], align="center")
        ax.set_yticks()
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title = ("Cifar 10 Predictions")

        st.pyplot(fig)
    else:
        st.text("you havent uploaded an image yet ")
