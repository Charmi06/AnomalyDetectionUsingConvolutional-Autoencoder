import gradio as gr
import tensorflow as tf
import numpy as np
from model import ConvolutionalAutoencoder
from PIL import Image


#To detect anomaly, the autoencoder model reconstructs the input image and calculates the 
# difference between the original and reconstructed images. 
# The difference between the two images is called the reconstruction error.

# Define the predict_image function with anomaly detection
def predict_image(inp, threshold):
    autoencoder = ConvolutionalAutoencoder(num_neuron=256, kernal1=32, kernal2=16, shape=(32, 32, 3))
    autoencoder(np.zeros((1, 32, 32, 3)))  # Call the model to create its variables
    autoencoder.load_weights("model.h5")

    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    if inp.mode != "RGB":
        inp = inp.convert("RGB")
    inp = inp.resize((32, 32))
    inp_arr = np.array(inp)/255.0
    inp_arr = np.expand_dims(inp_arr, axis=0)

    try:
        # Get the reconstructed image
        pred = autoencoder.predict(inp_arr)[0]

        # Calculate the reconstruction error
        error = np.mean(np.abs(inp_arr - pred))

        # Set the output image to red if the error is above the threshold
        if error > threshold:
            output_img = Image.open("icon-detect1.png")
            is_anomaly = "Anomaly detected!"
        else:
            output_img = Image.fromarray((pred*255.0).astype(np.uint8))
            is_anomaly = False
        return output_img, error, is_anomaly

    except Exception as e:
        return None, None, None

# Define the Gradio interface with anomaly detection
iface = gr.Interface(
    fn=predict_image,
    inputs=[gr.inputs.Image(shape=(None, None)), gr.inputs.Slider(0.4, 2.0, 0.1, label="Anomaly Threshold")],
    outputs=[gr.outputs.Image(type="pil"), "number", "text"],
    output_names=["Output Images", "Error", "Is Anomaly"]
)

# Launch the interface
iface.launch()