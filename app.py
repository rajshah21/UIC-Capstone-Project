import streamlit as st
from PIL import Image
from diffusers import DiffusionPipeline
import io
import time
import base64
from io import BytesIO
import torch

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Load your images (replace 'image1.png' and 'image2.png' with your actual image file paths or URLs)
# If images are URLs, you can use requests library to fetch and display them directly
image1 = Image.open("UIC.png")
image2 = Image.open("ccc.png")

# Display "In Association with:" and the images in a single row
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <h3 style="margin-right: 10px;">In Association with:</h3>
        <img src="data:image/png;base64,{}" style="height: 50px; margin-right: 10px;">
        <img src="data:image/png;base64,{}" style="height: 50px;">
    </div>
    """.format(image_to_base64(image1), image_to_base64(image2)),
    unsafe_allow_html=True
)

# CSS to increase the width of the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 350px;  /* Adjust the width as desired */
    }
    /* Increase the font size of slider titles */
    .stSlider label {
        font-size: 50px;  /* Adjust the font size as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Generate Synthetic Vehicle Damage Images")

# Sidebar for Adaptor Weights Menu
st.sidebar.title("Adaptor Weights")
st.sidebar.write("Set the adaptor weights for each damage type (values between 0 and 1):")
flat_tire = st.sidebar.slider("Flat Tire", 0.0, 1.0, 0.5)
glass_shatters = st.sidebar.slider("Glass Shatters", 0.0, 1.0, 0.5)
scratches = st.sidebar.slider("Scratches", 0.0, 1.0, 0.5)
dents = st.sidebar.slider("Dents", 0.0, 1.0, 0.5)

# Main page for Prompt Input and Generate Button
prompt = st.text_input("Enter a prompt:")
generate_button = st.button("Generate Image")

#Function to Load the Diffusion Model with LoRA weights
def load_model():

    torch.cuda.empty_cache()
    
    #Loading the Diffusion Model
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to('cuda')

    #Loading Flat Tire LoRA
    pipe.load_lora_weights('IDS-560-Group-3/stable-diffusion-flat-tire-lora-v2', weight_name='pytorch_lora_weights.safetensors', adapter_name='flat-tire')

    #Loading Scratches LoRA
    pipe.load_lora_weights('IDS-560-Group-3/stable-diffusion-scratches-lora-v3', weight_name='pytorch_lora_weights.safetensors', adapter_name='scratches')

    #Loading Dents LoRA
    pipe.load_lora_weights('IDS-560-Group-3/stable-diffusion-dents-lora-v3', weight_name='pytorch_lora_weights.safetensors', adapter_name='dents')

    #Loading Glass Shatter LoRA
    pipe.load_lora_weights('IDS-560-Group-3/stable-diffusion-glass-shatter-lora-v3', weight_name='pytorch_lora_weights.safetensors', adapter_name='glass-shatter')

    return pipe

# Function to Generate Images based on Prompt and Adapter Weights
def generate_image(pipeline, prompt, flat_tire, glass_shatters, scratches, dents):
    # Set adapter weights and generate image
    pipeline.set_adapters(['flat-tire','scratches','dents','glass-shatter'], adapter_weights=[flat_tire, scratches, dents, glass_shatters])
    image = pipeline(prompt).images[0]

    
    return image

# Initialize the model on app startup
if 'pipe' not in st.session_state:
    st.session_state.pipe = load_model()


# Generate and display image when the button is clicked
if generate_button:
    st.write("Generating image, please wait...")
    
    # Timer to show processing time
    start_time = time.time()
    generated_image = generate_image(st.session_state.pipe, prompt, flat_tire, glass_shatters, scratches, dents)
    end_time = time.time()
    st.write(f"Image generated in {end_time - start_time:.2f} seconds")
    
    # Display the generated image
    st.image(generated_image, caption="Generated Output Image")

    