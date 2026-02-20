import streamlit as st
import requests
import base64
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# ==============================
# Load API Key
# ==============================
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    st.error("NVIDIA_API_KEY not found in .env file")
    st.stop()

BASE_URL = "https://ai.api.nvidia.com/v1/genai"

MODELS = {
    "FLUX.1-dev (Text-to-Image)": "black-forest-labs/flux.1-dev",
    "FLUX.1-schnell (Fast)": "black-forest-labs/flux.1-schnell",
    "FLUX.1-Kontext-dev (Image Editing)": "black-forest-labs/flux.1-kontext-dev",
    "Stable Diffusion 3 Medium": "stabilityai/stable-diffusion-3-medium",
}

st.set_page_config(page_title="NVIDIA NIM Image Generator", layout="centered")
st.title("🖼 NVIDIA NIM AI Image Generator")

prompt = st.text_area("Enter Prompt", height=120)

model_name = st.selectbox("Select Model", list(MODELS.keys()))
endpoint = MODELS[model_name]

uploaded_image = None
if "kontext" in endpoint:
    uploaded_image = st.file_uploader(
        "Upload Image for Editing",
        type=["png", "jpg", "jpeg"]
    )

generate = st.button("Generate Image")


def generate_image(prompt, endpoint, uploaded_image=None):

    url = f"{BASE_URL}/{endpoint}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # ------------------------------
    # FLUX.1-dev
    # ------------------------------
    if "flux.1-dev" in endpoint:
        payload = {
            "prompt": prompt,
            "mode": "base",
            "cfg_scale": 3.5,
            "width": 1024,
            "height": 1024,
            "seed": 0,
            "steps": 50
        }

    # ------------------------------
    # FLUX.1-schnell
    # ------------------------------
    elif "flux.1-schnell" in endpoint:
        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "seed": 0,
            "steps": 4
        }

    # ------------------------------
    # FLUX.1-Kontext-dev
    # CORRECT FORMAT
    # ------------------------------
    elif "flux.1-kontext-dev" in endpoint:

        if uploaded_image is None:
            st.error("Please upload an image for Kontext model.")
            return None

        uploaded_image.seek(0)
        image_bytes = uploaded_image.read()

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = uploaded_image.type  # image/jpeg or image/png

        payload = {
            "prompt": prompt,
            "image": f"data:{mime_type};base64,{encoded_image}",
            "steps": 30,
            "cfg_scale": 3.5,
            "seed": 0
        }

    # ------------------------------
    # Stable Diffusion 3 Medium
    # ------------------------------
    elif "stable-diffusion-3-medium" in endpoint:
        payload = {
            "prompt": prompt,
            "cfg_scale": 5,
            "aspect_ratio": "16:9",
            "seed": 0,
            "steps": 50,
            "negative_prompt": ""
        }

    else:
        st.error("Unsupported model selected.")
        return None

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"Error {response.status_code}")
        st.write(response.text)
        return None

    result = response.json()

    # Handle both response formats
    if "image" in result:
        image_base64 = result["image"]
    elif "artifacts" in result:
        image_base64 = result["artifacts"][0]["base64"]
    else:
        st.error("Unexpected response format")
        st.write(result)
        return None

    image_bytes = base64.b64decode(image_base64)
    return image_bytes


if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating image..."):
            img_bytes = generate_image(prompt, endpoint, uploaded_image)

            if img_bytes:
                image = Image.open(BytesIO(img_bytes))
                st.image(image, caption="Generated Image", use_container_width=True)

                st.download_button(
                    label="⬇ Download Image",
                    data=img_bytes,
                    file_name="generated_image.png",
                    mime="image/png"
                )
