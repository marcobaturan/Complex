################################################################################
#                                                                              #
#  PICTELL: Web App for small multi-modal model tasks.                         #
#                                                                              #
################################################################################

"""
PICTELL:

    Web app for image description by computer vision and language,
    text-to-speech (TTS) and speech-to-text (STT) capabilities.
    text-to-image (T2I) and image-to-text (I2T) capabilities.
    author: @MarcoBaturan

"""

# Import necessary modules and libraries
try:
    import sys
    import streamlit as st
    from io import BytesIO, StringIO
    import moondream as md
    from PIL import Image
    import time
    import os
    import pyttsx3
    import threading
    from contextlib import contextmanager
    from streamlit_mic_recorder import mic_recorder, speech_to_text
    from PIL import Image
    import time
    from huggingface_hub import InferenceClient
    from llama_cpp import Llama
    print("All modules loaded.")
except Exception as e:
    print("Some modules are missing: " + str(e))

# Initialize the text-to-speech engine
engine = None
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Context manager to handle the text-to-speech engine
@contextmanager
def get_engine():
    global engine
    try:
        if engine is None:
            engine = pyttsx3.init()
            """ RATE"""
            engine.setProperty('rate', 125)  # Setting up new voice rate
        yield engine
    finally:
        pass

# Load the model using Streamlit's caching mechanism
@st.cache_resource
def load_model():

    return md.vl(model=os.path.join(base_dir, "Models", 'moondream-0_5b-int4.mf.gz'))

# Function to speak text asynchronously
def speak_async(text):
    def run_speak():
        with get_engine() as engine:
            engine.say(text)
            engine.runAndWait()
    thread = threading.Thread(target=run_speak)
    thread.start()
    return thread

# Main function to run the Streamlit app
def main():
    # Configure the Hugging Face client
    client = InferenceClient(
        provider="hf-inference",
        api_key=""  # Replace with your API Key
    )

    llm = Llama(
                model_path = os.path.join(base_dir, "Models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
                n_ctx=2048,
                n_threads=8,
                n_gpu_layers=35)
    st.title("Orthopedic AI")
    "Run this function to display the streamlit app"
    tab1, tab2, tab3 = st.tabs(["Image to Text", "Text to Image", "Text to Speech"])
    with tab1:
        st.text("Image to Text")
        # Define the upload folder for images
        UPLOAD_FOLDER = "Uploaded pictures"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Display the app information
        st.info(__doc__)

        # Load the model
        model = load_model()

        # Display the title of the app
        st.markdown("<h1 style='text-align: center; color: red;'>Image Description</h1>", unsafe_allow_html=True)

        # File uploader for images
        file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        show_image = st.empty()

        # Check if a file is uploaded
        if not file:
            show_image.info("Please upload an image")

        if file is not None:
            # Get the file extension
            file_extension = file.name.split(".")[-1].lower()
            temp_file = f"temp_image.{file_extension}"

            # Define the complete route for the folder
            complete_route = os.path.join(UPLOAD_FOLDER, temp_file)

            # Save the file in the folder
            with open(complete_route, "wb") as f:
                f.write(file.getbuffer())

            # Display the image
            show_image.image(file)

            # Open the image for analysis
            image = Image.open(complete_route)

            # Encode the image
            encoded_image = model.encode_image(image)

            # Get the caption of the image from chunks
            full_caption = ''
            for chunk in model.caption(encoded_image, stream=True)["caption"]:
                full_caption += chunk

            # Display the description of the picture
            st.subheader("Description of the picture:")
            st.write(full_caption)

            # Speak the description asynchronously
            speak_thread = speak_async(full_caption)
    with tab2:
        # Title of the application
        st.title("Text-to-Image Generator")

        # Input box for user text description
        description = st.text_input("Enter a description for the image:")
        # Button to trigger image generation
        if st.button("Generate Image"):
            if description:
                with st.spinner("Generating image..."):
                    # Generate image from text input
                    image = client.text_to_image(
                        description,
                        model=os.path.join(base_dir, "Models", "Storyboard_sketch.safetensors")
                    )
                    # Display the generated image
                    st.image(image, caption="Generated Image", use_column_width=True)
            else:
                # Warning message if no description is entered
                st.warning("Please enter a description before generating the image.")
    with tab3:
        st.text('Speech to text')
        state = st.session_state

        if 'text_received' not in state:
            state.text_received = []
        c1, c2 = st.columns(2)
        with c1:
            st.write("Convert speech to text, in English.")
        with c2:
            text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

        if text:
            state.text_received.append(text)

        for text in state.text_received:
            st.text('You said')
            st.text(text)
            messages = llm.create_chat_completion(
                      messages = [
                        {
                          "role": "system",
                          "content": "You are story writing assistant"

                        },
                        {
                          "role": "user",
                          "content": text
                        }
                      ]
                )
            st.text('I tell you:')
            st.write(messages['choices'][0]['message']['content'])

        st.write("Record your voice, and play the recorded audio:")
        audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

        if audio:
            st.audio(audio['bytes'])



# Entry point of the script
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
