import streamlit as st
import os
import gdown
from PIL import Image
from ImageLoader import ImageClassifier

class_names = [
               'battery', 
               'biological', 
               'cardboard', 
               'clothes', 
               'glass', 
               'metal', 
               'paper', 
               'plastic', 
               'shoes', 
               'trash'
               ]

SentenceBank = ['But... i wouldn\'t bet even a single penny on that!',
                'Maybe yes? But i think it\'s a big No!', 
                'Could be! But i don\'t think so', 
                'Not really confident to recommend', 
                'It\'s likely a Yes!', 
                'Probably Yes :D', 
                'Feels like a yes than a no!', 
                'I\'m pretty solid on this one!', 
                'I\'m really sure about this one!', 
                'Definitely yes! No doubt!']

MODEL_PATH = "Model Weight/Best ConvNeXt.pt"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    link = "https://drive.google.com/uc?id=1KkJEdh12Mi6jYz0rlb_47GDqgFKwcplu"
    gdown.download(link, MODEL_PATH, quiet=False)

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "assets", "App_Icon.png")

st.image(image_path, width=240)

# st.title(":green[Trash]:grey[Net]")
st.markdown("### **:blue[Smart] :green[Trash] Classifier!**")

model = ImageClassifier(MODEL_PATH, device='cpu')

uploaded = st.file_uploader("Upload a waste image here!", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="**Waste Image Uploaded!**", use_column_width=True)

    if st.button("Guess!"):
        class_name, confidence, _ = model.predict(img, class_names)

        st.success(f"It's a picture of **{class_name}** waste!")

        index = int(confidence*100/10)

        st.info(SentenceBank[index])

        with st.expander("Click here to see how confident the AI was!"):

            if confidence > 0.75 :
                st.write(f"The AI was :green[{(confidence*100):.2f}%] sure!")
            elif confidence <= 0.75 and confidence >= 0.5 :
                st.write(f"The AI was :yellow[{(confidence*100):.2f}%] sure!")
            elif confidence <= 0.5 and confidence >= 0.25 :
                st.write(f"The AI was :orange[{(confidence*100):.2f}%] sure!")
            else :

                st.write(f"The AI was :red[{(confidence*100):.2f}%] sure!")





