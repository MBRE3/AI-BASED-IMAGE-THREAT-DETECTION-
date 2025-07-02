import streamlit as st
import requests
from PIL import Image

st.title("Altered Image & Security Threat Detector")
st.write("Upload an image to check if it has been digitally altered or poses a security threat.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            files = {"image": uploaded_file.getvalue()}
            try:
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Analysis Results")
                    st.write(f"**Threat Detected:** {'Yes' if data['threat_detected'] else 'No'}")
                    st.write(f"**Overall Score:** {data['total_score']}")
                    st.write(f"**Confidence:** {data['confidence']}%")
                    st.write("**Threat Type(s):** " + ", ".join(data['threat_types']))
                    st.markdown("#### Detailed Breakdown:")
                    for key, value in data['individual_scores'].items():
                        st.write(f"**{key.capitalize()}:** {value}")
                else:
                    st.error("Error analyzing image: " + response.text)
            except Exception as e:
                st.error("Connection error: " + str(e))
