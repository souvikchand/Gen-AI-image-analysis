import streamlit as st
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration,BlipForQuestionAnswering
import torch

@st.cache_resource
def load_blip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return processor, model, device

def answer_question(image, question, processor, model, device):
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
def main():
    st.title("Image Chat Assistant")
    
    # Load model
    processor, model, device = load_blip_model()
    
    # Image upload
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        col1, col2, col3 = st.columns([0.33,0.33,0.33])

        with col1:
            detect= st.button("🔍 Detect Objects", key="btn1")

        with col2:
            describe= st.button("📝 Describe Image", key="btn2")
        with col3:
            story= st.button("📖 Generate Story", key="btn3")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        chat_container = st.container(height=400)
        with chat_container:
        
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
            if prompt := st.chat_input("Ask about the image"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
            
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = answer_question(image, prompt, processor, model, device)
                        #response= "response sample"
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()