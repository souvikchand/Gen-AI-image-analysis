import streamlit as st
from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForQuestionAnswering
#from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
from functions import *
import io


#load models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",revision="no_timm")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    sales_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    sales_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

    return {
        "detector": model,
        "processor": processor,
        "clip": clip_model,
        "clip process": clip_processor,
        #"t5 token": t5_tokenizer,
        #"t5": t5_model,
        'story_teller': pipeline("text-generation", model="nickypro/tinyllama-15M"),
        "sales process": sales_processor,
        "sales model": sales_model,
        "device": device
    }



def main():
    st.header("📱 Nano AI Image Analyzer")

    uploaded_file= st.file_uploader("upload image")#, type=['.PNG','png','jpg','jpeg'])
    models= load_models()
    st.write('models loaded')

#im2=detect_objects(image_path=image, models= models)
#st.write(im2)
#st.write("done")
#annotated_image= draw_bounding_boxes(image, im2)
#st.image(annotated_image, caption="Detected Objects", use_container_width=True)

#buttons UI
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.write("Filename:", uploaded_file.name)
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width= False, width=200) 
        
        col1, col2, col3 = st.columns([0.33,0.33,0.33])

        with col1:
            detect= st.button("🔍 Detect Objects", key="btn1")
        with col2:
            describe= st.button("📝 Describe Image", key="btn2")
        with col3:
            story= st.button("📖 Generate Story", key="btn3",
                             help="story is generated based on caption")


        if detect:
            with st.spinner("Detecting objects..."):
                try:
                    detections = detect_objects(image.copy(), models)
                    annotated_image= draw_bounding_boxes(image, detections)
                    st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                    show_detection_table(detections)
                except:
                    st.write("some error!! try another image")

        elif describe:
            with st.spinner("trying to describe..."):
                description= get_image_description(image.copy(),models)
                st.write(description)

        elif story:
            #st.write('btn3 clicked')
            with st.spinner("getting a story..."):
                description= get_image_description(image.copy(),models)
                story= generate_story(description, models)
                st.write(story)

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
                        response = answer_question(image, 
                                                prompt, 
                                                models["sales process"], 
                                                models["sales model"], 
                                                models["device"])
                        #response= "response sample"
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    

if __name__ == "__main__":
    main()