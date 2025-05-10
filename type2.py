import streamlit as st
from PIL import Image, ImageDraw
from transformers import pipeline

# Tiny models only
@st.cache_resource
def load_models():
    return {
        # Tiny object classifier (5MB)
        #'detector': pipeline("image-classification", model="google/mobilenet_v1.0_224"),
        
        # Micro captioning model (45MB)
        #'captioner': pipeline("image-to-text", model="bipin/image-caption-generator"),
        
        # Nano story generator (33MB)
        'story_teller': pipeline("text-generation", model="sshleifer/tiny-gpt2")
    }

def analyze_image(image, models):
    """Combined analysis to minimize model loads"""
    results = {}
    
    # Object classification (not detection)
    with st.spinner("Identifying contents..."):
        results['objects'] = models['detector'](image)
    
    # Image captioning
    with st.spinner("Generating caption..."):
        results['caption'] = models['captioner'](image)[0]['generated_text']
    
    return results

def generate_story(caption, models):
    """Generate short story"""
    return models['story_teller'](
        f"Write a 3-sentence story about: {caption}",
        max_length=100,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']

def main():
    st.title("📱 Nano AI Image Analyzer")
    
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        
        models = load_models()
        analysis = None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analyze", key="analyze"):
                analysis = analyze_image(image, models)
                st.session_state.analysis = analysis
                
                st.subheader("Main Objects")
                for obj in analysis['objects'][:3]:
                    st.write(f"- {obj['label']} ({obj['score']:.0%})")
        
        with col2:
            if st.button("📝 Describe", key="describe"):
                if 'analysis' not in st.session_state:
                    st.warning("Analyze first!")
                else:
                    st.subheader("Caption")
                    st.write(st.session_state.analysis['caption'])
        
        with col3:
            if st.button("📖 Mini Story", key="story"):
                if 'analysis' not in st.session_state:
                    st.warning("Analyze first!")
                else:
                    story = generate_story(
                        st.session_state.analysis['caption'], 
                        models
                    )
                    st.subheader("Short Story")
                    st.write(story)

if __name__ == "__main__":
    main()