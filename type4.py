import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch
import re

@st.cache_resource
def load_detection_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model

def parse_detection_text(detection_text):
    """Robust parsing of detection text with error handling"""
    detections = []
    pattern = r'\[([\d\s,]+)\]\s+([a-zA-Z\s]+)\s+([\d.]+)'
    
    for line in detection_text.split('\n'):
        if not line.strip():
            continue
        
        try:
            match = re.match(pattern, line)
            if match:
                coords = [int(x.strip()) for x in match.group(1).split(',')]
                label = match.group(2).strip()
                score = float(match.group(3))
                
                if len(coords) == 4:
                    detections.append({
                        'box': {'xmin': coords[0], 'ymin': coords[1], 
                                'xmax': coords[2], 'ymax': coords[3]},
                        'label': label,
                        'score': score
                    })
        except (ValueError, AttributeError) as e:
            st.warning(f"Skipping malformed detection line: {line}")
            continue
            
    return detections

def detect_objects(image, processor, model):
    """Run DETR object detection with proper error handling"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=0.7
        )[0]
        
        detection_text = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detection_text += f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}] " \
                            f"{model.config.id2label[label.item()]} {score.item()}\n"
        
        return detection_text, results
        
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
        return "", None

def draw_boxes(image, detections):
    """Draw bounding boxes with different colors for different classes"""
    draw = ImageDraw.Draw(image)
    color_map = {
        'person': 'red',
        'cell phone': 'blue',
        'default': 'green'
    }
    
    for det in detections:
        box = det['box']
        label = det['label']
        color = color_map.get(label.lower(), color_map['default'])
        
        draw.rectangle(
            [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
            outline=color,
            width=3
        )
        draw.text(
            (box['xmin'], box['ymin'] - 15),
            f"{label} ({det['score']:.2f})",
            fill=color
        )
    return image

def main():
    st.title("Object Detection with DETR")
    processor, model = load_detection_model()
    
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                detection_text, results = detect_objects(image, processor, model)
                
                if detection_text:
                    st.subheader("Detection Results")
                    
                    # Show raw detections
                    with st.expander("Raw Detection Output"):
                        st.text(detection_text)
                    
                    # Show parsed results
                    detections = parse_detection_text(detection_text)
                    if detections:
                        annotated_image = draw_boxes(image.copy(), detections)
                        st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                        
                        # Display in table
                        st.subheader("Detected Objects")
                        st.table([
                            {
                                "Object": d["label"],
                                "Confidence": f"{d['score']:.2%}",
                                "Position": f"({d['box']['xmin']}, {d['box']['ymin']}) to ({d['box']['xmax']}, {d['box']['ymax']})"
                            }
                            for d in detections
                        ])
                    else:
                        st.warning("No valid detections found")

if __name__ == "__main__":
    main()