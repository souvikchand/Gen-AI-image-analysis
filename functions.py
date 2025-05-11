from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
import torch
import pandas as pd
import streamlit as st
from pathlib import Path

def safe_image_open(uploaded_file):
    try:
        # Convert to lowercase and remove spaces
        filename = Path(uploaded_file.name).stem.lower().replace(" ", "_") + ".png"
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def QA(image, question, models):
    inputs= models['sales process'](image, question, return_tensors= 'pt')
    out = models['sales model'].generate(**inputs)
    return out

def answer_question(image, question, processor, model, device):
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(outputs[0], skip_special_tokens=True)

def generate_story(caption, models):
    """Generate short story"""
    #caption= "a beutiful landscape"
    return models['story_teller'](
        f"Write story about: {caption}",
        max_length=500,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']

def generate_story2(prompt, models):
    input_text = f"Write a short story about {prompt}"
    input_ids = models["t5 token"].encode(input_text, return_tensors="pt", max_length=64, truncation=True)
    output_ids = models["t5"].generate(input_ids, max_length=512)
    story = models["t5 token"].decode(output_ids[0], skip_special_tokens=True)
    return story

def get_image_description(image_path, models):
    image = image_path
    text_inputs = ["a dog", " cat", "a man", "a woman", "a child", "gruop of friends", 
                   "a scenic view", "a cityscape", "a forest", "a beach", "a mountain", "a group of people", "a car", "a bird",
                   "a beautiful landscape", "a couple in love", "an animal", "amazing space",
                   "incridible earth", "motion", "singularity", "anime", "emotions",
                   "sorrow", "joy"]

    inputs = models["clip process"](text=text_inputs, images=image, return_tensors="pt", padding=True)
    outputs = models["clip"](**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best = text_inputs[probs.argmax()]
    return best

def show_detection_table(detection_text):
    """
    Convert detection text into a formatted Streamlit table
    
    Args:
        detection_text: String in format "[x1,y1,x2,y2] label score"
        
    Returns:
        Displays a Streamlit table with columns: Object Type, Box Coordinates, Score
    """
    # Parse each line into a list of dictionaries
    detections = []
    for line in detection_text.strip().split('\n'):
        if not line:
            continue
            
        # Parse the components
        bbox_part, label, score = line.rsplit(' ', 2)
        bbox = bbox_part.strip('[]')
        
        detections.append({
            'Object Type': label,
            'Box Coordinates': f"[{bbox}]",
            'Score': float(score)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(detections)
    
    # Format the score column
    df['Score'] = df['Score'].map('{:.2f}'.format)
    
    # Display in Streamlit with some styling
    st.dataframe(
        df,
        column_config={
            "Object Type": "Object Type",
            "Box Coordinates": "Box [x1,y1,x2,y2]",
            "Score": st.column_config.NumberColumn(
                "Confidence",
                format="%.2f",
            )
        },
        hide_index=True,
        use_container_width=True
    )

def draw_bounding_boxes(image, detection_text):
    """
    Draw bounding boxes on image with different colors for people vs other objects
    
    Args:
        image: PIL Image object
        detection_text: String in format "[x1,y1,x2,y2] label score"
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Define colors
    PERSON_COLOR = (255, 0, 0)    # Red for people
    CAR_COLOR = (255, 165, 0)
    OTHER_COLOR = (0, 255, 0)      # Green for other objects
    TEXT_COLOR = (255, 255, 255)   # White text
    
    # Parse each detection line
    for line in detection_text.strip().split('\n'):
        if not line:
            continue
            
        # Parse the detection info
        bbox_part, label, score = line.rsplit(' ', 2)
        bbox = list(map(int, bbox_part.strip('[]').split(',')))
        confidence = float(score)
        
        # Determine box color
        #box_color = PERSON_COLOR if label == 'person' else OTHER_COLOR
        if label == "person":
            box_color= PERSON_COLOR
        elif label == "car":
            box_color= CAR_COLOR
        else:
            box_color= OTHER_COLOR
        
        # Draw bounding box
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline=box_color,
            width=3
        )
        
        # Draw label with confidence
        label_text = f"{label} {confidence:.2f}"
        text_position = (bbox[0], bbox[1] - 15)
        
        # Draw text background
        text_bbox = draw.textbbox(text_position, label_text)
        draw.rectangle(
            [(text_bbox[0]-2, text_bbox[1]-2), (text_bbox[2]+2, text_bbox[3]+2)],
            fill=box_color
        )
        
        # Draw text
        draw.text(
            text_position,
            label_text,
            fill=TEXT_COLOR
        )
    
    return image

def detect_objects(image_path, models):
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
    """
    image = image_path

    #processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    #model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    processor= models['processor']
    model= models['detector']

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections

def detect_objects4(image, models):
    processor= models['processor']
    model= models['detector']
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

def detect_objects3(image, models, threshold=0.7):
    """Object detection with bounding boxes using DETR"""
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    processor = models['processor']
    model = models['detector']

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Run model
    outputs = model(**inputs)

    # Get original image size (height, width)
    target_size = torch.tensor([image.size[::-1]])

    # Post-process results
    results = processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=threshold)[0]

    # Draw results
    draw = ImageDraw.Draw(image)
    formatted_results = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()
        label_text = model.config.id2label[label.item()]
        score_val = score.item()

        # Draw box
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline="red",
            width=3
        )
        draw.text(
            (box[0], box[1] - 10),
            f"{label_text} ({score_val:.2f})",
            fill="red"
        )

        formatted_results.append({
            "label": label_text,
            "score": score_val,
            "box": {
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3]
            }
        })

    return image, formatted_results


def detect_objects2(image, models):
    """Function 1: Object detection with bounding boxes"""
    results = models['detector'](image)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for result in results:
        box = result['box']
        draw.rectangle(
            [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
            outline="red",
            width=3
        )
        draw.text(
            (box['xmin'], box['ymin'] - 10),
            f"{result['label']} ({result['score']:.2f})",
            fill="red"
        )
    return image, results


"""@st.cache_resource
def load_light_models():
    #Load lighter version of models with proper DETR handling
    models = {}
    
    # Load DETR components separately
    with st.spinner("Loading object detection model..."):
        models['detr_processor'] = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        models['detr_model'] = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    # Use pipeline for captioning
    with st.spinner("Loading captioning model..."):
        models['captioner'] = pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-base"
        )
    
    return models"""

"""@st.cache_resource
def load_models():
    return {
        # Using tiny models for faster loading
        'detector': pipeline("object-detection", model="hustvl/yolos-tiny")
        #'captioner': pipeline("image-to-text", model="Salesforce/blip-image-captioning-base"),
        #'story_teller': pipeline("text-generation", model="gpt2")
    }"""