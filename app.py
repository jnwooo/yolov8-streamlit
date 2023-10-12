# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import easyocr  
import cv2
import numpy as np

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation', 'Potholes Detection', 'License Plate Detection' , 'License Plate Detection with EasyOCR', 'PPE Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)
elif model_type == 'Potholes Detection':
    model_path = Path(settings.CUSTOM_MODEL1)
elif model_type == 'License Plate Detection':
    model_path = Path(settings.CUSTOM_MODEL2)
elif model_type == 'License Plate Detection with EasyOCR':
    model_path = Path(settings.CUSTOM_MODEL2)
elif model_type == 'PPE Detection':
    model_path = Path(settings.CUSTOM_MODEL3)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

        if source_img is not None and model_type == 'License Plate Detection with EasyOCR':
            if st.sidebar.button('Read License Plate'):
                # Perform OCR on the detected number plate regions
                # Ensure that you have the coordinates of the number plate regions (e.g., 'license_plates')
                reader = easyocr.Reader(['en'], gpu=False)  # You can specify the language

                # Detect license plates
                license_plates = model.predict(uploaded_image, conf=confidence)

                # Create a copy of the original image to modify
                modified_image = np.array(uploaded_image)

                for i, license_plate in enumerate(license_plates[0].boxes.data.tolist()):
                    x1, y1, x2, y2, score, class_id = license_plate

                    # Process license plate
                    license_plate_image_gray = cv2.cvtColor(modified_image[int(y1):int(y2), int(x1):int(x2), :], cv2.COLOR_BGR2GRAY)
                    _, license_plate_image_thresh = cv2.threshold(license_plate_image_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate image
                    detections = reader.readtext(license_plate_image_thresh)

                    if detections:
                        detected_plate_text = detections[0][1]  # Extract the detected text

                        # Replace the class name with the detected license plate text
                        license_plate_text = detected_plate_text

                        # Draw a bounding box around the detected car plate
                        cv2.rectangle(modified_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Calculate the text size
                        text_size, _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)

                        # Calculate the position to center the text
                        text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
                        text_y = int(y1 - 10)

                        # Calculate the background size (adjust as needed)
                        background_width = text_size[0] + 20  # Add some extra space for padding
                        background_height = text_size[1] + 10  # Add some extra space for padding

                        # Calculate the position for the background
                        background_x1 = text_x - 10
                        background_y1 = text_y - text_size[1] - 5  # Adjusted to be higher
                        background_x2 = background_x1 + background_width
                        background_y2 = text_y + 5  # Adjusted to be lower

                        # Draw the enlarged filled background for the text
                        background_color = (0, 0, 0)
                        cv2.rectangle(modified_image, (background_x1, background_y1), (background_x2, background_y2), background_color, -1)

                        # Draw the centered text on the enlarged filled background
                        text_color = (255, 255, 255)  # Text color
                        cv2.putText(modified_image, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, text_color, 2)


                # Convert the modified image back to PIL format for displaying with Streamlit
                image_with_text = PIL.Image.fromarray(modified_image)

                # Display the modified image with updated bounding boxes and license plate text
                st.image(image_with_text, caption='Detected License Plate',
                        use_column_width=True)
          
elif source_radio == settings.VIDEO:
    if model_type == 'License Plate Detection with EasyOCR':
        helper.upload_easyocr(confidence, model)
    else:
        helper.infer_uploaded_video(confidence, model)

elif source_radio == settings.WEBCAM:
    if model_type in ['Detection', 'Segmentation', 'Potholes Detection', 'License Plate Detection', 'PPE Detection']:
        helper.play_webcam(confidence, model)
    else:st.error("Webcam is not supported for License Plate Detection with EasyOCR")

elif source_radio == settings.YOUTUBE:
    if model_type in ['Detection', 'Segmentation', 'Potholes Detection', 'License Plate Detection', 'PPE Detection']:
        helper.play_youtube_video(confidence, model)
    else:st.error("Youtube is not supported for License Plate Detection with EasyOCR")

else:
    st.error("Please select a valid source type!")
