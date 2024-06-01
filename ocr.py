import cv2
import numpy as np
from PIL import Image
import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# Enhanced preprocessing function
def preprocess_image(image):
    try:
        # Convert PIL image to numpy array
        image = np.array(image)
        st.write(f"Original image shape: {image.shape}")

        # Convert to RGB if it's a grayscale image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # Handle RGBA images
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        st.write(f"Image shape after conversion to RGB: {image.shape}")

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        st.write(f"Image shape after conversion to grayscale: {gray_image.shape}")

        # Apply GaussianBlur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply thresholding to get binary image
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st.write(f"Image shape after thresholding: {binary_image.shape}")

        # Perform morphological operations to remove small noises
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Optionally, apply dilation and erosion to enhance text features
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)
        binary_image = cv2.erode(binary_image, kernel, iterations=1)

        # Convert back to 3 channel image for the model
        binary_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        st.write(f"Final preprocessed image shape: {binary_image_rgb.shape}")

        # Convert back to PIL image
        binary_image_rgb = Image.fromarray(binary_image_rgb)

        return binary_image_rgb
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

# Function to extract text from image using the OCR model
def extract_text_from_image(image):
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        st.error(f"Error in extracting text from image: {e}")
        return None

# Main function for the Streamlit app
def main():
    st.title("CAPTCHA Text Extractor")
    uploaded_file = st.file_uploader("Choose a CAPTCHA image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded CAPTCHA Image', use_column_width=True)
            
            st.write("Processing...")
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                extracted_text = extract_text_from_image(preprocessed_image)
                if extracted_text is not None:
                    st.write(f"Extracted Text: {extracted_text}")
                else:
                    st.error("Failed to extract text from the image.")
            else:
                st.error("Failed to preprocess the image.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
