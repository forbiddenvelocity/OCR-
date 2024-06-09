import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

import cv2
import numpy as np

def preprocess_image(image):
    try:
        image_np = np.array(image)

        # Convert to RGB if it's a grayscale image
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # Handle RGBA images
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
     
        # Noise reduction
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
      
        # Threshold to get binary image
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)
        binary_image = cv2.erode(binary_image, kernel, iterations=1)
   
        # Convert back to 3 channel image for the model
        binary_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
   
        return binary_image_rgb  # Return as numpy array directly
    except Exception as e:
        print(f"Failed to preprocess image: {str(e)}")
        return None
