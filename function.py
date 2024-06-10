from PIL import Image
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("./results3/final_model")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def preprocess(image):
    image_np = np.array(image.convert("RGB"))
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    _,binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)

    processed_image = Image.fromarray(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB))
    return processor(processed_image, return_tensors = 'pt').pixel_values

def extract_text(pixel_values):
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def extracted_text_from_image(image_path):
    image = Image.open(image_path)
    pixel_values = preprocess(image)
    extracted_text = extract_text(pixel_values)
    return extracted_text

image_path = "train2\captcha_image_600.png"
extracted_text = extracted_text_from_image(image_path)
print(f"Extracted Text: {extracted_text}")
