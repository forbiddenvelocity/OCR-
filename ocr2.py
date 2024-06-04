from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# Load the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Load an image
image_url = "https://southandaman.dcourts.gov.in/?_siwp_captcha&id=e75k3kdg5npa6mx2zxhpxfh41uyb2k01pme35m8h"
image = Image.open(requests.get(image_url, stream=True).raw)

# Preprocess the image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate the output
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
