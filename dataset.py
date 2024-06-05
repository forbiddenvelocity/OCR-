import os
import cv2
import pandas as pd
import json
from preprocessing import preprocess_image  # Ensure this is the correct import statement

def create_dataset(captcha_folder, output_folder, label_file, annotation_json):
    # Load labels from JSON file
    with open(annotation_json, 'r') as f:
        labels = json.load(f)

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List to hold the data for the dataframe
    data = []

    for filename, label in labels.items():
        image_path = os.path.join(captcha_folder, filename)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            preprocessed_image = preprocess_image(image)  # Ensure your preprocessing function is correctly imported
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, preprocessed_image)
            
            # Append the filename and the label to the data list
            data.append([filename, label])
        else:
            print(f"Warning: {image_path} does not exist and will be skipped.")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['filename', 'label'])
    df.to_csv(label_file, index=False)

# Example usage
create_dataset(
    captcha_folder="evaluation_data",
    output_folder="processed_evaluation_images",
    label_file="evaluationlabels.csv",
    annotation_json="evaluation_annotations.json"
)
