import json

# Create a dictionary to hold the file names and empty strings
captcha_images = {f"captcha_image_{i}.png": "" for i in range(601, 1000)}

# Convert the dictionary to a JSON formatted string
json_output = json.dumps(captcha_images, indent=4)

# Printing the JSON output
print(json_output)
