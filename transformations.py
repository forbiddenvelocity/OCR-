from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)  # Convert to range [0, 255]
])
