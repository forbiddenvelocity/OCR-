import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from captchadataset import CaptchaDataset
from transformations import transform

# Load pre-trained TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("./results/final_model")

# Set the decoder_start_token_id and pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Load datasets
train_dataset = CaptchaDataset(
    images_dir="train3processed",  # Adjust the path as needed
    labels_file="train2processedlabels.csv",
    transform=transform
)

eval_dataset = CaptchaDataset(
    images_dir="processed_images",  # Adjust the path as needed
    labels_file="labels.csv",
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

# Define the collate function
def collate_fn(batch):
    pixel_values = [processor(image, return_tensors="pt").pixel_values.squeeze() for image, _ in batch]
    labels = [processor.tokenizer(label, padding="max_length", max_length=32, return_tensors="pt").input_ids.squeeze() for _, label in batch]
    pixel_values = torch.stack(pixel_values)
    labels = torch.stack(labels)
    return {"pixel_values": pixel_values, "labels": labels}

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./results/logs",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,  # Limit the total number of checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    tokenizer=processor.feature_extractor,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("./results3/final_model")
