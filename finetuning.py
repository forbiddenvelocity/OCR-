import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.nn.utils.rnn import pad_sequence

class CAPTCHADataset(Dataset):
    def __init__(self, images_dir, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label = img_name.split('.')[0]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(label, return_tensors="pt").input_ids

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze()
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1').to(device)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-stage1')

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

train_dataset = CAPTCHADataset(images_dir='./captcha_images_v2', processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

def data_collator(features):
    pixel_values = torch.stack([feature["pixel_values"] for feature in features]).to(device)
    labels = [feature["labels"] for feature in features]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id).to(device)
    return {"pixel_values": pixel_values, "labels": labels_padded}

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="no",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=50,
    output_dir="./trocr-finetuned",
    logging_dir='./logs',
    logging_steps=50,
    fp16=True, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()

val_dataset = CAPTCHADataset(images_dir='./captcha_images_v2', processor=processor)
val_dataloader = DataLoader(val_dataset, batch_size=4)

model.eval()
for batch in val_dataloader:
    pixel_values = batch['pixel_values'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model.generate(pixel_values)
    predictions = processor.batch_decode(outputs, skip_special_tokens=True)
    references = processor.batch_decode(labels, skip_special_tokens=True)
    for prediction, reference in zip(predictions, references):
        print(f'Prediction: {prediction}, Reference: {reference}')
