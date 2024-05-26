from transformers import AutoProcessor, BlipProcessor, AutoModelForVisualQuestionAnswering, BlipForConditionalGeneration
import torch
from PIL import Image


def blip_img2txt(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large").to(device)
    
    image = Image.open(img_path)

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    return generated_caption
