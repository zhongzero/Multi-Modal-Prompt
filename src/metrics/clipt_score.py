from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import os
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image

def calc_clipt_score(prompt, generated_imgs_dir):
    generated_imgs = []
    for file in os.listdir(generated_imgs_dir):
        if file.endswith(".png"):
            generated_imgs.append(Image.open(os.path.join(generated_imgs_dir, file)).convert("RGB"))
    generated_imgs = torch.cat([preprocess_image(np.array(image)) for image in generated_imgs])
    
    prompts = [prompt] * generated_imgs.shape[0]

    clipt_score = round(float(clip_score(images=generated_imgs, text=prompts, model_name_or_path="openai/clip-vit-base-patch16")/100.0), 4)
    # print(f"clip-t score: {clipt_score}")
    return clipt_score