import numpy as np
import torch
from PIL import Image
import os
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance



def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return image

def calc_fid(real_img_path, generated_imgs_dir):
    real_img = Image.open(real_img_path).convert("RGB")
    real_img = preprocess_image(np.array(real_img))
    # print(f"real_img.shape: {real_img.shape}")
    generated_imgs = []
    for file in os.listdir(generated_imgs_dir):
        if file.endswith(".png"):
            generated_imgs.append(Image.open(os.path.join(generated_imgs_dir, file)).convert("RGB"))
    generated_imgs = torch.cat([preprocess_image(np.array(image)) for image in generated_imgs])
    # print(f"generated_imgs.shape: {generated_imgs.shape}")
    real_imgs = real_img.repeat(generated_imgs.shape[0],1,1,1)
    # print(f"real_imgs.shape: {real_imgs.shape}")
    print("real image path: ", real_img_path)
    print("generated images dir: ", generated_imgs_dir)
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_imgs, real=True)
    fid.update(generated_imgs, real=False)
    fid_score=round(float(fid.compute()),4)
    # print(f"fid_score: {fid_score}")
    return fid_score
