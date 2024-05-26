from .metric_utils.dino_vit import VITs16
from .metric_utils.clip_vit import CLIP
from .metric_utils.evaluate_dino import *
import torch
import os
from PIL import Image

def calc_dino_clipi_score(real_img_path, generated_imgs_dir):
    real_img = Image.open(real_img_path).convert("RGB")
    generated_imgs = []
    for file in os.listdir(generated_imgs_dir):
        if file.endswith(".png"):
            generated_imgs.append(Image.open(os.path.join(generated_imgs_dir, file)).convert("RGB"))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dino_model = VITs16(device)
    clip_model = CLIP(device)


    # calculate dino score/clip-i score
    dino_score_list = evaluate_dino_score_list(real_img, generated_imgs, device, dino_model)
    clipi_score_list = evaluate_clipi_score_list(real_img, generated_imgs, device, clip_model)

    # print("dino score list: ", dino_score_list)
    # print("clip score list: ", clipi_score_list)
    
    dino_score = round(float(np.average(dino_score_list)), 4)
    clipi_score = round(float(np.average(clipi_score_list)), 4)
    
    # print("dino score:",dino_score)
    # print("clip-i score:",clipi_score)
    # print("dino score's standard deviation: ",np.std(dino_score_list))
    # print("clip-i score's standard deviation: ",np.std(clipi_score_list))
    
    return dino_score, clipi_score
