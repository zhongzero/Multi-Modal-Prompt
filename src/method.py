import os
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
from tqdm import tqdm
import math
import json

from blip import blip_img2txt
from gptExtract import gpt_txtExtract

special_token = ["sks", "olis", "sigue"]

def get_eval_text_prompt(args, img_paths_list):
    eval_text_prompt = args.multi_modal_prompt
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        raw_caption = blip_img2txt(img_path)
        foreground, background, action = gpt_txtExtract(raw_caption, args)
        assert foreground != "None"
        pos_img = eval_text_prompt.find("[img]")
        # use background as main object only when the image is behind "on the background of"
        if eval_text_prompt[:pos_img-1].strip().endswith("on the background of"):
            if background == "None":
                main_object = foreground
            else:
                # main_object = background
                main_object = f"{background} with {foreground}"
        else:
            main_object = foreground
        eval_text_prompt = eval_text_prompt.replace("[img]", main_object, 1)
    return eval_text_prompt

def get_pure_text_prompt(args, img_paths_list):
    # with open(f"{args.output_model_dir}/pure_text_prompt.txt", "r") as f:
    #     pure_text_prompt = f.read()
    with open(f"{args.output_model_dir}/replace_list.txt", "r") as f:
        replace_list = f.read().split("\n")
    pure_text_prompt = args.multi_modal_prompt
    for i, img_path in enumerate(img_paths_list):
        pure_text_prompt = pure_text_prompt.replace("[img]", replace_list[i], 1)
    return pure_text_prompt

def inference(model_id, prompt, args):
    print("-----------start inference-----------")
    print(f"model_id = {model_id}")
    print(f"prompt = {prompt}")
    if model_id == args.pretrained_base_model:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    else :
        unet = UNet2DConditionModel.from_pretrained(f"{model_id}/unet")
        text_encoder = CLIPTextModel.from_pretrained(f"{model_id}/text_encoder")
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16, use_safetensors=True).to("cuda")

    pipe.safety_checker = None  # remove safety check
    
    if args.method == "textual_inversion":
        pipe.load_textual_inversion(args.output_model_dir)
    if args.method == "custom_diffusion" or args.method == "updated_custom_diffusion":
        pipe.unet.load_attn_procs(args.output_model_dir, weight_name="pytorch_custom_diffusion_weights.bin")
        for i in range(args.multi_modal_prompt.count("[img]")):
            pipe.load_textual_inversion(args.output_model_dir, weight_name=f"<{special_token[i]}>.bin")
    
    sample_batch_size = 5
    bar = tqdm(range(math.ceil(args.generate_imgs_num/sample_batch_size)))
    for i in bar:
        generate_num = min(sample_batch_size, args.generate_imgs_num-i*sample_batch_size)
        bar.set_description(f"Generate {i*sample_batch_size+1}-th to {i*sample_batch_size+generate_num}-th class images")
        images = pipe(prompt, num_inference_steps=200, guidance_scale=7.5, num_images_per_prompt=generate_num).images
        for j, image in enumerate(images):
            image.save(f"{args.output_img_dir}/output{i*sample_batch_size+j}.png")


def updated_dreambooth(args, img_paths_list):
    if args.regenerate:
        print("args.regenerate = True")
        final_model_id = args.pretrained_base_model if len(img_paths_list) == 0 else f"{args.output_model_dir}/finetune_model-v{len(img_paths_list)}"
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(final_model_id, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    finetune_list = []
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        raw_caption = blip_img2txt(img_path)
        foreground, background, action = gpt_txtExtract(raw_caption, args)
        assert foreground != "None"
        pos_img = pure_text_prompt.find("[img]")
        # use background as main object only when the image is behind "on the background of"
        if pure_text_prompt[:pos_img-1].strip().endswith("on the background of"):
            if background == "None":
                main_object = foreground
            else:
                # main_object = background
                main_object = f"{background} with {foreground}"
        else:
            main_object = foreground
        pure_text_prompt = pure_text_prompt.replace("[img]", special_token[i]+" "+main_object, 1)
        replace_list.append(special_token[i]+" "+main_object)
        finetune_list.append([main_object, special_token[i]+" "+main_object, img_path])
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))
    
    for i, finetune in enumerate(finetune_list):
        if os.path.exists(f"{args.output_model_dir}/generate_class_object{i}"):
            os.system(f"rm -r {args.output_model_dir}/generate_class_object{i}")
        print(f"finetune {i}-th object")
        class_prompt = finetune[0]
        instance_prompt = finetune[1]
        img_path = finetune[2]
        print(f"object class_prompt = {class_prompt}")
        print(f"object instance_prompt = {instance_prompt}")
        print(f"object img_path = {img_path}")
        pretrained_model_name_or_path = args.pretrained_base_model if i == 0 else f"{args.output_model_dir}/finetune_model-v{i}"
        class_data_dir = f"{args.output_model_dir}/generate_class_object{i}"
        output_dir = f"{args.output_model_dir}/finetune_model-v{i+1}"
        
        os.makedirs(f"{args.output_model_dir}/given_object", exist_ok=True)
        if len(os.listdir(f"{args.output_model_dir}/given_object")) != 0:
            os.system(f"rm -r {args.output_model_dir}/given_object/*")
        os.system(f"cp {img_path} {args.output_model_dir}/given_object")
        instance_data_dir=f"{args.output_model_dir}/given_object"
        print(f"instance_data_dir = {instance_data_dir}")
        os.system(f"accelerate launch src/train_dreambooth.py \
                    --pretrained_model_name_or_path=\"{pretrained_model_name_or_path}\"  \
                    --train_text_encoder \
                    --instance_data_dir=\"{instance_data_dir}\" \
                    --class_data_dir=\"{class_data_dir}\" \
                    --output_dir=\"{output_dir}\" \
                    --with_prior_preservation --prior_loss_weight=1.0 \
                    --instance_prompt=\"{instance_prompt}\" \
                    --class_prompt=\"{class_prompt}\" \
                    --resolution=512 \
                    --train_batch_size=1 \
                    --use_8bit_adam \
                    --gradient_checkpointing \
                    --learning_rate=2e-6 \
                    --lr_scheduler=constant \
                    --lr_warmup_steps=0 \
                    --num_class_images=100 \
                    --max_train_steps=200")
        print(f"finetuning {i}-th object is done")

    final_model_id = args.pretrained_base_model if len(finetune_list) == 0 else f"{args.output_model_dir}/finetune_model-v{len(finetune_list)}"
    inference(final_model_id, pure_text_prompt, args)

def updated_custom_diffusion(args, img_paths_list):
    if args.regenerate:
        print("args.regenerate = True")
        final_model_id = args.pretrained_base_model
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(final_model_id, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    concept_list = []
    modifier_token = ""
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        raw_caption = blip_img2txt(img_path)
        foreground, background, action = gpt_txtExtract(raw_caption, args)
        assert foreground != "None"
        pos_img = pure_text_prompt.find("[img]")
        # use background as main object only when the image is behind "on the background of"
        if pure_text_prompt[:pos_img-1].strip().endswith("on the background of"):
            if background == "None":
                main_object = foreground
            else:
                # main_object = background
                main_object = f"{background} with {foreground}"
        else:
            main_object = foreground
        pure_text_prompt = pure_text_prompt.replace("[img]", f"<{special_token[i]}> {main_object}", 1)
        replace_list.append(f"<{special_token[i]}> {main_object}")
        
        os.makedirs(f"{args.output_model_dir}/given_object{i}", exist_ok=True)
        if len(os.listdir(f"{args.output_model_dir}/given_object{i}")) != 0:
            os.system(f"rm -r {args.output_model_dir}/given_object{i}/*")
        os.system(f"cp {img_path} {args.output_model_dir}/given_object{i}")
        
        if os.path.exists(f"{args.output_model_dir}/download_class_object{i}"):
            os.system(f"rm -r {args.output_model_dir}/download_class_object{i}")
        
        current_concept = {
            "instance_prompt": f"A photo of <{special_token[i]}> {main_object}",
            "class_prompt": main_object,
            "instance_data_dir": f"{args.output_model_dir}/given_object{i}",
            "class_data_dir": f"{args.output_model_dir}/download_class_object{i}"
        }
        concept_list.append(current_concept)
        modifier_token = f"{modifier_token}+<{special_token[i]}>" if i != 0 else f"<{special_token[i]}>"
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))
    
    concepts_list_dir=f"{args.output_model_dir}/concept_list.json"
    with open(concepts_list_dir, "w") as f:
        json.dump(concept_list, f, indent=4)
    
    for i, concept in enumerate(concept_list):
        # os.system(f"python src/retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir \"{concept['class_data_dir']}\" --num_class_images 200")
        os.system(f"python src/retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir \"{concept['class_data_dir']}\" --num_class_images {args.num_class_images} --use_generate_data --pretrained_model \"{args.pretrained_base_model}\"")
    
    os.system(f"accelerate launch src/train_custom_diffusion.py \
                --pretrained_model_name_or_path=\"{args.pretrained_base_model}\"  \
                --output_dir=\"{args.output_model_dir}\" \
                --concepts_list=\"{concepts_list_dir}\" \
                --with_prior_preservation \
                --real_prior \
                --prior_loss_weight=1.0 \
                --resolution=512  \
                --train_batch_size=2  \
                --learning_rate=1e-5  \
                --lr_warmup_steps=0 \
                --max_train_steps=500 \
                --num_class_images={args.num_class_images} \
                --scale_lr \
                --hflip  \
                --modifier_token \"{modifier_token}\" \
                --no_safe_serialization \
                --checkpointing_steps=2000")
        
    print(f"finetuning all objects is done")
    
    final_model_id = args.pretrained_base_model
    inference(final_model_id, pure_text_prompt, args)

def blip_direct(args, img_paths_list):
    if args.regenerate:
        print("args.regenerate = True")
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(args.pretrained_base_model, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        raw_caption = blip_img2txt(img_path)
        foreground, background, action = gpt_txtExtract(raw_caption, args)
        assert foreground != "None"
        pos_img = pure_text_prompt.find("[img]")
        # use background as main object only when the image is behind "on the background of"
        if pure_text_prompt[:pos_img-1].strip().endswith("on the background of"):
            if background == "None":
                main_object = foreground
            else:
                # main_object = background
                main_object = f"{background} with {foreground}"
        else:
            main_object = foreground
        pure_text_prompt = pure_text_prompt.replace("[img]", main_object, 1)
        replace_list.append(main_object)
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))
    inference(args.pretrained_base_model, pure_text_prompt, args)

def finetune_direct(args, img_paths_list):
    if args.regenerate:
        print("args.regenerate = True")
        final_model_id = args.pretrained_base_model if len(img_paths_list) == 0 else f"{args.output_model_dir}/finetune_model-v{len(img_paths_list)}"
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(final_model_id, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    finetune_list = []
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        pure_text_prompt = pure_text_prompt.replace("[img]", special_token[i], 1)
        replace_list.append(special_token[i])

        finetune_list.append([special_token[i], img_path])
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))

    for i, finetune in enumerate(finetune_list):
        print(f"finetune {i}-th object")
        instance_prompt = finetune[0]
        img_path = finetune[1]
        print(f"object instance_prompt = {instance_prompt}")
        print(f"object img_path = {img_path}")
        pretrained_model_name_or_path = args.pretrained_base_model if i == 0 else f"{args.output_model_dir}/finetune_model-v{i}"
        output_dir = f"{args.output_model_dir}/finetune_model-v{i+1}"
        os.system(f"python src/finetune.py \
                    --pretrained_model_name_or_path=\"{pretrained_model_name_or_path}\"  \
                    --output_dir=\"{output_dir}\" \
                    --instance_data_path=\"{img_path}\" \
                    --instance_prompt=\"{instance_prompt}\" \
                    --learning_rate=1e-6 \
                    --max_train_steps=400 \
                    --resolution=512 \
                    --train_batch_size=1 \
                    --gradient_accumulation_steps=4 \
                    --lr_scheduler=constant \
                    --lr_warmup_steps=0")
        print(f"finetuning {i}-th object is done")

    final_model_id = args.pretrained_base_model if len(finetune_list) == 0 else f"{args.output_model_dir}/finetune_model-v{len(finetune_list)}"
    inference(final_model_id, pure_text_prompt, args)

def dreambooth(args, img_paths_list, classes_list):
    if args.regenerate:
        print("args.regenerate = True")
        final_model_id = args.pretrained_base_model if len(img_paths_list) == 0 else f"{args.output_model_dir}/finetune_model-v{len(img_paths_list)}"
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(final_model_id, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    finetune_list = []
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        main_object = classes_list[i]
        pure_text_prompt = pure_text_prompt.replace("[img]", special_token[i]+" "+main_object, 1)
        replace_list.append(special_token[i]+" "+main_object)

        finetune_list.append([main_object, special_token[i]+" "+main_object, img_path])
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))

    for i, finetune in enumerate(finetune_list):
        if os.path.exists(f"{args.output_model_dir}/generate_class_object{i}"):
            os.system(f"rm -r {args.output_model_dir}/generate_class_object{i}")
        print(f"finetune {i}-th object")
        class_prompt = finetune[0]
        instance_prompt = finetune[1]
        img_path = finetune[2]
        print(f"object class_prompt = {class_prompt}")
        print(f"object instance_prompt = {instance_prompt}")
        print(f"object img_path = {img_path}")
        
        pretrained_model_name_or_path = args.pretrained_base_model if i == 0 else f"{args.output_model_dir}/finetune_model-v{i}"
        class_data_dir = f"{args.output_model_dir}/generate_class_object{i}"
        output_dir = f"{args.output_model_dir}/finetune_model-v{i+1}"
        
        os.makedirs(f"{args.output_model_dir}/given_object", exist_ok=True)
        if len(os.listdir(f"{args.output_model_dir}/given_object")) != 0:
            os.system(f"rm -r {args.output_model_dir}/given_object/*")
        os.system(f"cp {img_path} {args.output_model_dir}/given_object")
        instance_data_dir=f"{args.output_model_dir}/given_object"
        print(f"instance_data_dir = {instance_data_dir}")
        os.system(f"accelerate launch src/train_dreambooth.py \
                    --pretrained_model_name_or_path=\"{pretrained_model_name_or_path}\"  \
                    --train_text_encoder \
                    --instance_data_dir=\"{instance_data_dir}\" \
                    --class_data_dir=\"{class_data_dir}\" \
                    --output_dir=\"{output_dir}\" \
                    --with_prior_preservation --prior_loss_weight=1.0 \
                    --instance_prompt=\"{instance_prompt}\" \
                    --class_prompt=\"{class_prompt}\" \
                    --resolution=512 \
                    --train_batch_size=1 \
                    --use_8bit_adam \
                    --gradient_checkpointing \
                    --learning_rate=2e-6 \
                    --lr_scheduler=constant \
                    --lr_warmup_steps=0 \
                    --num_class_images=100 \
                    --max_train_steps=200")
        print(f"finetuning {i}-th object is done")

    final_model_id = args.pretrained_base_model if len(finetune_list) == 0 else f"{args.output_model_dir}/finetune_model-v{len(finetune_list)}"
    inference(final_model_id, pure_text_prompt, args)
    
def textual_inversion(args, img_paths_list, classes_list):
    if args.regenerate:
        print("args.regenerate = True")
        final_model_id = args.pretrained_base_model
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(final_model_id, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    finetune_list = []
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        main_object = classes_list[i]
        pure_text_prompt = pure_text_prompt.replace("[img]", f"<{special_token[i]}>", 1)
        replace_list.append(f"<{special_token[i]}>")

        finetune_list.append([main_object, img_path])
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))

    for i, finetune in enumerate(finetune_list):
        print(f"finetune {i}-th object")
        class_prompt = finetune[0]
        img_path = finetune[1]
        print(f"object class_prompt = {class_prompt}")
        print(f"object img_path = {img_path}")
        
        pretrained_model_name_or_path = args.pretrained_base_model
        output_dir = args.output_model_dir
        
        os.makedirs(f"{args.output_model_dir}/given_object", exist_ok=True)
        if len(os.listdir(f"{args.output_model_dir}/given_object")) != 0:
            os.system(f"rm -r {args.output_model_dir}/given_object/*")
        os.system(f"cp {img_path} {args.output_model_dir}/given_object")
        instance_data_dir=f"{args.output_model_dir}/given_object"
        print(f"instance_data_dir = {instance_data_dir}")
        os.system(f"accelerate launch src/textual_inversion.py \
                    --pretrained_model_name_or_path=\"{pretrained_model_name_or_path}\" \
                    --train_data_dir=\"{instance_data_dir}\" \
                    --learnable_property=\"object\" \
                    --placeholder_token=\"<{special_token[i]}>\" \
                    --initializer_token=\"{class_prompt}\" \
                    --resolution=512 \
                    --train_batch_size=1 \
                    --gradient_accumulation_steps=4 \
                    --max_train_steps=400 \
                    --learning_rate=5e-04 \
                    --scale_lr \
                    --lr_scheduler=\"constant\" \
                    --lr_warmup_steps=0 \
                    --checkpointing_steps=3000 \
                    --output_dir=\"{output_dir}\"")
        print(f"finetuning {i}-th object is done")

    final_model_id = args.pretrained_base_model
    inference(final_model_id, pure_text_prompt, args)
    
def custom_diffusion(args, img_paths_list, classes_list):
    if args.regenerate:
        print("args.regenerate = True")
        final_model_id = args.pretrained_base_model
        pure_text_prompt = get_pure_text_prompt(args, img_paths_list)
        inference(final_model_id, pure_text_prompt, args)
        return
    pure_text_prompt = args.multi_modal_prompt
    concept_list = []
    modifier_token = ""
    replace_list = []
    for i, img_path in enumerate(img_paths_list):
        print(f"process_img_path = {img_path}")
        pure_text_prompt = pure_text_prompt.replace("[img]", f"<{special_token[i]}> {classes_list[i]}", 1)
        replace_list.append(f"<{special_token[i]}> {classes_list[i]}")
        
        os.makedirs(f"{args.output_model_dir}/given_object{i}", exist_ok=True)
        if len(os.listdir(f"{args.output_model_dir}/given_object{i}")) != 0:
            os.system(f"rm -r {args.output_model_dir}/given_object{i}/*")
        os.system(f"cp {img_path} {args.output_model_dir}/given_object{i}")
        
        if os.path.exists(f"{args.output_model_dir}/download_class_object{i}"):
            os.system(f"rm -r {args.output_model_dir}/download_class_object{i}")
        
        current_concept = {
            "instance_prompt": f"A photo of a <{special_token[i]}> {classes_list[i]}",
            "class_prompt": classes_list[i],
            "instance_data_dir": f"{args.output_model_dir}/given_object{i}",
            "class_data_dir": f"{args.output_model_dir}/download_class_object{i}"
        }
        concept_list.append(current_concept)
        modifier_token = f"{modifier_token}+<{special_token[i]}>" if i != 0 else f"<{special_token[i]}>"
    print(f"pure_text_prompt = {pure_text_prompt}")
    with open(f"{args.output_model_dir}/pure_text_prompt.txt", "w") as f:
        f.write(pure_text_prompt)
    with open(f"{args.output_model_dir}/replace_list.txt", "w") as f:
        f.write("\n".join(replace_list))
    
    concepts_list_dir=f"{args.output_model_dir}/concept_list.json"
    with open(concepts_list_dir, "w") as f:
        json.dump(concept_list, f, indent=4)

    for i, concept in enumerate(concept_list):
        # os.system(f"python src/retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir \"{concept['class_data_dir']}\" --num_class_images 200")
        os.system(f"python src/retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir \"{concept['class_data_dir']}\" --num_class_images {args.num_class_images} --use_generate_data --pretrained_model \"{args.pretrained_base_model}\"")
    
    os.system(f"accelerate launch src/train_custom_diffusion.py \
                --pretrained_model_name_or_path=\"{args.pretrained_base_model}\"  \
                --output_dir=\"{args.output_model_dir}\" \
                --concepts_list=\"{concepts_list_dir}\" \
                --with_prior_preservation \
                --real_prior \
                --prior_loss_weight=1.0 \
                --resolution=512  \
                --train_batch_size=2  \
                --learning_rate=1e-5  \
                --lr_warmup_steps=0 \
                --max_train_steps=500 \
                --num_class_images={args.num_class_images} \
                --scale_lr \
                --hflip  \
                --modifier_token \"{modifier_token}\" \
                --no_safe_serialization \
                --checkpointing_steps=2000")
        
    print(f"finetuning all objects is done")
    
    final_model_id = args.pretrained_base_model
    inference(final_model_id, pure_text_prompt, args)