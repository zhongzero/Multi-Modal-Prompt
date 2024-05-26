import argparse
import os

from method import blip_direct, finetune_direct, dreambooth, textual_inversion, custom_diffusion, updated_dreambooth, updated_custom_diffusion
from metrics.dino_clipi_score import calc_dino_clipi_score
from metrics.clipt_score import calc_clipt_score
from method import get_eval_text_prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--multi_modal_prompt',
        type=str,
        default=None,
        required=True,
        help="Multi-modal prompt. e.g. \"A photo of dog on the bench\", \"A photo of [img] on the [img]\""
    )
    parser.add_argument(
        '--img_paths',
        type=str,
        default="[]",
        help="List of picture paths(in the order of appearance in the prompt) in the format of \"[img1_path, img2_path, ...]\". e.g. \"[./cat.jpg, ./dog.jpg]\""
    )
    parser.add_argument(
        "--method",
        type=str,
        default="updated_dreambooth",
        help="Method to generate image. choose from [updated_dreambooth, updated_custom_diffusion, dreambooth, custom_diffusion, textual_inversion, blip_direct, finetune_direct]",
    )
    parser.add_argument(
        '--classes',
        type=str,
        default=None,
        help="Provided only when method belongs to [\"dreambooth\", \"textual_inversion\", \"custom_diffusion\"]. List of pictures' class(in the order of appearance in the prompt) in the format of \"[img1_class, img2_class, ...]\". e.g. \"[cat, dog]\""
    )
    parser.add_argument(
        '--output_model_dir',
        type=str,
        default="pretrained_models",
        help="The output directory to store the fine-tuned model checkpoint and some intermediate outputs."
    )
    parser.add_argument(
        '--output_img_dir',
        type=str,
        default="output_imgs",
        help="The output directory to store the generated images."
    )
    parser.add_argument(
        '--output_result_path',
        type=str,
        default="result.txt",
        help="The output file to store evaluation results."
    )
    parser.add_argument(
        '--pretrained_base_model',
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="The pretrained base model used in our paradigm."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help="The number of images generated by pretrained_base_model for prior preservation loss.",
    )
    parser.add_argument(
        "--generate_imgs_num",
        type=int,
        default=20,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="regenerate images. Make sure you have run the method using the same argments before. "
              "Use this option to regenerate images with pretrained model and intermediate outputs.",
    )
    parser.add_argument(
        "--gpt_api_key",
        type=str,
        default="sk-RDzt6LOa3JctbuAAbtHkp7OxAaMjCUkkvZARlFBGg0pI5hhC",
        help="Api key of openai gpt3.5",
    )
    parser.add_argument(
        "--gpt_base_url",
        type=str,
        default="https://api.chatanywhere.com.cn/v1",
        help="Base url of openai gpt3.5",
    )
    
    
    args = parser.parse_args()

    assert args.img_paths[0] == "[" and args.img_paths[-1] == "]"

    print(f"args.multi_modal_prompt = {args.multi_modal_prompt}")
    print(f"args.img_paths = {args.img_paths}")
    return args


def main():
    args = parse_args()

    img_paths_list = []
    if len(args.img_paths) > 2:
        img_paths_list = args.img_paths[1:-1].split(",")
        img_paths_list = [path.strip() for path in img_paths_list]
    print(f"img_paths_list = {img_paths_list}")
    assert args.multi_modal_prompt.count("[img]") == len(img_paths_list)
    if args.method in ["dreambooth", "textual_inversion", "custom_diffusion"]:
        assert args.classes[0] == "[" and args.classes[-1] == "]"
        if len(args.classes) > 2:
            classes_list = args.classes[1:-1].split(",")
            classes_list = [class_.strip() for class_ in classes_list]
            assert args.multi_modal_prompt.count("[img]") == len(classes_list)
    print(f"args.method = {args.method}") 
    os.makedirs(args.output_model_dir, exist_ok = True) # if output_model_dir is not exist, create it
    os.makedirs(args.output_img_dir, exist_ok = True) # if output_img_dir is not exist, create it
    if args.method == "blip_direct":
        blip_direct(args, img_paths_list)
    elif args.method == "finetune_direct":
        finetune_direct(args, img_paths_list)
    elif args.method == "dreambooth":
        dreambooth(args, img_paths_list, classes_list)
    elif args.method == "textual_inversion":
        textual_inversion(args, img_paths_list, classes_list)
    elif args.method == "custom_diffusion":
        custom_diffusion(args, img_paths_list, classes_list)
    elif args.method == "updated_dreambooth":
        updated_dreambooth(args, img_paths_list)
    elif args.method == "updated_custom_diffusion":
        updated_custom_diffusion(args, img_paths_list)
    else:
        raise RuntimeError(f"method {args.method} not supported")
    
    with open(os.path.join(args.output_result_path), "a") as f:
        f.write("----------------------------------------------------------------------------------------------------\n")
        f.write(f"\"{args.multi_modal_prompt}\", [img]: \"{args.img_paths}\"\n")
        f.write(f"{args.method}\n")
        f.write("\n")
        # evaluate
        prompt = get_eval_text_prompt(args, img_paths_list)
        print(f"prompt: {prompt}")
        print(f"args.method: {args.output_img_dir}")
        print(f"args.output_img_dir: {args.output_img_dir}")
        
        for i in range(0,len(img_paths_list)):
            dino_score, clipi_score = calc_dino_clipi_score(img_paths_list[i], args.output_img_dir)
            print(f"dino score(with {i+1}-th img): {dino_score}",)
            print(f"clip-i score(with {i+1}-th img): {clipi_score}")
            f.write(f"dino score(with {i+1}-th img): {dino_score}\n")
            f.write(f"clip-i score(with {i+1}-th img): {clipi_score}\n")
        clipt_score = calc_clipt_score(prompt, args.output_img_dir)
        print(f"clip-t score: {clipt_score}")
        f.write(f"clip-t score: {clipt_score}\n")
        f.write("\n")

if __name__ == "__main__":
    main()