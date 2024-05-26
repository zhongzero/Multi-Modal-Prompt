import argparse
import logging
import math
import os
import itertools
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from logging import getLogger
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler


logger = getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory to store the fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--instance_data_path",
        type=str,
        default=None,
        required=True,
        help="the path of the instance image.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "class images for prior preservation loss."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=5, help="Batch size for sampling images."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


    args = parser.parse_args()
    

    return args


class FinetuneDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        instance_data_path,
        instance_prompt,
        tokenizer,
        with_prior_preservation,
        class_data_root=None,
        class_prompt=None,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer
        
        self.instance_image_path = Path(instance_data_path)
        if not self.instance_image_path.exists():
            raise ValueError(f"Instance image path {self.instance_image_path} doesn't exists.")
        
        self.num_instance_images = 1
        self.instance_prompt = instance_prompt
        self._length = 1
        self.with_prior_preservation = with_prior_preservation

        if with_prior_preservation:
            assert class_data_root is not None, "class_data_root must be provided when with_prior_preservation is True"
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            assert self.num_class_images > 0, f"Class data root {self.class_data_root} is empty."
            self._length = self.num_class_images
            self.class_prompt = class_prompt

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_image_path)
        instance_image = exif_transpose(instance_image) # 如果图片有旋转信息，进行旋转

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images_pixel_values"] = self.image_transforms(instance_image)

        text_inputs = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        example["instance_input_ids"] = text_inputs.input_ids

        if self.with_prior_preservation:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images_pixel_values"] = self.image_transforms(class_image)

            class_text_inputs = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            example["class_input_ids"] = class_text_inputs.input_ids

        return example


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    
    if args.with_prior_preservation:
        assert args.class_data_dir is not None, "class_data_dir must be provided when with_prior_preservation is True"
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images == args.num_class_images:
            logger.info(f"Class images already exist in {args.class_data_dir}.")
        else:
            if os.path.exists(args.class_data_dir):
                os.system(f"rm -r {args.class_data_dir}/*")
            logger.info(f"Number of class images to sample: {args.num_class_images}.")
            pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None, dtype=torch.float16).to("cuda")
            bar = tqdm(range(math.ceil(args.num_class_images/args.sample_batch_size)))
            for i in bar:
                generate_num = min(args.sample_batch_size, args.num_class_images-i*args.sample_batch_size)
                bar.set_description(f"Generate {i*args.sample_batch_size+1}-th to {i*args.sample_batch_size+generate_num}-th class images")
                images = pipe(args.class_prompt, num_images_per_prompt=generate_num).images
                for j, image in enumerate(images):
                    image.save(f"{args.class_data_dir}/{i*args.sample_batch_size+j}.jpg")
                
        


    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None).to("cuda")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None).to("cuda")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None).to("cuda")
    
    # Freeze vae; set unet and text_encoder to trainable
    vae.requires_grad_(False)
    text_encoder.train()
    unet.train()
    
    train_dataset = FinetuneDataset(
        instance_data_path=args.instance_data_path,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        class_data_root=args.class_data_dir,
        class_prompt=args.class_prompt,
        size=args.resolution,
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["instance_images_pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["instance_input_ids"] for example in examples])
        if examples[0].get("class_images_pixel_values") is not None:
            class_pixel_values = torch.stack([example["class_images_pixel_values"] for example in examples])
            class_pixel_values = class_pixel_values.to(memory_format=torch.contiguous_format).float()
            class_input_ids = torch.stack([example["class_input_ids"] for example in examples])
            # pixel_values为(1, 3, 512, 512), class_pixel_values为(1, 3, 512, 512), 把它们拼接成(2, 3, 512, 512)
            return {"pixel_values": torch.cat([pixel_values, class_pixel_values], dim=0), "input_ids": torch.cat([input_ids, class_input_ids], dim=0)}
        else:
            return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0,
    )
    
    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        itertools.chain(unet.parameters(), text_encoder.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )
    
    if args.lr_scheduler!="constant":
        raise RuntimeError("Wrong. We only support \" args.lr_scheduler==\"constant\" \" currently")
    if noise_scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("Wrong. We only support \" noise_scheduler.config.prediction_type=\"epsilon\" \" currently")
    
    def compute_loss(batch):
        # Convert images to latent space
        latents = vae.encode(batch["pixel_values"].to(torch.float32).to("cuda")).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"].to("cuda"))[0]
        # encoder_hidden_states = batch["input_embedding"]

        # Get the target for loss depending on the prediction type
        target = noise

        # Predict the noise residual and compute loss
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            loss = compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(unet.parameters(), text_encoder.parameters()), args.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            global_step += 1
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
                break

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        revision=None,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
