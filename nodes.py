import os
import torch
import pyrallis
import numpy as np

from PIL import Image
from pathlib import Path
from .config import RunConfig, Range
from .appearance_transfer_model import AppearanceTransferModel
from .diffusers.training_utils import set_seed
from .utils.latent_utils import load_latents_or_invert_images, get_init_latents_and_noises

import folder_paths


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


class LoadImagePath:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGEPATH",)
    RETURN_NAMES = ("imagepath",)
    FUNCTION = "load_image_path"
    CATEGORY = "CrossImageAttention"

    def load_image_path(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        image_path = Path(image_path)
        return (image_path,)


class CIAConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "app_image_path": ("IMAGEPATH",),
                "struct_image_path": ("IMAGEPATH",),
                "domain_name": (
                    [
                        "animal",
                        "buildings"
                    ], {
                        "default": "animal"
                    }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 1000000}),
                "laod_latents": ("BOOLEAN", {"default": False}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A photo of a aniaml"}),
            }
        }

    RETURN_TYPES = ("CIA_CONFIG",)
    RETURN_NAMES = ("cia_config",)
    FUNCTION = "load_config"
    CATEGORY = "CrossImageAttention"

    def load_config(self, app_image_path, struct_image_path, domain_name, seed, laod_latents, prompt):
        config = RunConfig(
            app_image_path=app_image_path,
            struct_image_path=struct_image_path,
            domain_name=domain_name,
            seed=seed,
            load_latents=laod_latents,
            prompt=prompt
        )
        pyrallis.dump(config, open(config.output_path / 'config.yaml', 'w'))
        set_seed(config.seed)
        return (config,)


class AppearanceTransferModelModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cia_config": ("CIA_CONFIG",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "CrossImageAttention"

    def load_model(self, cia_config):
        model = AppearanceTransferModel(cia_config)
        return (model,)


class LoadLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cia_config": ("CIA_CONFIG",),
                "model": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL", "INIT_LATENTS", "INIT_ZS",)
    RETURN_NAMES = ("model", "init_latents", "init_zs",)
    FUNCTION = "load_latents"
    CATEGORY = "CrossImageAttention"

    def load_latents(self, cia_config, model):
        latens_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cia_config)
        model.set_latents(latens_app, latents_struct)
        model.set_noise(noise_app, noise_struct)
        init_latents, init_zs = get_init_latents_and_noises(model=model, cfg=cia_config)
        model.pipe.scheduler.set_timesteps(cia_config.num_timesteps)
        model.enable_edit = True
        return (model, init_latents, init_zs,)


class AppearanceTransferInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cia_config": ("CIA_CONFIG",),
                "model": ("MODEL",),
                "init_latents": ("INIT_LATENTS",),
                "init_zs": ("INIT_ZS",),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inference"
    CATEGORY = "CrossImageAttention"

    def inference(self, cia_config, model, init_latents, init_zs, guidance_scale):
        start_step = min(cia_config.cross_attn_32_range.start, cia_config.cross_attn_64_range.start)
        end_step = max(cia_config.cross_attn_32_range.end, cia_config.cross_attn_64_range.end)
        images = model.pipe(
            prompt=[cia_config.prompt] * 3,
            latents=init_latents,
            guidance_scale=guidance_scale,
            num_inference_steps=cia_config.num_timesteps,
            swap_guidance_scale=cia_config.swap_guidance_scale,
            callback=model.get_adain_callback(),
            eta=1,
            zs=init_zs,
            generator=torch.Generator('cuda').manual_seed(cia_config.seed),
            cross_image_attention_range=Range(start=start_step, end=end_step),
        ).images

        # import pdb;pdb.set_trace()
        # Save images
        images[0].save(cia_config.output_path / f"out_transfer---seed_{cia_config.seed}.png")
        images[1].save(cia_config.output_path / f"out_style---seed_{cia_config.seed}.png")
        images[2].save(cia_config.output_path / f"out_struct---seed_{cia_config.seed}.png")
        joined_images = np.concatenate(images[::-1], axis=1)
        Image.fromarray(joined_images).save(cia_config.output_path / f"out_joined---seed_{cia_config.seed}.png")
        output_images = convert_preview_image(images)

        return (output_images,)


NODE_CLASS_MAPPINGS = {
    "LoadImagePath": LoadImagePath,
    "CIAConfig": CIAConfig,
    "AppearanceTransferModelModelLoader": AppearanceTransferModelModelLoader,
    "LoadLatents": LoadLatents,
    "AppearanceTransferInference": AppearanceTransferInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagePath": "Load Image Path",
    "CIAConfig": "CIA Config",
    "AppearanceTransferModelModelLoader": "Appearance Transfer Model Loader",
    "LoadLatents": "Load Latents",
    "AppearanceTransferInference": "Appearance Transfer Inference",
}
