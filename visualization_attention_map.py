import argparse
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.io import ImageReadMode

from tqdm import tqdm, trange
from einops import rearrange, repeat
from omegaconf import OmegaConf
from functools import partial

# diffusers
import requests
from io import BytesIO
import diffusers
from diffusers import DDIMScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from triag.diffuser_utils import McaPipeline_Replace, LocalBlend
from triag.mca_utils import regiter_attention_editor_diffusers
from triag.mca_p2p import McaControlReplace

from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.ops import box_convert
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.transforms.functional import pil_to_tensor

import PIL
from PIL import Image, ImageDraw, ImageFont
import random
from torchvision import transforms

import supervision as sv

from huggingface_hub import hf_hub_download
import argparse

# visualization
from triag import vis_utils

# attn_reweight
from triag.ptp_utils import get_equalizer, expand_mask_tensor
from triag.ptp_utils import MVTecBankSimple, show_overlay_pil

# prompt optimization
from triag.prompt_optimize import StableDiffusion

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

model_path = '/data1/Shared/Models/stable-diffusion-v1-5'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
sd_model = McaPipeline_Replace.from_pretrained(model_path, scheduler=scheduler, safety_checker=None).to(device)
tokenizer = sd_model.tokenizer

def load_image_k(image_path, device):
    image = read_image(image_path, mode=ImageReadMode.RGB)
    # screw 这个读取出来是torch.Size([1, 3, 512, 512])
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

source_prompt = ""
ref_prompt = ""
cls_object = "hazelnut"
anomaly_type = "crack"
target_prompt = f"a photo of a {cls_object} with a {anomaly_type}"
# target_prompt = "a photo of a tile with a crack"
num_inference_steps = 50

# ref image
# hazelnut
REF_IMAGE_PATH = "data/anomaly/000.png"
ref_mask_path = "./data/anomaly/000_mask.png"
# mvtec
# REF_IMAGE_PATH = "/data1/Shared/Data/mvtec_anomaly_detection/wood/test/hole/000.png"
# ref_mask_path = "/data1/Shared/Data/mvtec_anomaly_detection/wood/ground_truth/hole/000_mask.png"
# visa
# REF_IMAGE_PATH = "./visa_seas/fryum/test/small_scratches/001.png"
# ref_mask_path = "./visa_seas/fryum/ground_truth/small_scratches/001.png"
ref_image = load_image_k(REF_IMAGE_PATH, device)  # torch.Size([1, 3, 512, 512])
print(ref_image.shape)

# invert the source image
ref_start_code, ref_latents_list = sd_model.invert(ref_image,
                                        ref_prompt,
                                        guidance_scale=7.5,
                                        num_inference_steps=num_inference_steps,
                                        return_intermediates=True)

pil_image = transforms.ToPILImage()(read_image(REF_IMAGE_PATH))
pil_image.resize((512,512))

# mvtec
SOURCE_IMAGE_PATH = "./data_agument/hazelnut/400.png"
source_mask_path = "./generated_mask/hazelnut/crack/1.jpg"
# visa
# SOURCE_IMAGE_PATH = "./data_agument_visa/fryum/000.png"
# source_mask_path = "./visa_seas/fryum/ground_truth/small_scratches/003.png"
# source_mask_path = "./visa_mask/cashew/same_colour_spot/mask/000.png"
source_image = load_image_k(SOURCE_IMAGE_PATH, device)

# results of direct synthesis
source_start_code, source_latents_list = sd_model.invert(source_image,
                                            source_prompt,
                                            guidance_scale=7.5,
                                            num_inference_steps=num_inference_steps,
                                            return_intermediates=True)

pil_image = transforms.ToPILImage()(read_image(SOURCE_IMAGE_PATH))
pil_image.resize((512,512))

# mask process
ref_mask = Image.open(ref_mask_path).convert("L")
source_mask = Image.open(source_mask_path).convert("L")
ref_mask = ref_mask.resize((512, 512), Image.NEAREST)
source_mask = source_mask.resize((512, 512), Image.NEAREST)
ref_mask = T.ToTensor()(ref_mask)
source_mask = T.ToTensor()(source_mask)
ref_mask = (ref_mask > 0.5).float().squeeze(0)   # → [512,512]
source_mask = (source_mask > 0.5).float().squeeze(0)
ref_mask = ref_mask.to(device)
source_mask = source_mask.to(device)


lbl = LocalBlend(source_mask.float())
self_replace_steps = 0.1
cross_replace_steps = 0.4, 0.8  # timestep for attention-reweight
SAQI_st = 0
START_LAYPER = 9
END_LAYPER = 16
START_STEP = 4
END_STEP = 50
# a photo of a hazelnut with a crack
prompts = [f'a photo of a {cls_object}', f'a photo of a {cls_object} with a {anomaly_type}', target_prompt]
ref_start_code = ref_start_code.expand(len(prompts), -1, -1, -1)
source_start_code = source_start_code.expand(len(prompts), -1, -1, -1)

negative_prompt = (
    # --- 通用画面降质词 ---
    # "over-exposure, under-exposure, saturated, duplicate, out of frame, "
    # "lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, "
    # "ugly, bad anatomy, bad proportions, "

    # --- 专门抑制“完好无损 / 光滑”语义 ---
    # MvTec
    "intact shell, smooth surface, pristine, flawless, clean, "
    "no crack, no cut, no scratch, no squeeze, perfect condition"

    # Visa
    # "intact, flawless, pristine, perfect condition, undamaged, unbroken, "
    # "complete, smooth, clean, spotless, single, symmetrical, aligned, uniform color, pristine surface"
)
# import inspect, sys
# print("Class module:", McaControlReplace.__module__)
# print("File:", inspect.getsourcefile(McaControlReplace))
# print("Init sig:", inspect.signature(McaControlReplace.__init__))

bank = MVTecBankSimple(bank_path="./embed_bank/mvtec")

diffusion_model = StableDiffusion(
    device=device,
    model_path=model_path,
    fp16=False, # TODO: use fp16
)

opt_embeddings = None
# optimize target prompt to algin with reference image
step_prompt_opt = 1
diffusion_model.get_text_embeds([target_prompt], [negative_prompt])
for p in diffusion_model.parameters():
    p.requires_grad_(False)
diffusion_model.embeddings['pos'].requires_grad_(True)

if bank.exists(cls_object, anomaly_type, REF_IMAGE_PATH):
    emb_cpu = bank.load(cls_object, anomaly_type, REF_IMAGE_PATH)
    diffusion_model.embeddings['pos'].data.copy_(
        emb_cpu.to(diffusion_model.embeddings['pos'].device,
                   dtype=diffusion_model.embeddings['pos'].dtype)
    )
    print("[Bank] HIT")
    opt_embeddings = diffusion_model.get_embedding()
else:
    print("[Bank] MISS → run optimization ...")
    optimizer = Adam([diffusion_model.embeddings['pos']], lr=3e-3)
    # load image
    input_image = load_image(REF_IMAGE_PATH).resize((512, 512))
    input_image = (pil_to_tensor(input_image).to(torch.float16) / 255.).to(device).unsqueeze(0)
    # Text Embedding Optimization
    pbar = trange(step_prompt_opt, desc='prompt optimization', leave=True)
    for _ in pbar:
        optimizer.zero_grad()
        loss = diffusion_model.train_step(
            input_image.clone().detach()
        )
        loss.backward(retain_graph=True)
        optimizer.step()
        pbar.set_description(f"Step A, loss: {loss.item():.3f}")            

    opt_embeddings = diffusion_model.get_embedding()
    bank.save(cls_object, anomaly_type, REF_IMAGE_PATH, diffusion_model.embeddings['pos'])

# test using embedding for generation
output = diffusion_model.prompt_to_img(opt_embeddings)
output_image = Image.fromarray(output[0])
output_image.save(f'./attention_map/anomaly_no_latent.png')
output = diffusion_model.prompt_to_img(opt_embeddings, latents=ref_start_code)
output_image = Image.fromarray(output[0])
output_image.save(f'./attention_map/anomaly_reference.png')
output = diffusion_model.prompt_to_img(opt_embeddings, latents=source_start_code)
output_image = Image.fromarray(output[0])
output_image.save(f'./attention_map/anomaly_source.png')
del diffusion_model

### pay 5 times more attention to the word "crack"
# equalizer = get_equalizer(prompts[2], (anomaly_type,), (100,), tokenizer).to(device)
# print("equalizer: ",equalizer)
# editor = McaControlReplace(prompts, tokenizer, [0,1,2,3], 
#                            self_replace_steps, cross_replace_steps, 
#                            equalizer, START_STEP, END_STEP, START_LAYPER, END_LAYPER, 
#                            mask_s=ref_mask.float(), mask_t=source_mask.float(), 
#                            attn_store_judge=True)
# regiter_attention_editor_diffusers(sd_model, editor)

# # mask_generated是为了看看latent层级上对比生成出来的mask
# # image_mcactrl, cross_image, mask_generated = sd_model(prompts,
# image_mcactrl, cross_image = sd_model(prompts,
#                     latents=source_start_code,
#                     num_inference_steps=num_inference_steps,
#                     guidance_scale=7.5,   # for MvTec/VisA OK
#                     ref_intermediate_latents=[ref_latents_list, source_latents_list],
#                     lbl = lbl,
#                     neg_prompt = negative_prompt,
#                     mask_r = ref_mask.float(),
#                     mask_t = source_mask.float(),
#                     return_intermediates=True,
#                     opt_embeddings=opt_embeddings
#                     )

# pil_image = transforms.ToPILImage()(image_mcactrl[-1:].squeeze())

# cross_show_time = [5,10,15,20,25,30,35,40,45,50]

# # 保存交叉注意力图
# select = 2  # 选择查看第几个prompt的交叉注意力
# attention_maps_list = editor.get_average_attention_list()
# # 这个是在原图上映射，不太准确，还是用prompt-to-prompt的黑白的算了
# # for i, timestep_cross in enumerate(cross_show_time):
# #     vis_utils.show_cross_attention_photo_for_list(attention_store=editor,
# #                                         prompt=prompts[select],
# #                                         tokenizer=tokenizer,
# #                                         res=32,
# #                                         select=select,
# #                                         indices_to_alter=[5,8],
# #                                         orig_image=transforms.ToPILImage()(cross_image[i][-1:].squeeze()),
# #                                         cross_show_time=cross_show_time)
# #     transforms.ToPILImage()(cross_image[i][-1:].squeeze()).save(f"./attention_map/cross_attention/anomaly_{cross_show_time[i]}.png")

# # 灰白的，越白表示该token对当前像素注意力越多。纯黑说明没注意
# for i, timestep_cross in enumerate(cross_show_time):
#     vis_utils.show_cross_attention_for_list(attention_store=editor,
#                                         prompt=prompts[select],
#                                         tokenizer=tokenizer,
#                                         res=32,
#                                         select=select,
#                                         indices_to_alter=[5,8],
#                                         cross_show_time=cross_show_time)
#     transforms.ToPILImage()(cross_image[i][-1:].squeeze()).save(f"./attention_map/cross_attention/anomaly_{cross_show_time[i]}.png")

# # 单个的50步注意力保存
# # attention_images=vis_utils.show_cross_attention(attention_store=editor,
# #                                 prompt=prompts[0],
# #                                 tokenizer=tokenizer,
# #                                 res=32,
# #                                 indices_to_alter=[5,8],
# #                                 orig_image=pil_image)

# # 可视化下self-attention map，仿效PNP
# # average_attention_self = editor.get_average_attention()[0]
# # for idx, attn in enumerate(average_attention_self):
# #     maps = vis_utils.self_attn_pca_rgb_pnp(attn, batch_size=len(prompts))  # -> [B] 个 (H,W,3)
# #     for b, rgb in enumerate(maps):
# #         Image.fromarray(rgb).save(f"./attention_map/self_attention_map/Attn_Layer{idx}_num{b}.png")
#     # 还是用上面的吧
#     # 只要target的attention-map
#     # Image.fromarray(maps[-1]).save(f"./attention_map/self_attention_map/Attn_Layer{idx}.png")

# # for place in ["down", "mid", "up"]:
# #     q_list = editor.q_store[place]  # list of [B,N,D]
# #     for layer_idx in range(len(q_list)):
# #         self_attention_images = vis_utils.q_to_rgb(q_list[layer_idx], save_path=f"./attention_map/self_attention_Q/", type_attention='Q', place=place, layer_idx=layer_idx)

# # for place in ["down", "mid", "up"]:
# #     v_list = editor.v_store[place]  # list of [B,N,D]
# #     for layer_idx in range(len(v_list)):
# #         self_attention_images = vis_utils.q_to_rgb(v_list[layer_idx], save_path=f"./attention_map/self_attention_V/", type_attention='V', place=place, layer_idx=layer_idx)

# print("pil_image: ",type(pil_image))
# pil_image.save("./attention_map/anomaly.png")
# # # image-mask overlay
# out = transforms.ToPILImage()(image_mcactrl[-1:].squeeze())
# out_1 = show_overlay_pil(out, mask_source, title = "Contrast")
# out_2 = show_overlay_pil(out, source_mask, title = "Contrast_expand")
# # out_3 = show_overlay_pil(out, mask_generated, title = "Contrast_latent_mask")
# # save_output_image(out, Path(task["save_path"]))
# out_1.savefig("./attention_map/anomaly_overlay.png", bbox_inches="tight")
# out_2.savefig("./attention_map/anomaly_overlay_expand.png", bbox_inches="tight")
# # out_3.savefig("./attention_map/anomaly_latent_mask.png", bbox_inches="tight")
