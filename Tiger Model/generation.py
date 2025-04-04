# coding=utf-8
# Copyright 2024 Hui Lu, Fang Dai, Siqiong Yao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes modifications based on the original implementation from:
# HuggingFace Inc. (2023), Diffusers Library, available at:
# https://github.com/huggingface/diffusers
# The original code is licensed under the Apache License, Version 2.0.




from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageFont
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers import DiffusionPipeline
import torch
import os
import random
from torchvision import transforms
import cv2 as cv

######################################################################## Foreback Generation #########################################################################
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image1 = torch.from_numpy(image)
    return image1
    
fig_name = "Image.png" 


init_image = Image.open("../Figure/paper/image/%s" % fig_name)
init_image = init_image.resize((512, 512))

mask_image_nd = Image.open("../Figure/paper/mask_nd/%s" % fig_name)
mask_image_nd = mask_image_nd.resize((512, 512))
control_image_nd = make_inpaint_condition(init_image, mask_image_nd)
#### background
mask_image_bg = Image.open("../Figure/paper/mask_bg/%s" % fig_name)
mask_image_bg = mask_image_bg.resize((512, 512))

con_name =  "Background_Source.png"
control_image_bg = Image.open('../dataset/Allclass/condition_bg/%s' %  con_name)
control_image_bg = control_image_bg.resize((512, 512))

# controlnet_nd = ControlNetModel.from_pretrained(
#     "../modelsaved/finetrainmodel/checkpoint-3000/controlnet", torch_dtype=torch.float16)

# pipe_nd = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "../model/pretrainmodel", controlnet=controlnet_nd, torch_dtype=torch.float16)
# pipe_nd.scheduler = DDIMScheduler.from_config(pipe_nd.scheduler.config)
# pipe_nd.enable_model_cpu_offload()

# prompt =[  ["papillary","follicular","medullary"],
#                     ["malignant","benign"],
#                     ["solid","cystic","spongiform"],
#                     ["wider-than-tall","taller-than-wide","circular"],
#                     ["clear","unclear"],
#                     ["irregular","regular"],
#                     ["uneven echo", "even echo"],
#                     ["low echo", "strong echo"],
#                     ["white points", "no points"],
#                     ["enormous nodes", "middle nodes","mini nodes"]]
# for i in range(100):
#     seed_nd =  random.randint(1,1000000)
#     generator = torch.Generator().manual_seed(seed_nd)
#     image_nd = pipe_nd(
#         # prompt = "papillary malignant solid taller white points uneven echo low echo middle nodes",
#         # prompt = "malignant follicular solid cystic uneven echo", 
#         prompt = "malignant medullary, solid, cystic, uneven echo white points enormous nodes", 
#         negative_prompt = "", 
#         num_inference_steps=50,
#         guidance_scale = 8, 
#         generator=generator,
#         eta=1.0,
#         controlnet_conditioning_scale = 0.2,
#         image=init_image,
#         mask_image=mask_image_nd,
#         control_image=control_image_nd,
#         ).images[0]
#     image_nd.save("../Figure/%s_%s.png" % (fig_name,seed_nd))

seed_bg = random.randint(1,1000000)
generator = torch.Generator().manual_seed(seed_bg)
controlnet_bg = ControlNetModel.from_pretrained(
    "../modelsaved/finetrainmodel/checkpoint-5000/controlnet", torch_dtype=torch.float16)

pipe_bg = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "../model/pretrainmodel", controlnet=controlnet_bg, torch_dtype=torch.float16
)
pipe_bg.scheduler = DDIMScheduler.from_config(pipe_bg.scheduler.config)
pipe_bg.enable_model_cpu_offload()

image_bg = pipe_bg(
    prompt = "black and white definition detail",
    negative_prompt = "", 
    num_inference_steps=20,
    guidance_scale = 0.02, 
    generator=generator,
    eta=1.0,
    controlnet_conditioning_scale = 1.0, 
    # control_guidance_start = 0.01, 
    # control_guidance_end = 0.7, 
    image=init_image,
    mask_image=mask_image_bg,
    control_image=control_image_bg,
    ).images[0]
image_bg.save("/export/home/daifang/Diffusion/own_code/Figure/PTC_T_bg2.png")
image_nd = np.array(image_nd)
image_bg = np.array(image_bg)
im2 = np.concatenate((image_nd, image_bg), axis=1)
im2 = Image.fromarray(im2)
im2.save("../Figure/generationfigure.png")


############################################################################################################################################################################

# image = np.array(image)
# init_image = np.array(init_image)
# mask_image = np.array(mask_image)
# control_image = np.array(control_image)
# control_image1 = np.zeros_like(init_image)
# control_image1[:,:,0] = control_image
# control_image1[:,:,1] = control_image
# control_image1[:,:,2] = control_image
# im2 = np.concatenate((init_image, mask_image, control_image1, image), axis=1) 
# im4 = Image.fromarray(im2)
# image_bg.save("../Figure/img_%s_%s.png" % (fig_name, seed))
# image_nd = np.array(image)
# init_image = np.array(init_image)
# mask_image = np.array(mask_image)
# im2 = np.concatenate((init_image, mask_image), axis=1) 
# im3 = np.concatenate((im2, image), axis=1) 
# im3 = Image.fromarray(im3)
# draw = ImageDraw.Draw(img3)
# draw.text((0,60),'你好',(0,0,0),font=font)
# image_nd.save("../Figure/noide_%s_%s2.png" % (fig_name, seed))

######################################################################## background #########################################################################
# def make_inpaint_condition(image, image_mask):
#     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
#     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

#     assert image.shape[0:1] == image_mask.shape[0:1]
#     image[image_mask > 0.5] = -1.0  # set as masked pixel
#     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
#     image1 = torch.from_numpy(image)
#     image = torch.from_numpy(image)[0]
#     toPIL = transforms.ToPILImage() 
#     pic = toPIL(image)
#     pic.save('random.jpg')
#     return image1

# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from diffusers.utils import load_image
# import torch

# base_model_path = "../model/pretrainmodel"
# controlnet_path = "../modelsaved/finetrainmodel/checkpoint-17000/controlnet"

# controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
# )

# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# # remove following line if xformers is not installed

# pipe.enable_model_cpu_offload()

# control_image = load_image("../dataset/controlnet_condition_nd/condition_nd/20191101_094933_7.png")
# prompt = "thyroid malignant solid clean irregular white points "

# # generate image
# generator = torch.manual_seed(0)
# image = pipe(prompt, num_inference_steps=50, generator=generator, image=control_image).images[0]

# image.save("../Figure/outputfigure.png")


# prompt ( or , optional) — The prompt or prompts to guide image generation. If not defined, you need to pass .strList[str]prompt_embeds
# image (, , , , , , — , or ): The initial image to be used as the starting point for the image generation process. Can also accept image latents as , and if passing latents directly they are not encoded again.torch.FloatTensorPIL.Image.Imagenp.ndarrayList[torch.FloatTensor]List[PIL.Image.Image]List[np.ndarray]List[List[torch.FloatTensor]]List[List[np.ndarray]]List[List[PIL.Image.Image]]image
# control_image (, , , , , , — , or ): The ControlNet input condition to provide guidance to the for generation. If the type is specified as , it is passed to ControlNet as is. can also be accepted as an image. The dimensions of the output image defaults to ’s dimensions. If height and/or width are passed, is resized accordingly. If multiple ControlNets are specified in , images must be passed as a list such that each element of the list can be correctly batched for input to a single ControlNet.torch.FloatTensorPIL.Image.Imagenp.ndarrayList[torch.FloatTensor]List[PIL.Image.Image]List[np.ndarray]List[List[torch.FloatTensor]]List[List[np.ndarray]]List[List[PIL.Image.Image]]unettorch.FloatTensorPIL.Image.Imageimageimageinit
# height (, optional, defaults to ) — The height in pixels of the generated image.intself.unet.config.sample_size * self.vae_scale_factor
# width (, optional, defaults to ) — The width in pixels of the generated image.intself.unet.config.sample_size * self.vae_scale_factor
# num_inference_steps (, optional, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.int
# guidance_scale (, optional, defaults to 7.5) — A higher guidance scale value encourages the model to generate images closely linked to the text at the expense of lower image quality. Guidance scale is enabled when .floatpromptguidance_scale > 1
# negative_prompt ( or , optional) — The prompt or prompts to guide what to not include in image generation. If not defined, you need to pass instead. Ignored when not using guidance ().strList[str]negative_prompt_embedsguidance_scale < 1
# num_images_per_prompt (, optional, defaults to 1) — The number of images to generate per prompt.int
# eta (, optional, defaults to 0.0) — Corresponds to parameter eta (η) from the DDIM paper. Only applies to the DDIMScheduler, and is ignored in other schedulers.float
# generator ( or , optional) — A torch.Generator to make generation deterministic.torch.GeneratorList[torch.Generator]
# latents (, optional) — Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random .torch.FloatTensorgenerator
# prompt_embeds (, optional) — Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the input argument.torch.FloatTensorprompt
# negative_prompt_embeds (, optional) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, are generated from the input argument.torch.FloatTensornegative_prompt_embedsnegative_prompt
# output_type (, optional, defaults to ) — The output format of the generated image. Choose between or .str"pil"PIL.Imagenp.array
# return_dict (, optional, defaults to ) — Whether or not to return a StableDiffusionPipelineOutput instead of a plain tuple.boolTrue
# callback (, optional) — A function that calls every steps during inference. The function is called with the following arguments: .Callablecallback_stepscallback(step: int, timestep: int, latents: torch.FloatTensor)
# callback_steps (, optional, defaults to 1) — The frequency at which the function is called. If not specified, the callback is called at every step.intcallback
# cross_attention_kwargs (, optional) — A kwargs dictionary that if specified is passed along to the as defined in self.processor.dictAttentionProcessor
# controlnet_conditioning_scale ( or , optional, defaults to 1.0) — The outputs of the ControlNet are multiplied by before they are added to the residual in the original . If multiple ControlNets are specified in , you can set the corresponding scale as a list.floatList[float]controlnet_conditioning_scaleunetinit
# guess_mode (, optional, defaults to ) — The ControlNet encoder tries to recognize the content of the input image even if you remove all prompts. A value between 3.0 and 5.0 is recommended.boolFalseguidance_scale
# control_guidance_start ( or , optional, defaults to 0.0) — The percentage of total steps at which the ControlNet starts applying.floatList[float]
# control_guidance_end ( or , optional, defaults to 1.0) — The percentage of total steps at which the ControlNet stops applying.floatList[float]
#     draw.text((5, 5),  prompt,  fill = (255, 255, 255))
#     image.save("/export/home/daifang/Diffusion/own_code/Figure/PTCtest/%s_%s_%s.png" % (i,seed,prompt.replace(" ","_").replace("(","").replace(")","").replace("[","").replace("]","")))
