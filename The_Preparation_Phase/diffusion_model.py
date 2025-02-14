from diffusers import StableDiffusionPipeline
import torch
import os
import random
from The_Preparation_Phase.prompts import prompt_airplane,prompt_automobile,prompt_horse,prompt_bird,prompt_cat,prompt_dog,prompt_deer,prompt_frog,prompt_ship,prompt_truck

choice = 0  ## 0 presents 10 categories of data, 1 presents 100 categories of data
save_dir = '' ## the path to store the generated images

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir='your cache_dir', local_files_only=True)
pipe = pipe.to('cuda:0')

cifar10_prompt = [prompt_airplane ,prompt_automobile,prompt_horse,prompt_bird,prompt_cat,prompt_dog,prompt_deer,prompt_frog,prompt_ship,prompt_truck]
cifar100_prompt = [] 

if not os.path.isdir(save_dir):
    os.makedirs(save_dir,exist_ok=True)

if choice==0:
    for i in range(10):
        path1 = os.path.join(save_dir,str(i))
        if not os.path.isdir(path1):
            os.makedirs(path1,exist_ok=True)
        photo_path = os.path.join(path1,str(i))
        if not os.path.isdir(photo_path):
            os.makedirs(photo_path,exist_ok=True)

        for num in range(2000):
            my_prompt = random.choice(cifar10_prompt[i][num])
            image = pipe([my_prompt]).images[0]
            image=image.resize([32,32])
            image.save(photo_path+'/'+str(num)+".png")
elif choice == 1:
    for i in range(100):
        path1 = os.path.join(save_dir,str(i))
        if not os.path.isdir(path1):
            os.makedirs(path1,exist_ok=True)
        photo_path = os.path.join(path1,str(i))
        if not os.path.isdir(photo_path):
            os.makedirs(photo_path,exist_ok=True)

        for num in range(2000):
            my_prompt = random.choice(cifar100_prompt[i][num])
            image = pipe([my_prompt]).images[0]
            image=image.resize([32,32])
            image.save(photo_path+'/'+str(num)+".png")