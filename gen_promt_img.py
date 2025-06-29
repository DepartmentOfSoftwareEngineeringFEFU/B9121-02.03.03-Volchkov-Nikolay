# Установка зависимостей (выполнить в терминале, если не установлено):
# pip install torch torchvision transformers diffusers accelerate

import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
# Проверка устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# загрузка нашей модели архитектуры Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipe.safety_checker = None
pipe = pipe.to(device)

prompt = ""
image = pipe(
    prompt, 
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
image.save("generated_image.png")
print("Изображение успешно сгенерировано и сохранено как 'generated_image.png'!")