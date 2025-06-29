import logging
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline
import math
app = Flask(__name__)

# Настраиваем логирование
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
def round_to_nearest_multiple_of_8(x):
    return int(round(x / 8.0)) * 8

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

app.logger.info(f"Загружаем модель: {model_id} на устройство {device}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Отключаем safety_checker (или делаем его «пустым»)
# По умолчанию ожидает, что safety_checker вернёт (images, has_nsfw_concept),
# где has_nsfw_concept — список из bool для каждого изображения.
# Если вернуть bool, может быть ошибка "TypeError: 'bool' object is not iterable".
def dummy_safety_checker(images, clip_input):
    # Возвращаем список [False]*len(images), чтобы указать, что
    # якобы "никаких NSFW-объектов" не найдено.
    return images, [False]*len(images)

pipe.safety_checker = dummy_safety_checker

# Переносим сам пайплайн на устройство (CPU или GPU)
pipe = pipe.to(device)

app.logger.info("Модель загружена и готова к работе.")

@app.route("/generate", methods=["POST"])
def generate_image():
    """
    Ожидается JSON вида:
    {
      "prompt": "какой-то текст",
      "size": "500x600"
    }
    """
    app.logger.info("Начали обработку запроса /generate")

    # Получаем JSON из запроса
    data = request.json
    if not data:
        app.logger.warning("Не получили JSON в запросе!")
        return jsonify({"error": "No JSON data received"}), 400

   
    prompt = data.get("prompt", "")
    size_str = data.get("size", "")

    app.logger.debug(f"Получен prompt: {prompt}")
    app.logger.debug(f"Получен size: {size_str}")

    # Парсим размер
    width, height = 512, 512
    if "x" in size_str:
        try:
            w_str, h_str = size_str.split("x")
            width = int(w_str)
            height = int(h_str)
            app.logger.debug(f"Преобразовали size: width={width}, height={height}")

            # Если ширина/высота некратны 8 — корректируем
            new_width = round_to_nearest_multiple_of_8(width)
            new_height = round_to_nearest_multiple_of_8(height)

            if new_width != width or new_height != height:
                app.logger.warning(
                    f"Размеры {width}x{height} некратны 8. "
                    f"Скорректируем до {new_width}x{new_height}."
                )
                width, height = new_width, new_height

        except ValueError as e:
            app.logger.warning(f"Не удалось разобрать значение size '{size_str}'! Используем 512x512. Ошибка: {e}")
    else:
        app.logger.warning(f"Некорректный формат size '{size_str}'. Используем 512x512.")

    try:
        app.logger.debug("Запускаем генерацию изображения")
        # num_inference_steps и guidance_scale можно настроить под ваши задачи
        image = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]
        app.logger.debug("Генерация изображения завершена.")
    except Exception as e:
        app.logger.error(f"Ошибка при генерации изображения: {e}")
        return jsonify({"error": str(e)}), 500


    buf = BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")

    app.logger.info("Завершаем обработку запроса /generate. Отправляем изображение.")
    return jsonify({"image": encoded_image})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)