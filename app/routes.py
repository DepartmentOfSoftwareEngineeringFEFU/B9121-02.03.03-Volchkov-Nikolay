import os
import datetime
import logging
import base64
import json
from io import BytesIO
from flask import Blueprint, request, Response, stream_with_context
from PIL import Image
from .model_loader import load_pipeline
import threading
from queue import Queue, Empty

routes_bp = Blueprint("routes", __name__)
pipe = load_pipeline()

def round_to_multiple_of_8(x: int) -> int:
    return (x // 8) * 8

@routes_bp.route("/generate", methods=["POST"])
def generate_image():
    """
    Отправляет прогресс-ивенты и финальный base64 изображения в одном стриме.
    Клиент получает JSON-строки:
      {"progress": N}
      ...
      {"progress":100, "image":"<base64>"}
    """
    data = request.json or {}
    prompt = data.get("prompt", "")
    size_str = data.get("size", "512x512")
    steps = data.get("steps", 50)

    # parse user size
    try:
        w, h = map(int, size_str.split('x'))
    except ValueError:
        w, h = 512, 512
        logging.getLogger(__name__).warning(f"Invalid size '{size_str}', using default 512x512")

    # adjust to multiple of 8
    gen_w, gen_h = round_to_multiple_of_8(w), round_to_multiple_of_8(h)

    # prepare queue and storage
    q = Queue()
    container = {}

    # callback for progress
    def progress_cb(step, timestep, latents=None):
        percent = int(step / steps * 100)
        q.put({"progress": percent})

    # worker function
    def worker():
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            width=gen_w,
            height=gen_h,
            callback=progress_cb,
            callback_steps=1
        ).images[0]
        # resize back to original
        if (gen_w, gen_h) != (w, h):
            image = image.resize((w, h), Image.LANCZOS)
        # save image to disk
        output_dir = os.path.join(os.getcwd(), 'generated_images')
        os.makedirs(output_dir, exist_ok=True)
        filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.png')
        path = os.path.join(output_dir, filename)
        image.save(path)
        logging.getLogger(__name__).info(f"Saved image to {path}")
        container['image'] = image
        q.put({"progress": 100, "done": True})

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        log_path = os.path.join(os.getcwd(), 'generated_images', 'sent_payloads.log')
        while True:
            try:
                msg = q.get(timeout=0.1)
            except Empty:
                continue
            if msg.get('done'):
                # final payload
                buf = BytesIO()
                img = container['image']
                img.save(buf, format='PNG')
                encoded = base64.b64encode(buf.getvalue()).decode()
                payload = {"progress": 100, "image": encoded}
                # log payload
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps(payload) + '\n')
                yield json.dumps(payload) + '\n'
                break
            else:
                payload = {"progress": msg['progress']}
                # log payload
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps(payload) + '\n')
                yield json.dumps(payload) + '\n'

    return Response(stream_with_context(event_stream()), mimetype='application/json')