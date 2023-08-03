from flask import Flask
from flask import request, make_response
from flask_cors import CORS
# from auth_token import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
pipe.to(device)

pipe.enable_attention_slicing()

@app.get("/")
def generate(): 
    prompt = request.args.get("prompt", "")
    try:
        with autocast(device): 
            image = pipe(prompt, guidance_scale=8.5).images[0]

        image.save("testimage.png")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        return make_response({"image": prompt})
    except Exception as e:
        return make_response({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
