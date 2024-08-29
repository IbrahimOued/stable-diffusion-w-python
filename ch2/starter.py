# %%
import os
import torch
torch.cuda.empty_cache()
from diffusers import StableDiffusionPipeline
# %%

HF_TOKEN = os.environ("HF_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16, token=HF_TOKEN)

pipe.to("cuda")
# %%
prompt = "a photo of an astronaut riding a horse on mars,blazing fast, wind and sand moving back"

image = pipe(
    prompt, num_inference_steps=30
).images[0]
# %%
image