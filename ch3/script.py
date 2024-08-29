# %%
import os
import torch
torch.cuda.empty_cache()
from diffusers import StableDiffusionPipeline
from diffusers import FluxPipeline
# %%

HF_TOKEN = os.getenv("HF_TOKEN")

text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    token=HF_TOKEN, device_map="balanced")
# %%
# Generate the image
prompt ="high resolution, a photograph of an astronaut riding a horse"

image = text2img_pipe(
    prompt, num_inference_steps=30
).images[0]
# %%
image.save(f"./images/stable-diffusion/{prompt}-1.png")
# %% Generation seed

seed = 1234
generator = torch.Generator().manual_seed(seed)
prompt ="high resolution, a photograph of an astronaut riding a horse"

image = text2img_pipe(
    prompt,
    generator=generator
).images[0]

# %%
image.save(f"./images/stable-diffusion/{prompt}-2.png")
# %%
## Sampling scheduler

# The original Diffusion models have demonstrated impressive results in generating images.
# However, one drawback is the slow reverse-denoising process, which typically requires 1,000
# steps to transform a random noise data space into a coherent image (specifically, latent data space, a
# concept we will explore further in Chapter 4). This lengthy process can be burdensome.

# In the Hugging Face Diffusers package, these helpful components are referred to as schedulers.
# However, you may also encounter the term sampler in other resources. You may take a look at the
# Diffusers Schedulers [2] page for the latest supported schedulers.

# By default, the Diffusers package uses 'PNDMScheduler'. We can find it by running this line of code:

# check out the current scheduler
text2img_pipe.scheduler
# %%
# Based on my experience, the Euler scheduler is one of the top choices.
# Let’s apply the Euler scheduler to generate an image:
from diffusers import EulerDiscreteScheduler
text2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
    text2img_pipe.scheduler.config)
generator = torch.Generator("cuda:3").manual_seed(1234)
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = generator
).images[0]
# %%
image.save(f"./images/stable-diffusion/{prompt}-3.png")
# %%
# You can customize the number of denoising steps by using the num_inference_steps parameter.
# A higher step count generally leads to better image quality. Here, we set the scheduling
# steps to 20 and compared the results of the default PNDMScheduler and EulerDiscreteScheduler:


# Euler scheduler with 20 steps
from diffusers import EulerDiscreteScheduler
text2img_pipe.scheduler = EulerDiscreteScheduler.from_config(text2img_pipe.scheduler.config)
generator = torch.Generator("cuda:3").manual_seed(1234)
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = generator,
    num_inference_steps = 20
).images[0]
# %%
image.save(f"./images/stable-diffusion/{prompt}.png")
# %%
# Change model to "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = """high resolution, a photograph of Lionnel Messi dunking on Lebron James during the NBA playoffs"""

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda:3").manual_seed(0)
).images[0]
image.save(f"./images/flux.1/{prompt}.png")
# %%
# Guidance scale or Classifier-Free Guidance (CFG) is a parameter that controls the adherence of the generated image to the text prompt.
# A higher guidance scale will force the image to be more aligned with the prompt, while a lower guidance scale will give more space for
# Stable Diffusion to decide what to put into the image.
generator = torch.Generator("cuda:3").manual_seed(123)
prompt ="high resolution, a photograph of a maestro cat conducting an orchestra of dogs"
image_3_gs = text2img_pipe(
    prompt = prompt,
    num_inference_steps = 30,
    guidance_scale = 3,
    generator = generator
).images[0]
image_7_gs = text2img_pipe(
    prompt = prompt,
    num_inference_steps = 30,
    guidance_scale = 7,
    generator = generator
).images[0]
image_10_gs = text2img_pipe(
    prompt = prompt,
    num_inference_steps = 30,
    guidance_scale = 10,
    generator = generator
).images[0]
from diffusers.utils import make_image_grid
images = [image_3_gs,image_7_gs,image_10_gs]
for i, img in enumerate(images):
    img.save(f"./images/stable-diffusion/{prompt}-gs-{i}.png")
make_image_grid(images, rows=1, cols=3)
# %%