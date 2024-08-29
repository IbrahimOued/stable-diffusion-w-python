# %% [markdown]
# Custom Stable Diffusion Pipeline
# ## Generating latent vectors using diffusers
# ### 1. Load an image
# We can use the load_image function from diffusers to load an image from local storage or a URL.
# In the following code, we load an image named dog.png from the same directory of the current program:
# %%
from diffusers.utils import load_image
from IPython.display import display

image = load_image("dog.png")
display(image)
# %% [markdown]
# ### 2. Pre-process the image
# Each pixel of the loaded image is represented by a number ranging from 0 to 255. The image encoder from the Stable Diffusion process handles image data ranging from -1.0 to 1.0.
# So, we first need to make the data range conversion:
# %%
import numpy as np

# convert image object to array and 
# convert pixel data from 0 ~ 255 to 0 ~ 1
image_array = np.array(image).astype(np.float32)/255.0
# convert the number from 0 ~ 1 to -1 ~ 1
image_array = image_array * 2.0 - 1.0
# %% [markdown]
# Now, if we use Python code, image_array.shape, to check the
# image_array data shape, we will see the shape of the image data as – (512,512,3), arranged as (width, height, channel),
# instead of the commonly used (channel, width, height). Here, we need to convert the image data shape to (channel, width, height) or (3,512,512), using the transpose() function:
# %%
# transform the image array from width,height,
# channel to channel,width,height
image_array_cwh = image_array.transpose(2,0,1)
# %% [markdown]
# The 2 is in the first position of 2, 0, 1, which means moving the original third dimension (indexed as 2) to the first dimension.
# The same logic applies to 0 and 1. The original 0 dimension is now converted to the second position, and the original 1 is now in the third dimension.

# With this transpose operation, the NumPy array, image_array_cwh, is now in the (3,512,512) shape.

# The Stable Diffusion image encoder handles image data in batches, which, in this instance is four-dimensional data with the batch dimension
# in the first position; we need to add the batch dimension here:
# %%
# add batch dimension
image_array_cwh = np.expand_dims(image_array_cwh, axis = 0)
# %% [markdown]
# ### 3. Loading image data with torch and move to CUDA

# We will convert the image data to latent space using CUDA.
# To achieve this, we will need to load the data into the CUDA VRAM before handing it off to the next step model:
# %%
# load image with torch
import torch

image_array_cwh = torch.from_numpy(image_array_cwh)
image_array_cwh_cuda = image_array_cwh.to(
    "cuda",
    dtype=torch.float16
)
# %% [markdown]
# ### 4. Load the Stable Diffusion image encoder VAE
# This VAE model is used to convert the image from pixel space to latent space:
# %%
# Initialize VAE model
import torch

from diffusers import AutoencoderKL

vae_model = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "vae",
    torch_dtype=torch.float16
).to("cuda")
# %% [markdown]
# ### 5. Encode the image into a latent vector
# Now, everything is ready, and we can encode any image into a latent vector as PyTorch tensor:
# %%
latents = vae_model.encode(
    image_array_cwh_cuda).latent_dist.sample()
# Check the data and shape of the latent data:

print(latents[0])
print(latents[0].shape)
# %% [markdown]
# We can see that the latent is in the (4, 64, 64) shape, with each element in the range of -1.0 to 1.0.

# Stable Diffusion processes all the denoising steps on a 64x64 tensor 
# with 4-channel for a 512x512 image generation. The data size is way
# less than its original image size, 512x512 with three color channels.
# %% [markdown]
# ### 6. Decode latent to image (optional)
# 
# You may be wondering, can I convert the latent data back to the pixel image? Yes, we can do this with lines of code:
# %%
import numpy as np

from PIL import Image

def latent_to_img(latents_input, scale_rate = 1):
    latents_2 = (1 / scale_rate) * latents_input
    # decode image
    with torch.no_grad():
        decode_image = vae_model.decode(
            latents_input, 
            return_dict = False
        )[0][0]

    decode_image = (decode_image / 2 + 0.5).clamp(0, 1)
    # move latent data from cuda to cpu
    decode_image = decode_image.to("cpu")
    # convert torch tensor to numpy array
    numpy_img = decode_image.detach().numpy()
    # covert image array from (width, height, channel) 
    # to (channel, width, height)
    numpy_img_t = numpy_img.transpose(1,2,0)
    # map image data to 0, 255, and convert to int number
    numpy_img_t_01_255 = \
        (numpy_img_t*255).round().astype("uint8")

    # shape the pillow image object from the numpy array
    return Image.fromarray(numpy_img_t_01_255)

pil_img = latent_to_img(latents_input)
pil_img
# %% [markdown]
# ## Generating text embeddings using CLIP
# To generate the text embeddings (the embeddings contain the image features), we need first to tokenize the input text or prompt and then encode the token IDs into embeddings. Here are steps to achieve this:

# ### 1. Get the prompt token IDs:
# %%
input_prompt = "a running dog"

# input tokenizer and clip embedding model

import torch
from transformers import CLIPTokenizer,CLIPTextModel

# initialize tokenizer

clip_tokenizer = CLIPTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "tokenizer",
    dtype = torch.float16
)

input_tokens = clip_tokenizer(
    input_prompt,
    return_tensors = "pt"
)["input_ids"]

input_tokens
# %% [markdown]
# The preceding code will convert the a running dog text prompt to a token ID list as a torch tensor object – tensor([[49406, 320, 2761, 1929, 49407]]).

# ### 2. Encode the token IDs into embeddings:
# %%
# initialize CLIP text encoder model

clip_text_encoder = CLIPTextModel.from_pretrained(

    "runwayml/stable-diffusion-v1-5",
    subfolder="text_encoder",
    # dtype=torch.float16
).to("cuda")

# encode token ids to embeddings

prompt_embeds = clip_text_encoder(
    input_tokens.to("cuda")
)[0]
# %% [markdown]
# ### 3. Check the embedding data:

print(prompt_embeds)

print(prompt_embeds.shape)
# %% [markdown]
# Now, we can see the data of prompt_embeds as follows:
# Its shape is torch.Size([1, 5, 768]). Each token ID is encoded into a 768-dimension vector.
# %% [markdown]
# ### 4. Generate embedding for negative prompt embeddings: Even though we don’t have the negative prompt, we’ll also prepare an embedding vector with the same size as the input prompt. This will ensure that our code will support both only prompt and prompt/negative prompt cases:
# %%
# prepare neg prompt embeddings
uncond_tokens = "blur"
# get the prompt embedding length
max_length = prompt_embeds.shape[1]
# generate negative prompt tokens with the same length of prompt
uncond_input_tokens = clip_tokenizer(
    uncond_tokens,
    padding = "max_length",
    max_length = max_length,
    truncation = True,
    return_tensors = "pt"
)["input_ids"]

# generate the negative embeddings

with torch.no_grad():
    negative_prompt_embeds = clip_text_encoder(
        uncond_input_tokens.to("cuda")
        )[0]
# %% [markdown]
# ### 5. Concatenate prompt and negative prompt embedding into one vector: Because we will feed the whole prompt into UNet at once, and then handle the positive and negative signals at the UNet inference stage, we will concatenate the prompt and negative prompt embeddings into one torch vector:
# %%
prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
# %% [markdown]
# Next, we will initialize the time step data.
# ## Initializing time step embeddings
# We introduced the scheduler in Chapter 3. By using the scheduler, we can
# sample key steps for image generation. Instead of denoising 1,000 steps
# to generate an image in the original diffusion model (DDPM), by using a
# scheduler, we can generate an image in a mere 20 steps.

# In this section, we are going to use the Euler scheduler to generate
# time step embeddings, and then we’ll take a look at what the time step
# embeddings look like. No matter how good the diagram that tries to plot
# the process is, we can only understand how it works by reading the actual
# data and code:

# ### 1. Initialize a scheduler from the scheduler configuration for the model:
# %%
from diffusers import EulerDiscreteScheduler as Euler

# initialize scheduler from a pretrained checkpoint

scheduler = Euler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "scheduler"
)
# %% [markdown]
# The preceding code will initialize a scheduler from the checkpoint’s scheduler config file. Note that you can also create a scheduler, as we discussed in Chapter 3, like this:
# %%
import torch

from diffusers import StableDiffusionPipeline

from diffusers import EulerDiscreteScheduler as Euler

text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16
).to("cuda:0")

scheduler = Euler.from_config(text2img_pipe.scheduler.config)
# %% [markdown]
# However, this will require you to load a model first, which is not only slow but also unnecessary; the only thing we need is the model’s scheduler.

# Sample the steps for the image diffusion process:
# ### Sample the steps for the image diffusion process
inference_steps = 20
scheduler.set_timesteps(inference_steps, device = "cuda")
timesteps = scheduler.timesteps

for t in timesteps:
    print(t)
# %% [markdown]
# Here, the scheduler takes 20 steps out of the 1,000 steps, and those 20
# steps may be enough to denoise a complete Gaussian distribution for image
# generation. This step sampling technique also contributes to Stable Diffusion
# performance boosting.
# %% [markdown]
# ## Initializing the Stable Diffusion UNet
# The UNet architecture [5] was introduced by Ronneberger et al. for biomedical image segmentation purposes. Before the UNet architecture, a convolution network was commonly used for image classification tasks. When using a convolution network, the output is a single class label. However, in many visual tasks, the desired output should include localization too, and the UNet model solved this problem.

# The U-shaped architecture of UNet enables efficient learning of features at different scales. UNet’s skip connections directly combine feature maps from different stages, allowing a model to effectively propagate information across various scales. This is crucial for denoising, as it ensures the model retains both fine-grained details and global context during noise removal. These features make UNet a good candidate for the denoising model.

# In the Diffuser library, there is a class named UNet2DconditionalModel; this is a conditional 2D UNet model for image generation and related tasks. It is a key component of diffusion models and plays a crucial role in the image generation process. We can load a UNet model in just several lines of code, like this:
# %%
import torch
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder ="unet",
    torch_dtype = torch.float16
).to("cuda")
# %% [markdown]
# Together with the UNet model we have just loaded up, we have all the components required by Stable Diffusion. Not that hard, right? Next, we are going to use those building blocks to build two Stable Diffusion pipelines – one text-to-image and another image-to-image.
# %% [markdown]
# ## Implementing a text-to-image Stable Diffusion inference pipeline
# So far, we have all the text encoder, image VAE, and denoising UNet model initialized and loaded into the CUDA VRAM. The following steps will chain them together to form the simplest and working Stable Diffusion text-to-image pipeline:

# ### 1 Initialize a latent noise: In Figure 5.2, the starting point of inference is randomly initialized Gaussian latent noise. We can create one of the latent noise with this code:
# %%
# prepare noise latents

shape = torch.Size([1, 4, 64, 64])

device = "cuda"

noise_tensor = torch.randn(
    shape,
    generator = None,
    dtype = torch.float16

).to("cuda")
# %% [markdown]
# During the training stage, an initial noise sigma is used to help prevent the diffusion process from becoming stuck in local minima. When the diffusion process starts, it is very likely to be in a state where it is very close to a local minimum. init_noise_sigma = 14.6146 is used to help avoid this. So, during the inference, we will also use init_noise_sigma to shape the initial latent.
# %%
# scale the initial noise by the standard deviation required by the scheduler
latents = noise_tensor * scheduler.init_noise_sigma
# %% [markdown]
# ### 2. Loop through UNet: With all those components prepared, we are finally at the stage of feeding the initial latents to UNet to generate the target latent we want:
# %%
guidance_scale = 7.5

latents_sd = torch.clone(latents)

for i,t in enumerate(timesteps):
    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents_sd] * 2)
    latent_model_input = scheduler.scale_model_input(
        latent_model_input, t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict = False,
        )[0]

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1

    latents_sd = scheduler.step(
        noise_pred,
        t,
        latents_sd,
        return_dict=False)[0]

# %% [markdown]

# ### 3 Recover the image from the latent: We can reuse the latent_to_img function to recover the image from the latent space:
# %%
import numpy as np
from PIL import Image

def latent_to_img(latents_input):

    # decode image
    with torch.no_grad():
        decode_image = vae_model.decode(
            latents_input,
            return_dict = False
        )[0][0]

    decode_image = (decode_image / 2 + 0.5).clamp(0, 1)
    # move latent data from cuda to cpu
    decode_image = decode_image.to("cpu")
    # convert torch tensor to numpy array
    numpy_img = decode_image.detach().numpy()
    # covert image array from (channel, width, height) 
    # to (width, height, channel)
    numpy_img_t = numpy_img.transpose(1,2,0)
    # map image data to 0, 255, and convert to int number

    numpy_img_t_01_255 = \
        (numpy_img_t*255).round().astype("uint8")

    # shape the pillow image object from the numpy array
    return Image.fromarray(numpy_img_t_01_255)

latents_2 = (1 / 0.18215) * latents_sd

pil_img = latent_to_img(latents_2)
# %% [markdown]
# ## Implementing a text-guided image-to-image Stable Diffusion inference pipeline
# The only thing we need to do now is concatenate the starting image with the starting latent noise. The latents_input Torch tensor is the latent we encoded from a dog image earlier in this chapter:
# %%
strength = 0.7
# scale the initial noise by the standard deviation required by the 
# scheduler
latents = latents_input*(1-strength) + \
    noise_tensor*scheduler.init_noise_sigma
# %% [markdown]
# Note that the preceding code uses strength = 0.7; the strength denotes the weight of the
# original latent noise. If you want an image more similar to the initial image (the image
# you provided to the image-to-image pipeline), use a lower strength number; otherwise, increase it.