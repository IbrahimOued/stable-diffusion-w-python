# %%
import numpy as np
import matplotlib.pyplot as plt
import ipyplot
from PIL import Image
# Load an image
img_path = r"./dog.png"
image = plt.imread(img_path)
# Parameters
num_iterations = 16
beta = 0.1  # noise_variance
images = []
steps = ["Step:"+str(i) for i in range(num_iterations)]
# Forward diffusion process
for i in range(num_iterations):
    mean = np.sqrt(1 - beta) * image
    image = np.random.normal(mean, beta, image.shape)
    # convert image to PIL image object
    pil_image = Image.fromarray((image * 255).astype('uint8'), 'RGB')
    # add to image list
    images.append(pil_image)

ipyplot.plot_images(images, labels=steps, img_width=120)
# %%