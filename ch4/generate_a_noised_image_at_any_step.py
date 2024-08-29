import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from itertools import accumulate
def get_product_accumulate(numbers):
    product_list = list(accumulate(numbers, lambda x, y: x * y))
    return product_list
# Load an image
img_path = r"./dog.png"
image = plt.imread(img_path)
image = image * 2 - 1   # [0,1] to [-1,1]
# Parameters
num_iterations = 16
beta = 0.05 # noise_variance
betas = [beta]*num_iterations
alpha_list = [1 - beta for beta in betas]
alpha_bar_list = get_product_accumulate(alpha_list)
target_index = 5
x_target = (
    np.sqrt(alpha_bar_list[target_index]) * image
    + np.sqrt(1 - alpha_bar_list[target_index]) * 
    np.random.normal(0,1,image.shape)
)
x_target = (x_target+1)/2
x_target = Image.fromarray((x_target * 255).astype('uint8'), 'RGB')

display(x_target)