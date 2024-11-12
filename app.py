import cv2
import os
import numpy as np
from skimage.filters import gaussian  # No change here
from test import evaluate
import streamlit as st
from PIL import Image, ImageColor

def sharpen(img):
    img = img * 1.0
    # Apply Gaussian blur without the 'multichannel' argument
    gauss_out = gaussian(img, sigma=5)  # No multichannel argument

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r
    np.repeat(parsing[:, :, np.newaxis], 3, axis=2)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)
    
    # Apply parsing mask
    changed[parsing != part] = image[parsing != part]
    return changed


DEMO_IMAGE = 'imgs/116.jpg'

st.title('Virtual Makeup Application')

st.sidebar.title('Virtual Makeup')
st.sidebar.subheader('Parameters')

table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
    }

img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    demo_image = img_file_buffer

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

# original image
st.subheader('Original Image')
st.image(image)

# variable for model path
cp = 'cp/79999_iter.pth'
ori = image.copy()
h, w, _ = ori.shape

image = cv2.resize(image, (1024, 1024))

parsing = evaluate(demo_image, cp)
parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

parts = [table['hair'], table['upper_lip'], table['lower_lip']]

# color picks from streamlit for hair and lips
hair_color = st.sidebar.color_picker('Pick the Hair Color', '#8B4513')
hair_color = ImageColor.getcolor(hair_color, "RGB")

lip_color = st.sidebar.color_picker('Pick the Lip Color', '#edbad1')
lip_color = ImageColor.getcolor(lip_color, "RGB")

colors = [hair_color, lip_color, lip_color]

for part, color in zip(parts, colors):
    image = hair(image, parsing, part, color)

image = cv2.resize(image, (w, h))

# Display the output image
st.subheader('Output Image')
st.image(image)
