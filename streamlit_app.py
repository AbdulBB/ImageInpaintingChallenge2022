import streamlit as st
from PIL import Image
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import torch
from datasets import grid

MIN_OFFSET = 0
MAX_OFFSET = 8
MIN_SPACING = 2
MAX_SPACING = 6

#st.set_page_config(layout="wide")


def load_image(image_file):
    image = Image.open(image_file)
    image = np.asarray(image, dtype=np.float32)
    return image

def rmsdiff(im1, im2):
    """Calculates the root mean square error (RSME) between two images"""
    return np.sqrt(((im1 - im2) ** 2).mean())


transforms = A.Compose(
    [
        A.Resize(
            100, 100, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1
        ),  # resize the image to 100x100
        ToTensorV2(),  # convert the image to a tensor
    ]
)


st.title("Image Inpainting")
st.caption("Abdul Basit Banbhan")


st.header("Introduction")

"""This was a simple image inpainting project where a simple convolutional Neural Network CNN is used to fill in the 
holes in an image. This was a challenge in the course 'Programming in Python 2' at the Johannes Kepler University 
Linz. I already did the course; however, we had a different project. I wanted to do the project to showcase the 
students in the . semester and my mentoring group. I explained the code and how to start a basic machine learning 
project in Python.You can find the code in the repository. """

st.header("The Problem 🕵🏻‍♂️")
"""
The problem is to fill in the holes in an image. The holes are represented by black pixels. 
The goal is to fill in the holes by training a neural network to predict unknown off-grid pixels of an
image. The neural network should be able to predict the color of the off-grid pixels. 
The image size is $100$x$100$ pixels which is fed to the neural network as a tensor.
We have offset and spacing parameters that can be used to control the size of the grid.
To understand the parameters better, you can see the image below.
"""
st.image("./recoursces/grid_specifications.png")

"""
The first value in the offset and spacing parameter is the offset in the $N$-direction. 
The second value is the offset and spacing in the $M$-direction.
"""

"""For better understanding, you can also see the image below. The input image is shown in the left, which is the 
image with the holes. The output image is shown in the right, which is the image with the filled in holes. The real 
output image is shown in the middle. """

st.image("./recoursces/exp.png")


st.header("The Model 🧠")

"""This problem can be tackled, with a simple convolutional neural network (CNN). I will not go into the details of 
the CNNs. Convolutional neural networks (CNNs, or ConvNets) are a type of artificial neural network (ANN) used most 
frequently in deep learning to interpret visual data. Based on the shared-weight design of the convolution kernels or 
filters that slide along input features and produce translation-equivalent responses known as feature maps. These 
feature maps can be plotted as images. More on this in a different blog post. Lets start with a simple CNN. As one can see
 below the CNN has $6$ hidden layers. All layer has a kernel size of $7$x$7$ and a stride of $1$ with a padding size of $3$x$3$.
 The output layer has $3$ output channels which represents the RGB channels."""

st.text("""SimpleCNN(
  (hidden_layers): Sequential(
    (0): Conv2d(4, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): ReLU()
    (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (3): ReLU()
    (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (5): ReLU()
    (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (7): ReLU()
    (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (9): ReLU()
    (10): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (11): ReLU()
  )
  (output_layer): Conv2d(128, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
)""")

"""
To see how many parameters the model has, one can see it below:
"""

st.text("""----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1        [-1, 128, 100, 100]          25,216
              ReLU-2        [-1, 128, 100, 100]               0
            Conv2d-3        [-1, 128, 100, 100]         802,944
              ReLU-4        [-1, 128, 100, 100]               0
            Conv2d-5        [-1, 128, 100, 100]         802,944
              ReLU-6        [-1, 128, 100, 100]               0
            Conv2d-7        [-1, 128, 100, 100]         802,944
              ReLU-8        [-1, 128, 100, 100]               0
            Conv2d-9        [-1, 128, 100, 100]         802,944
             ReLU-10        [-1, 128, 100, 100]               0
           Conv2d-11        [-1, 128, 100, 100]         802,944
             ReLU-12        [-1, 128, 100, 100]               0
           Conv2d-13          [-1, 3, 100, 100]          18,819
================================================================
Total params: 4,058,755
Trainable params: 4,058,755
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.15
Forward/backward pass size (MB): 117.42
Params size (MB): 15.48
Estimated Total Size (MB): 133.05
----------------------------------------------------------------

""")

"""
The model just has $4,058,755$ parameters which is not many compared to other models. 
"""

"""Lets start with playing around with the model. Here one can drag and drop a image from the folder 'pics' or just 
take a picture. With the slider you can control the size of the grid. The button 'Random offset and spacing' 
will generate a random offset and spacing for the grid. Have fun! 🥳

"""

st.header("Playground 🎢")

OFFSET = st.slider("↕️ Offset ↔️", MIN_OFFSET, MAX_OFFSET, (0, 1))
SPACING = st.slider("↕️ Spacing ️↔️", MIN_SPACING, MAX_SPACING, (2, 3))
clicked = st.button("Random Offset and Spacing 🔀")

image_file = st.file_uploader("Upload Images 📸", type=["png", "jpg", "jpeg"])


if image_file is not None:
    file_details = {
        "filename": image_file.name,
        "filetype": image_file.type,
        "filesize": image_file.size,
    }
    #st.write(file_details)
    image_array = load_image(image_file)

    image_file = transforms(image=image_array)["image"]  # apply the transform

    if clicked:
        offset = np.random.randint(MIN_OFFSET, MAX_OFFSET, size=2)  # random offset
        spacing = np.random.randint(MIN_SPACING, MAX_SPACING, size=2)  # random spacing
    else:
        offset = OFFSET
        spacing = SPACING

    target, image_array, known_array, _ = grid(
        np.asarray(image_file, dtype=np.float32), offset, spacing
    )  # apply the grid
    known_array = known_array[
        0:1, ::, ::
    ]  # remove the channel dimension (3, 256, 256) -> (256, 256)

    full_image = torch.cat(
        (torch.from_numpy(image_array), torch.from_numpy(known_array)), 0
    )  # concatenate the image and the known array

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        st.subheader("Input Image 📷")
        st.image(np.transpose(image_array.astype('uint8'), (1, 2, 0)), width=200)

    with col2:
        st.subheader("Target Image 🎯")
        st.image(np.transpose(target.astype('uint8'), (1, 2, 0)), width=200)

    with st.spinner('Wait for it...'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(r"./results/best_model.pt", map_location=torch.device('cpu'))
        model.to(device)
        with torch.no_grad():
            output = model(full_image.to(device))
        output = output.detach().cpu().numpy()
    st.success('Done!')
    
    with col3:
        st.subheader("Output Image🪄")
        st.image(np.transpose(output.astype('uint8'), (1, 2, 0)), width=200)
    st.write("Root Mean Square Error ",rmsdiff(target, output))






