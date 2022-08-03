import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MIN_OFFSET = 0
MAX_OFFSET = 8
MIN_SPACING = 2
MAX_SPACING = 6


def grid(image_array: np.ndarray, offset: tuple, spacing: tuple):
    target = image_array.copy()  # copy to avoid modifying the original image
    known_array = np.zeros_like(image_array)  # array to keep track of known pixels
    known_array[
        :, offset[1] :: spacing[1], offset[0] :: spacing[0]
    ] = 1  # set known pixels to 1
    target_array = image_array[known_array == 0].copy()  # array of unknown pixels
    image_array[known_array == 0] = 0  # set unknown pixels to 0
    return (
        target,
        image_array,
        known_array,
        target_array,
    )  # return target, image, known, target_array


class ImageDataset(Dataset):
    def __init__(
        self, image_paths, transform=False, random_offset=False, random_spacing=False
    ):
        self.image_paths = image_paths  # list of paths
        self.transform = transform  # albumentations transform
        self.random_offset = random_offset  # random offset
        self.random_spacing = random_spacing  # random spacing

    def __len__(self):
        return len(self.image_paths)  # of how many data(images?) you have

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]  # get the image path
        image_file = Image.open(image_filepath)  # open the image
        image_file = np.asarray(image_file, dtype=np.float32)  # convert to numpy array

        if self.transform is not None:
            image_file = self.transform(image=image_file)[
                "image"
            ]  # apply the transform

        if self.random_offset:
            offset = np.random.randint(MIN_OFFSET, MAX_OFFSET, size=2)  # random offset
        else:
            offset = (1, 1)

        if self.random_spacing:
            spacing = np.random.randint(
                MIN_SPACING, MAX_SPACING, size=2
            )  # random spacing
        else:
            spacing = (2, 2)

        target, image_array, known_array, _ = grid(
            np.asarray(image_file, dtype=np.float32), offset, spacing
        )  # apply the grid
        known_array = known_array[
            0:1, ::, ::
        ]  # remove the channel dimension (3, 256, 256) -> (256, 256)

        full_image = torch.cat(
            (torch.from_numpy(image_array), torch.from_numpy(known_array)), 0
        )  # concatenate the image and the known array

        return full_image, target  # return the image and the target
