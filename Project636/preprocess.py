import torch
import numpy as np


def PreprocessImage(image: torch.Tensor, training: bool) -> torch.Tensor:
    image = image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    if training is True:
        # padding at sides
        image = np.lib.pad(image, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        # randomly horizontally shift
        ul = np.random.randint(8, size=(2,))
        image = image[ul[0]:ul[0] + 32, ul[1]:ul[1] + 32, :]
        if np.random.randint(2) == 0:
            image = np.fliplr(image)
    # normalization
    image = (image - np.mean(image)) / np.std(image)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)

def PreprocessImageBatch(images: torch.Tensor, training: bool):
    images = images.cpu().detach()
    for i in range(images.size()[0]):
        img = PreprocessImage(images[i], training)
        images[i] = img
    return images

