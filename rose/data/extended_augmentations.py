import torch
import math
import warnings
from typing import Optional, Sequence
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

class RandomGaussianResizedCrop(torch.nn.Module):
    """Crop a random portion of image using Gaussian distribution for `i, j` selection and resize it to a given size.

    If the image is a torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given size.
    The position for the crop is determined using a Gaussian distribution.

    Args:
        size (int or sequence): Expected output size of the crop, for each edge.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop.
        ratio (tuple of float): Lower and upper bounds for the random aspect ratio of the crop.
        interpolation (InterpolationMode): Desired interpolation enum. Default is ``InterpolationMode.BILINEAR``.
        antialias (bool, optional): Whether to apply antialiasing. Default is True.
        mean (tuple of float, optional): The mean of the Gaussian distribution for the `i, j` coordinates. Default is (0.5, 0.5).
        std (tuple of float, optional): The standard deviation of the Gaussian distribution. Default is (0.1, 0.1).
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        mean=(0.5, 0.5),
        std=(0.1, 0.1),
    ):
        super().__init__()
        # Validate and set the size (ensure size is a tuple)
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, Sequence) and len(size) == 1:
            size = (size[0], size[0])
        elif not isinstance(size, Sequence) or len(size) != 2:
            raise ValueError("Please provide only two dimensions (h, w) for size.")

        self.size = size

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio
        self.mean = mean
        self.std = std

    @staticmethod
    def get_params(img: Tensor, scale: list[float], ratio: list[float], mean: tuple[float], std: tuple[float]) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop using Gaussian distribution for `i, j` coordinates.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): Range of scale of the original size cropped.
            ratio (list): Range of aspect ratio of the original aspect ratio cropped.
            mean (tuple): The mean of the Gaussian distribution for `i, j`.
            std (tuple): The standard deviation of the Gaussian distribution.

        Returns:
            tuple: Params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                # Using Gaussian distribution to sample (i, j)
                i = int(torch.normal(float(mean[0] * (height - h)), float(std[0] * (height - h)), size=(1,)).item())
                j = int(torch.normal(float(mean[1] * (width - w)), float(std[1] * (width - w)), size=(1,)).item())

                # Clamp to ensure valid cropping region
                i = max(0, min(i, height - h))
                j = max(0, min(j, width - w))

                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio, self.mean, self.std)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias}"
        format_string += f", mean={self.mean}, std={self.std})"
        return format_string