import typing as t
import random
import numpy as np
import torch
from torch import nn


class GridMixAugLoss(nn.Module):
    def __init__(
            self,
            alpha: t.Tuple[float, float] = (0.1, 0.9),
            n_holes_x: t.Union[int, t.Tuple[int, int]] = 20,
            hole_aspect_ratio: t.Union[float, t.Tuple[float, float]] = 1.,
    ):
        super().__init__()
        self.alpha = alpha
        self.n_holes_x = n_holes_x
        self.hole_aspect_ratio = hole_aspect_ratio
        self.loss = nn.CrossEntropyLoss()

    def __str__(self):
        return "gridmix"

    @staticmethod
    def _get_gridmask(image_shape: t.Tuple[int, int], lam: float, nx: int, ar: float) -> np.ndarray:
        """ Method make grid mask

        :param image_shape: Shape of the images
        :param lam: Lambda parameter
        :param nx: Amount of holes by width
        :param ar: Aspect ratio of the hole
        :return:
        """
        holes = []
        height, width = image_shape

        if not 1 <= nx <= width // 2:
            raise ValueError("The hole_number_x must be between 1 and image width//2.")

        patch_width = width // nx
        patch_height = int(patch_width * ar)
        ny = height // patch_height

        # Calculate ratio of the hole - percent of hole in the patch
        ratio = np.sqrt(((1 - lam) * height * width) / (patch_width * nx * patch_height * ny))

        # Get hole size
        hole_width = int(patch_width * ratio)
        hole_height = int(patch_height * ratio)

        # min 1 pixel and max unit length - 1
        hole_width = min(max(hole_width, 1), patch_width - 1)
        hole_height = min(max(hole_height, 1), patch_height - 1)

        # set offset of the grid
        shift_x = 0
        shift_y = 0

        # Make grid holes
        for i in range(width // patch_width + 1):
            for j in range(height // patch_height + 1):
                x1 = min(shift_x + patch_width * i, width)
                y1 = min(shift_y + patch_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))

        mask = np.zeros(shape=image_shape, dtype=np.uint8)
        for x1, y1, x2, y2 in holes:
            mask[y1:y2, x1:x2] = 1
        return mask

    def get_sample(self, images: torch.Tensor, targets: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """ Method returns augmented images and targets

        :param images: Batch of non-augmented images
        :param targets: Batch of non-augmented targets
        :return: Augmented images and targets
        """
        # Get new indices
        indices = torch.randperm(images.size(0)).to(images.device)

        # Shuffle labels
        shuffled_targets = targets[indices].to(targets.device)

        # Get lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # Get image shape and grid mask
        height, width = images.shape[2:]
        if isinstance(self.n_holes_x, int):
            self.n_holes_x = (self.n_holes_x, self.n_holes_x)
        nx = random.randint(self.n_holes_x[0], self.n_holes_x[1])
        if isinstance(self.hole_aspect_ratio, float):
            self.hole_aspect_ratio = (self.hole_aspect_ratio, self.hole_aspect_ratio)
        ar = np.random.uniform(self.hole_aspect_ratio[0], self.hole_aspect_ratio[1])
        mask = self._get_gridmask(image_shape=(height, width), lam=lam, nx=nx, ar=ar)

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - (mask.sum() / (images.size()[-1] * images.size()[-2]))
        mask = torch.from_numpy(mask).to(targets.device)
        images = images * (1 - mask) + images[indices, ...] * mask
        lam_list = torch.from_numpy(np.ones(shape=targets.shape) * lam).to(targets.device)
        out_targets = torch.cat([targets, shuffled_targets, lam_list], dim=1).transpose(0, 1).unsqueeze(-1)

        return images, out_targets

    def forward(self, preds: torch.Tensor, trues: torch.Tensor):
        lam = trues[-1, :][0].float()
        trues1, trues2 = trues[0, :].long(), trues[1, :].long()
        loss = self.loss(preds, trues1) * lam + self.loss(preds, trues2) * (1 - lam)
        return loss
