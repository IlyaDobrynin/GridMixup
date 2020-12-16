# GridMix
A GridMixup augmentation, inspired by GridMask and CutMix

## Overview
This simple augmentation is inspired by the GridMask (https://arxiv.org/abs/2001.04086) and CutMix (https://arxiv.org/abs/1905.04899) augmentations.
The combination of this two augmentations form the proposed method.

### Example
**Before**<br>
![](images/img.png)<br>

**After**<br>
![](images/img_1.png)

GridMixup loss defined as:

`lam * CrossEntropyLoss(preds, trues1) + (1 - lam) * CrossEntropyLoss(preds, trues2)`

where:
- `lam` - the area of the main image
- `(1 - lam)` - area of the slave image 

### Parameters
GridMixupLoss takes follow arguments:
- `alpha` - parameter define area of the main image in mixed image. Could be `float` or `Tuple[float, float]`.
    - if `float`: lambda parameter gets from the beta-dictribution np.random.beta(alpha, alpha)
    - if `Tuple[float, float]`: lambda parameter gets from the uniform distribution np.random.uniform(alpha[0], alpha[1])
- `n_holes_x` - numper of holes in crop by X axis
- `hole_aspect_ratio` - aspect ratio of holes
- `crop_area_ratio` - parameter define area of the minor image in mixed image.
- `crop_aspect_ratio` - aspect ratio of crop