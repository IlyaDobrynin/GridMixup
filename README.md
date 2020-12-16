# GridMix
A GridMix augmentation, inspired by GridMask and CutMix

## Overview
This simple augmentation is inspired by the GridMask (https://arxiv.org/abs/2001.04086) and CutMix (https://arxiv.org/abs/1905.04899) augmentations.
The combination of this two augmentations form the proposed method.

### Example
**Before**<br>
![](images/img.png)<br>

**After**<br>
![](images/img_1.png)

GridMix loss defined as:

`lam * CrossEntropyLoss(preds, trues1) + (1 - lam) * CrossEntropyLoss(preds, trues2)`

where:
- `lam` - the area of the main image
- `(1 - lam)` - area of the slave image 
