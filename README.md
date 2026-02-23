# Multilabel Image Classification – Aimonk Assignment

## Framework
The implementation is done in PyTorch.

## Model
A ResNet50 pretrained on ImageNet was used. The final fully connected layer was replaced with a 4-unit output layer for multilabel classification.

## Transfer Learning
The network was initialized with pretrained weights and fine-tuned for the target dataset.

## Handling Missing Labels
Images with missing attributes (NA) were not removed. A masked binary cross entropy loss was used so that only available labels contribute to the loss.

## Handling Class Imbalance
Class-wise positive weights were computed and used in BCEWithLogitsLoss.

## Preprocessing
Resize to 224×224
Normalization using ImageNet statistics

## Data Augmentation
Random horizontal flip

## Training Details
Optimizer: Adam
Loss: Masked BCEWithLogitsLoss
Epochs: 15

## Outputs
-Trained model weights (.pth)
-Training loss curve
-Inference pipeline

## Future Improvements
-Fine-tuning deeper layers
-Focal loss
-Threshold tuning per attribute
-Test-time augmentation
-Weighted sampling

Due to time constraints the following techniques were not implemented but can further improve performance:

- Fine-tuning deeper ResNet layers
- Focal loss for severe imbalance
- Attribute-specific threshold tuning
- Test-time augmentation
- Weighted sampler
- MixUp / CutMix augmentation