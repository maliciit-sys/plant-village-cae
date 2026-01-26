# Models

## Convolutional Autoencoder (CAE)
- `cae_encoder.pth` - Pre-trained encoder weights
- `cae_full.pth` - Full autoencoder model

## Classifier
- `classifier_final.pth` - Final trained classifier

## Production
- `production/model_config.json` - Model configuration
- `production/inference.py` - Standalone inference script
- `production/model_weights.pth` - Lightweight weights (not tracked)

### Training Results:
- CAE SSIM: 0.9756
- CAE PSNR: 40.62 dB
- Classifier F1: 0.9774
- Test Accuracy: 98.02%
