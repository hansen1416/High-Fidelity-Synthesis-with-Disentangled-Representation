Download CelebA data from Google Drive

https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ

Use `./data/preprocess.py` to resize images to 64x64, 128x128, 256x256

### Train the model locally

You can train Beta VAE by running 

```python BetaVAETrainer.py```

It will save a checkpoint to `./ckpt`, which you will use when training GAN

To train the GAN model, you can just run

```python GANTrainer.py```

### Train the model in jupyter notebook

Copy ```idgan.ipynb``` to Google drive

### Use the Generator to generate a new image


Copy ```idgan_test.ipynb``` to Google drive
