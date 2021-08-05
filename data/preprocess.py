import os

from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from torchvision import transforms

project_dir = '/home/hlz/High-Fidelity-Synthesis-with-Disentangled-Representation'

celeba_64_dir = Path(os.path.join(project_dir, 'data/CelebA_64'))
celeba_128_dir = Path(os.path.join(project_dir, 'data/CelebA_128'))
celeba_256_dir = Path(os.path.join(project_dir, 'data/CelebA_256'))

def preprocess_celeba(path):
    crop = transforms.CenterCrop((160, 160))
    resample = Image.LANCZOS
    img = Image.open(path)
    img = crop(img)

    img_256_path = celeba_256_dir / path.name 
    img.resize((256, 256), resample=resample).save(img_256_path)

    img_128_path = celeba_128_dir / path.name 
    img.resize((128, 128), resample=resample).save(img_128_path)

    img_64_path = celeba_64_dir / path.name 
    img.resize((64, 64), resample=resample).save(img_64_path)

    return None

paths = list(Path(os.path.join(project_dir, 'data/img_align_celeba')).glob('*.jpg'))

with Pool(16) as pool:
    pool.map(preprocess_celeba, paths)