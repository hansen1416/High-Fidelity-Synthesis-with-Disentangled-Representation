import os
import sys
import random
from math import sqrt

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch import distributions
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from tqdm import tqdm

from BetaVAE import BetaVAE_H, normal_init
from Dataloader import return_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        raise NotImplementedError

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    kld = (-0.5*(1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0, True)
    return kld


class Solver(object):
    def __init__(self, max_iter, data_loader, load_ckpt=None):
        
        self.global_iter = 0

        self.max_iter = max_iter # 1e6 #3e5
        self.ckpt_save_iter = 10000
        self.log_line_iter = 100
        self.log_img_iter = 100
        # image output dir
        # self.output_dir = os.path.join(project_dir, 'dvae_celeba_output')
        # checkpoint save dir
        self.ckpt_dir = os.path.join(project_dir, 'ckpt')
            
        self.log_dir = os.path.join(project_dir, 'tensorboard')
        self.writer = SummaryWriter(self.log_dir)

        self.data_loader = data_loader

        self.nc = 3

        self.c_dim = 20
        self.beta = 6.4
        
        self.dec_dist = 'gaussian'

        self.net = BetaVAE_H(self.c_dim, self.nc).to(device)
        self.net.apply(normal_init)

        self.optim = optim.Adam(self.net.parameters(), lr=1e-4,
                                betas=(0.9, 0.999))
        
        if load_ckpt is not None:
            self.load_checkpoint(str(load_ckpt))

    def train(self):
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        out = False if self.global_iter < self.max_iter else True
        while not out:
            for x in self.data_loader:
                self.net.train()
                self.global_iter += 1
                pbar.update(1)

                x = x.to(device)
                x_recon, c, mu, logvar = self.net(x)

                recon_loss = reconstruction_loss(x, x_recon, self.dec_dist)
                kld = kl_divergence(mu, logvar)
                beta_vae_loss = recon_loss + self.beta*kld

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                pbar.set_description('[{}] recon_loss:{:.3f} kld:{:.3f}'.format(
                    self.global_iter, recon_loss.item(), kld.item()))

                if self.global_iter % self.log_line_iter == 0:
                    self.writer.add_scalar('recon_loss', recon_loss, self.global_iter)
                    self.writer.add_scalar('kld', kld, self.global_iter)

                if self.global_iter % self.log_img_iter == 0:
                    # visualize reconstruction 
                    x = make_grid(x, nrow=int(sqrt(x.size(0))), padding=2, pad_value=1)
                    x_recon = make_grid(x_recon.sigmoid(), nrow=int(sqrt(x_recon.size(0))), padding=2, pad_value=1)
                    x_vis = make_grid(torch.stack([x, x_recon]), nrow=2, padding=2, pad_value=0)
                    self.writer.add_image('reconstruction', x_vis, self.global_iter)

                    # visualize traverse
                    self.traverse(c_post=mu[:1], c_prior=torch.randn_like(mu[:1]))


                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint()
                    pbar.write('Saved checkpoint (iter:{})'.format(self.global_iter))

                if self.global_iter >= self.max_iter:
                    self.save_checkpoint()
                    pbar.write('Saved checkpoint (iter:{})'.format(self.global_iter))
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def traverse(self, c_post, c_prior, limit=3, npoints=7, pos=-1):
        assert isinstance(pos, (int, list, tuple))

        self.net.eval()
        c_dict = {'c_posterior':c_post, 'c_prior':c_prior}
        interpolation = torch.linspace(-limit, limit, npoints)

        for c_key in c_dict:
            c_ori = c_dict[c_key]
            samples = []
            for row in range(self.c_dim):
                if pos != -1 and row not in pos:
                    continue

                c = c_ori.clone()
                for val in interpolation:
                    c[:, row] = val
                    sample = self.net(c=c, decode_only=True).sigmoid().data
                    samples.append(sample)

            samples = torch.cat(samples, dim=0).cpu()
            samples = make_grid(samples, nrow=npoints, padding=2, pad_value=1)
            tag = 'latent_traversal_{}'.format(c_key)
            self.writer.add_image(tag, samples, self.global_iter)

        self.net.train()

    def save_checkpoint(self):
        model_states = {'net':self.net.state_dict(),
                        'c_dim':self.c_dim,
                        'nc':self.nc}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, str(self.global_iter))
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

        # file_path = os.path.join(self.ckpt_dir, 'last')
        # with open(file_path, mode='wb+') as f:
        #     torch.save(states, f)

    def load_checkpoint(self, filename):

        file_path = os.path.join(self.ckpt_dir, filename)

        if os.path.isfile(file_path):
            
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])

            tqdm.write("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            tqdm.write("=> no checkpoint found at '{}'".format(file_path))

if __name__== "__main__":

    torch.backends.cudnn.enabled =True  # 说明设置为使用使用非确定性算法
    torch.backends.cudnn.benchmark = True

    seed = 46

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    project_dir = '/home/hlz/High-Fidelity-Synthesis-with-Disentangled-Representation'

    celeba64dataloader = return_data()

    net = Solver(max_iter=500000, data_loader=celeba64dataloader, load_ckpt=250000)

    net.train()