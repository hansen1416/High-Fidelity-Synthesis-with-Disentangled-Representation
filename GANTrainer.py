import os
import math
import time
from pathlib import Path

import torch
from torch import autograd
import torch.optim as optim
from torch import distributions
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from BetaVAE import BetaVAE_H
from GANModel import Generator, Discriminator
from Checkpointio import CheckpointIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_zdist(dim, device=None):
    # Get distribution
    mu = torch.zeros(dim, device=device)
    scale = torch.ones(dim, device=device)
    zdist = distributions.Normal(mu, scale)

    # Add dim attribute
    zdist.dim = dim

    return zdist

def build_lr_scheduler(optimizer, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=150000,
        gamma=1,
        last_epoch=last_epoch
    )
    return lr_scheduler

class Trainer(object):
    def __init__(self, dvae, generator, discriminator, g_optimizer, d_optimizer,
                 reg_param, w_info):
        self.dvae = dvae
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.reg_param = reg_param
        self.w_info = w_info

    def generator_trainstep(self, z, cs):
        toogle_grad(self.generator, True)
        toogle_grad(self.dvae, True)
        toogle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.dvae.train()
        self.dvae.zero_grad()
        self.g_optimizer.zero_grad()

        loss = 0.
        c, c_mu, c_logvar = cs
        z_ = torch.cat([z, c], 1)
        x_fake = self.generator(z_)
        d_fake = self.discriminator(x_fake)

        gloss = self.compute_loss(d_fake, 1)
        loss += gloss

        chs = self.dvae(x_fake, encode_only=True)
        encloss = self.compute_infomax(cs, chs)
        loss += self.w_info*encloss

        loss.backward()
        self.g_optimizer.step()

        return gloss.item(), encloss.item()

    def discriminator_trainstep(self, x_real, z):
        toogle_grad(self.generator, False)
        toogle_grad(self.dvae, False)
        toogle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.dvae.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real)
        dloss_real = self.compute_loss(d_real, 1)
        dloss_real.backward(retain_graph=True)
        reg = self.reg_param * compute_grad2(d_real, x_real).mean()
        reg.backward()

        # On fake data
        with torch.no_grad():
            c, c_mu, c_logvar = cs = self.dvae(x_real, encode_only=True)
            z_ = torch.cat([z, c], 1)
            x_fake = self.generator(z_)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)
        dloss_fake.backward()

        self.d_optimizer.step()
        toogle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        return dloss.item(), reg.item(), cs

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

    def compute_infomax(self, cs, chs):
        c, c_mu, c_logvar = cs
        ch, ch_mu, ch_logvar = chs
        loss = (math.log(2*math.pi) + ch_logvar + (c-ch_mu).pow(2).div(ch_logvar.exp()+1e-8)).div(2).sum(1).mean()
        return loss


# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

if __name__ == "__main__":

    project_dir = '/home/hlz/High-Fidelity-Synthesis-with-Disentangled-Representation'

    celeba_64_dir = Path(os.path.join(project_dir, 'data/CelebA_64'))
    celeba_128_dir = Path(os.path.join(project_dir, 'data/CelebA_128'))
    celeba_256_dir = Path(os.path.join(project_dir, 'data/CelebA_256'))

    batch_size = 64
    d_steps = 1
    restart_every = -1
    inception_every = -1
    save_every = 1000
    backup_every = 100000

    # out_dir = os.path.join(project_dir, 'ckpt')
    checkpoint_dir = os.path.join(project_dir, 'ckpt')

    c_dim = 20
    z_dist_dim = 256
    nc = 3
    img_size = 64

    nfilter_generator = 64
    nfilter_max_generator = 512

    nfilter_discriminator = 64
    nfilter_max_discriminator = 512

    dvae = BetaVAE_H(
        c_dim=c_dim,
        nc=nc,
        infodistil_mode=True
    )
    generator = Generator(
        z_dim=z_dist_dim + c_dim,
        size=img_size,
        nfilter=nfilter_generator, 
        nfilter_max=nfilter_max_generator
    )
    discriminator = Discriminator(
        z_dim=z_dist_dim + c_dim,
        size=img_size,
        nfilter=nfilter_discriminator, 
        nfilter_max=nfilter_max_discriminator
    )

    dvae_ckpt = torch.load(os.path.join(project_dir, 'ckpt/500000')\
        , map_location=torch.device('cpu'))['model_states']['net']
    dvae.load_state_dict(dvae_ckpt)

    dvae = dvae.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001, alpha=0.99, eps=1e-8)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001, alpha=0.99, eps=1e-8)

    checkpoint_io = CheckpointIO(
        checkpoint_dir=os.path.join(project_dir, 'ckpt')
    )

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
    )

    cdist = get_zdist(c_dim, device=device)
    zdist = get_zdist(z_dist_dim, device=device)

    # Learning rate anneling
    g_scheduler = build_lr_scheduler(g_optimizer, last_epoch=-1)
    d_scheduler = build_lr_scheduler(d_optimizer, last_epoch=-1)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])

    train_dataset = datasets.ImageFolder(celeba_256_dir, transform)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=16,
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )

    trainer = Trainer(
        dvae, generator, discriminator, g_optimizer, d_optimizer,
        reg_param=10,
        w_info = 0.001
    )

    max_iter = 60000 # 300000
    pbar = tqdm(total=max_iter)
    it = -1
    epoch_idx = -1
    tstart = t0 = time.time()

    # it = epoch_idx = checkpoint_io.load(os.path.join(checkpoint_dir, 'model_00030000.pt'))

    out = False
    while not out:
        epoch_idx += 1
        tqdm.write('Start epoch %d...' % epoch_idx)

        for x_real, _ in train_loader:
            it += 1
            pbar.update(1)
            g_scheduler.step()
            d_scheduler.step()

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']

            x_real = x_real.to(device)

            # Discriminator updates
            z = zdist.sample((batch_size,))
            dloss, reg, cs = trainer.discriminator_trainstep(x_real, z)

            # Generators updates
            if ((it + 1) % d_steps) == 0:
                z = zdist.sample((batch_size,))
                gloss, encloss = trainer.generator_trainstep(z, cs)

            # (iii) Backup if necessary
            if ((it + 1) % backup_every) == 0:
                tqdm.write('Saving backup...')
                checkpoint_io.save(it, 'model_%08d.pt' % it)
                checkpoint_io.save(it, 'model.pt')

            # (iv) Save checkpoint if necessary
            if time.time() - t0 > save_every:
                tqdm.write('Saving checkpoint...')
                checkpoint_io.save(it, 'model.pt')
                t0 = time.time()

            if it >= max_iter:
                tqdm.write('Saving backup...')
                checkpoint_io.save(it, 'model_%08d.pt' % it)
                out = True
                break