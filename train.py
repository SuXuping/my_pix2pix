import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MapDataset

import config
import Data_Augmentation
from generator import Generator
from discriminator import Discriminator

from utils import save_checkpoint, load_checkpoint, save_some_examples
from torchvision.utils import save_image
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def train_one_epoch(disc,gen,loader,opt_gen,opt_disc,l1_loss,bce,g_scaler,d_scaler):
    loop = tqdm(loader,leave=True)

    for index,(x,y) in enumerate(loop):
        x = x.to(config.DECIVE)
        y = y.to(config.DECIVE)

        # train discriminator
        with torch.cuda.amp.autocast():  ###自动混合精度训练
            y_fake = gen(x)
            D_real = disc(x,y)
            D_real_loss = bce(D_real,torch.ones_like(D_real))
            D_fake = disc(x,y_fake.detach())
            D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_fake_loss + D_real_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        ## train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x,y_fake)
            G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
            L1_loss = l1_loss(y_fake,y) * config.l1_lambda
            G_loss = G_fake_loss + L1_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if index % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def main():
    disc = Discriminator(in_channels=3).to(config.DECIVE)
    gen = Generator(in_channels=3, features=64).to(config.DECIVE)

    opt_disc = optim.Adam(disc.parameters(),lr=config.lr,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr=config.lr,betas=(0.5,0.999))

    BCE_LOSS = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.load_model:
        load_checkpoint(
            config.checkpoint_gen,gen,opt_gen,config.lr
        )
        load_checkpoint(
            config.checkpoint_disc,disc,opt_disc,config.lr
        )

    train_dataset = MapDataset(root_dir=config.train_dir)
    val_dataset = MapDataset(root_dir=config.val_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    g_scaler = torch.cuda.amp.GradScaler()  ##使用混合精度训练
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.num_epochs):
        train_one_epoch(disc,gen,train_loader,opt_gen,opt_disc,L1_LOSS,BCE_LOSS,g_scaler,d_scaler)

        if config.save_model and epoch % 5 == 0:
            save_checkpoint(gen,opt_gen,filename=config.checkpoint_gen)
            save_checkpoint(disc,opt_disc,filename=config.checkpoint_disc)

        save_some_examples(gen,val_loader,epoch,folder="evaluation")

if __name__ == "__main__":
    main()