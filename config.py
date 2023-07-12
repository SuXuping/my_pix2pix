import torch

DECIVE = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "data/train"
val_dir = "data/val"
lr = 2e-4
batch_size = 4
num_workers = 2
num_epochs = 500
l1_lambda = 100
lambda_gp = 10
image_size = 256
image_channel = 3

load_model = False
save_model = False
checkpoint_disc = "disc.pth"
checkpoint_gen = "gen.pth"
