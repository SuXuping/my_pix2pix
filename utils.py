import torch
import config
from torchvision.utils import save_image

def save_some_examples(generator,val_loader,epoch,folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DECIVE),y.to(config.DECIVE)

    generator.eval()
    with torch.no_grad():
        y_fake = generator(x)
        y_fake = y_fake * 0.5 + 0.5  #去除normalization
        save_image(y_fake,folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5,folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        generator.train()

def save_checkpoint(model,optimizer,filename="my_checkpoint.pth"):
    print("=> saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> loading checkpoint")
    checkpoint = torch.load(checkpoint_file,map_location=config.DECIVE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
