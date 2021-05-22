import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms as tvt

import argparse
import os
import shutil as sh
import cfg
from model import MNModel

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=cfg.NUM_EPOCHS, help='Number of epochs to train')
    return parser.parse_args()

def load_data():
    train_transform = tvt.Compose([
        tvt.RandomHorizontalFlip(),
        tvt.Grayscale(),
        tvt.ColorJitter(brightness=cfg.BRIGHTNESS, contrast=cfg.CONTRAST),
        tvt.RandomCrop(cfg.CROPSIZE),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.5,], std=[1,]),
    ])
    val_transform = tvt.Compose([
        tvt.Grayscale(),
        tvt.CenterCrop(cfg.CROPSIZE),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.5,], std=[1,]),
    ])
    train_ds = ImageFolder(root=cfg.TRAIN_DIR, transform=train_transform)
    val_ds = ImageFolder(root=cfg.VAL_DIR, transform=val_transform)
    train_dl = data.DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl = data.DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    return train_dl, val_dl

def get_model_num(models_dir):
    models_exist = os.listdir(models_dir)
    if models_exist:
        model_num = str(int(sorted(models_exist)[-1]) + 1).zfill(3)
    else:
        model_num = '001'
    return model_num

def train(args):
    train_dl, val_dl = load_data()

    models_dir = cfg.MODELS_DIR
    model_num = get_model_num(models_dir)
    model_dir = os.path.join(models_dir, model_num)
    meta_dir = os.path.join(model_dir, 'metainfo')
    os.makedirs(model_dir)
    os.makedirs(meta_dir)
    sh.copy('cfg.py', meta_dir)
    sh.copy('model.py', meta_dir)
    ckpt_template = '{:03d}_train_{:.3f}_val_{:.3f}.pth'
    print('Training model {}, path to checkpoints: {}'.format(model_num, model_dir))

    model = MNModel().cuda()
    lossfunc = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.STEPSIZE, cfg.GAMMA)
    for epoch in range(args.epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0

        for inputs, labels in train_dl:
            inputs = inputs.cuda()
            labels = labels.cuda().type(torch.float)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = lossfunc(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for inputs, labels in val_dl:
                inputs = inputs.cuda()
                labels = labels.cuda().type(torch.float)

                outputs = model(inputs)
                loss = lossfunc(outputs, labels)
                val_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        print('[{}] train loss: {:.3f}, val loss: {:.3f}'.format(epoch + 1, train_loss, val_loss))
        torch.save(model.state_dict(),
                   os.path.join(model_dir, ckpt_template.format(epoch + 1, train_loss, val_loss))
                  )

    print('Finished Training')

if __name__ == '__main__':
    args = parse_args()
    train(args)