import torch
from torchvision import transforms as tvt

import argparse
import os
import cv2
import numpy as np
import time
from glob import glob
from PIL import Image
from model import MNModel
from utils import load_fd, load_lmd, rotate_img_lms, get_face_rect

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to directory with images')
    return parser.parse_args()

def prepare_face(img, bbox):
    preproc = tvt.Compose([
        tvt.Grayscale(),
        tvt.ToTensor(),
        tvt.Resize(224),
        tvt.Normalize(mean=[0.5,], std=[1,]),
    ])
    x,y,w,h = bbox
    face = img[y:y+h, x:x+w]
    return preproc(Image.fromarray(face))

def eval_img(path, model, fd, lmd):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(img_gray)
    if len(faces) == 0:
        return -1, 0
    else:
        bbox = faces[0]
        _, landmarks = lmd.fit(img_gray, faces)
        lms = np.array(landmarks[0][0][[36,39,42,45]])

        img_gray, lms = rotate_img_lms(img_gray, lms)
        bbox = get_face_rect(img_gray, lms, faces[0], scale=2.0)

        inp = prepare_face(img_gray, bbox)
        start_time = time.time()
        score = torch.sigmoid(model(inp.unsqueeze(0).cuda()))
        end_time = time.time()
        diff = int((end_time - start_time)*1000)
        return 1 if score > 0.5 else 0, diff

def evaluate(args):
    print('Loading CNN model...')
    model = MNModel().cuda()
    model.load_state_dict(torch.load('hair-model.pth'))
    model.eval();
    fs = glob(os.path.join(args.input, '*'))
    print('Loading FD model...')
    fd = load_fd('haarcascade_frontalface_alt2.xml')
    print('Loading LMD model...')
    lmd = load_lmd('lbfmodel.yaml')
    res = []
    elapsed_times = []
    for f in fs:
        pred, elapsed_time = eval_img(f, model, fd, lmd)
        if pred != -1:
            elapsed_times.append(elapsed_time)
        res.append('{},{}'.format(f, pred))
    with open('./result.csv', 'w') as f:
        f.writelines('\n'.join(res))
    print('Done, output: ./result.csv')
    print('Avg processing time: {:.2f} ms'.format(np.mean(elapsed_times)))

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
