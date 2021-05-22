import cv2
import os
import numpy as np
import torch
import urllib.request as urlreq

FACIAL_LANDMARKS_5_IDXS = {
    'right_eye': (36, 39),
    'left_eye': (42, 45),
}

def to_img(tensor):
    return torch.squeeze(tensor.permute(1, 2, 0) + 0.5)

def load_fd(path='haarcascade_frontalface_alt2.xml'):
    if path is None or not os.path.exists(path):
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml'
        path = 'haarcascade_frontalface_alt2.xml'
        print('Downloading basic FD from {}'.format(url))
        urlreq.urlretrieve(url, path)
        print('File downloaded...')
    detector = cv2.CascadeClassifier(path)
    return detector

def load_lmd(path='lbfmodel.yaml'):
    if path is None or not os.path.exists(path):
        url = 'https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml'
        path = 'lbfmodel.yaml'
        print('Downloading basic LMD from {}'.format(url))
        urlreq.urlretrieve(url, path)
        print('File downloaded...')
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(path)
    return landmark_detector

def rotate_landmarks(landmarks, rot_center, angle):
    landmarks = np.array(landmarks)
    rot_mat = cv2.getRotationMatrix2D(tuple(rot_center), angle, 1.0)[:, :2]
    new_landmarks = []
    for point in landmarks:
        temp_point = (np.matmul(rot_mat, point - rot_center) + rot_center).tolist()
        new_landmarks.append(temp_point)
    return np.array(new_landmarks).astype(int)

def rotate_img(img, rot_center, angle):
    img_size = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(tuple(rot_center), angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img_size[::-1], flags=cv2.INTER_CUBIC)
    return result

def rotate_img_lms(img, landmarks):
    l_eye = landmarks[:2].mean(axis=0)
    r_eye = landmarks[2:].mean(axis=0)
    rot_center = (l_eye + r_eye) / 2
    angle = np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0])
    rot_img = rotate_img(img, np.array(rot_center), np.rad2deg(angle))
    rot_lms = rotate_landmarks(landmarks, np.array(rot_center), np.rad2deg(angle))
    return rot_img, rot_lms

def get_face_rect(img, landmarks, bbox, scale):
    height, width = img.shape[:2]
    l_eye = landmarks[:2].mean(axis=0)
    r_eye = landmarks[2:].mean(axis=0)
    center = (l_eye + r_eye) / 2
    cx, cy = center
    size_lim = min(cx, width - cx - 1, cy, height - cy - 1)
    h = min(size_lim * 2, int(bbox[-1] * scale))
    w = h
    return np.array([cx - w//2, cy - h//2, w, h]).astype('int')