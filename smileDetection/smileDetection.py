import numpy as np
import dlib
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=1250, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=2),
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        x = F.softmax(x, dim=1)
        return x


def image_read(path):
    detector = dlib.get_frontal_face_detector()
    p = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p)
    image = cv2.imread(path)

    face = detector(image)[0]

    landmarks = predictor(image, face)
    left_x = landmarks.part(48).x
    left_y = landmarks.part(50).y

    right_x = landmarks.part(54).x
    right_y = landmarks.part(56).y

    w = np.abs(right_x - left_x)
    h = np.abs(right_y - left_y)

    padding_w = np.round(w * 0.4)
    padding_h = np.round(h * 0.9)

    left_x -= padding_w
    right_x += padding_w
    left_y -= padding_h
    right_y += padding_h

    cropped_lip = image[int(left_y):int(right_y), int(left_x):int(right_x)]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    image = transform(Image.fromarray(cropped_lip))
    image = torch.unsqueeze(image, dim=0)

    return image


def smileDetection(path):
    model = LeNet5()
    model.load_state_dict(torch.load('smileDetection.pt'))
    model.eval()

    image = image_read(path)
    pred = model(image)
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred = torch.max(pred[0], 0)[1].item()

    if pred == 0:
        return 'no_smile'
    else:
        return 'smile'

print(smileDetection('D:\\yolo\\smileDetection\\img_align_celeba\\000001.jpg'))