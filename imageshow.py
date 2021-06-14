
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torch.nn as nn

import haarcascade_detect
import numpy as np
import argparse
import torch.nn.functional as F



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def eval(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            bs, ncrops, c, h, w = np.shape(images)
            images = images.view(-1, c, h, w)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %2f %%' % (100 * correct / total))







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--trained_model", default = "model_state.pth.tar", type= str,
				help = "Trained state_dict file path to open")
ap.add_argument("--input", type= str, default = "meto.png",help= "Input path for detect")
ap.add_argument("--output", type= str, default ="output.jpg",help = "Output path to save")
args = ap.parse_args()




classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
crop_size= 44
image_path = args.input
#Load model
trained_model = torch.load(args.trained_model)
print("Load weight model with {} epoch".format(trained_model["epoch"]))

model = VGG("VGG19")
model.load_state_dict(trained_model["model_weights"])
model.to(device)
model.eval()

transform_test = transforms.Compose([
		transforms.TenCrop(crop_size),
		transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
		])

def detect():

    original_image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', gray_image)
    faces = haarcascade_detect.face_detect(gray_image)

    if faces != []:
        for (x, y, w, h) in faces:
            roi = original_image[y:y+h, x:x+w]
            # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray=roi
            roi_gray = cv2.resize(roi_gray, (48, 48))

            roi_gray = Image.fromarray(np.uint8(roi_gray))
            inputs = transform_test(roi_gray)

            ncrops, c, ht, wt = np.shape(inputs)
            inputs = inputs.view(-1, c, ht, wt)
            inputs = inputs.to(device)
            # roi = original_image[y:y+h, x:x+w]
            # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # roi_gray = cv2.resize(roi_gray, (48, 48))

            # roi_gray = Image.fromarray(np.uint8(roi_gray))
            # inputs = transform_test(roi_gray)

            # ncrops, c, ht, wt = np.shape(inputs)
            # inputs = inputs.view(-1, c, ht, wt)
            # inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(ncrops, -1).mean(0)
            _, predicted = torch.max(outputs, 0)
            expression = classes[int(predicted.cpu().numpy())]

            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = "{}".format(expression)

            cv2.putText(original_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 100, 100), 2)
    
    cv2.imshow('imagee', original_image)
    # cv2.imwrite(args.output, original_image)
    cv2.imwrite(r"output\0.jpg", original_image)
if __name__ == '__main__':
    detect()
    print("Done!")