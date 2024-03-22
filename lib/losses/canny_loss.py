import torch
import torch.nn as nn
import cv2
import numpy as np


def canny(images):
    for i in range(images.shape[0]):
        image = images[i, 0, :, :]
        image = image // 0.5000001 * 255   # 二值化
        image_2 = image.cpu().detach().numpy()
        image_2 = image_2.astype(np.uint8)
        img = cv2.Canny(image_2, 30, 150)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img.type = torch.float32
        if i != 0:
            if i != 1:
                img_final = torch.cat((img_final, img), 0)
            else:
                img_final = torch.cat((img_first, img), 0)
        else:
            img_first = img
    return img_final / 255

class CannyLoss(nn.Module):
    def __init__(self):
        super(CannyLoss, self).__init__()
        self.loss = nn.BCELoss(size_average=True)

    def forward(self, pred, labels):
        pred = canny(pred)
        labels = canny(labels)
        return self.loss(pred, labels)




