import torch
import torch.nn as nn
import cv2


def pred_convert(pred):
    img_list = []
    for i in range(pred.shape[0]):
        img = pred[i]
        img = img.argmax(0)
        img = img.unsqueeze(0)
        img_list.append(img)
    res = torch.cat(img_list, dim=0)
    return res


def canny(tensors):
    list = []
    for tensor in tensors:
        numpy_image = tensor.cpu().numpy()
        edges = cv2.Canny((numpy_image * 255).astype('uint8'), 100, 200)
        edges_tensor = torch.from_numpy(edges / 255.0).float()
        edges_tensor = edges_tensor.unsqueeze(0)
        list.append(edges_tensor)
    res = torch.cat(list, dim=0)
    return res


class CannyLoss(nn.Module):
    def __init__(self, reduction=False):
        super(CannyLoss, self).__init__()
        if reduction:
            self.loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        pred = pred_convert(pred)
        pred = canny(pred)
        labels = canny(labels)
        return self.loss(pred, labels)
