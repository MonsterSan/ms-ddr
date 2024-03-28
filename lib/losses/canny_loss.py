import torch
import torch.nn as nn
import cv2




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
    def __init__(self, output=False):
        super(CannyLoss, self).__init__()
        self.output = output
        self.out_loss = nn.CrossEntropyLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, pred, labels):
        # 8 512 512
        edge_mask = canny(labels).to(torch.int64).to('cuda')
        if self.output:
            loss = self.out_loss(pred, labels.long())
            edge_mask = edge_mask.detach()
            out_loss = (loss * edge_mask).mean()
            return out_loss
        else:
            loss = self.ce_loss(pred, edge_mask)
            return loss
