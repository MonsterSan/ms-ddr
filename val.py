import argparse
import os
import random
import torch
import logging
import sys

from lib.datasets.dataset_crack import CrackDataset
from torch.utils.data import DataLoader

from lib.models.ddrnet import ddrnet_23, ddrnet_silm
from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2
from lib.models.bisenetv1_without_arm import BiSeNetV1_without_Arm
from lib.models.bisenetv1_without_ffm import BiSeNetV1_without_ffm
from lib.models.bisenetv1_with_aspp import BiSeNetV1_with_aspp
from lib.models.bisenetv1_with_dwaspp import BiSeNetV1_with_dwaspp

from torch.nn.modules.loss import CrossEntropyLoss
from lib.losses.ohem_cross_entropy_loss import OhemCrossEntropyLoss

from lib.utils.loss_avg_meter import LossAverageMeter
from lib.utils.confusion_matrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='bisenetv1_with_dwaspp', help='model name')
parser.add_argument('--log_path', type=str,
                    default='./run/bisenetv1_with_dwaspp_20240113_211031', help='log path')
parser.add_argument('--checkpoint_type', type=str,
                    default='best_miou', help="best_miou or last or min_loss")
# D:\\data\\Crack_Forest_paddle\\Crack_Forest_paddle
# /home/user/data/lumianliefeng/Crack_Forest_paddle
parser.add_argument('--dataset_root', type=str,
                    default='/home/user/data/lumianliefeng/Crack_Forest_paddle', help='dataset root directory')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == '__main__':
    # create dataloader
    val_dataset = CrackDataset(args.dataset_root, args.img_size, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    data_length = len(val_dataset)

    # create model
    if args.model == 'ddrnet':
        model = ddrnet_silm(args.num_classes)
        losses = [OhemCrossEntropyLoss()]
        loss_weights = [1]
    elif args.model == 'bisenetv1':
        model = BiSeNetV1(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
    elif args.model == 'bisenetv1_without_arm':
        model = BiSeNetV1_without_Arm(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
    elif args.model == 'bisenetv1_without_ffm':
        model = BiSeNetV1_without_ffm(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
    elif args.model == 'bisenetv1_with_aspp':
        model = BiSeNetV1_with_aspp(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
    elif args.model == 'bisenetv1_with_dwaspp':
        model = BiSeNetV1_with_dwaspp(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)
        losses = [CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss()]
        loss_weights = [1, 1, 1, 1, 1]
    else:
        model = None
        raise KeyError("unknown model: {}".format(args.model))

    if args.checkpoint_type == 'best_miou':
        model_type = "best_miou.pth"
    elif args.checkpoint_type == 'last':
        model_type = "lastest.pth"
    elif args.checkpoint_type == 'min_loss':
        model_type = "lowest_loss.path"
    else:
        raise KeyError("checkpoint_type should be 'best_miou' or 'last' or 'min_loss'")
    model.to('cuda')
    weight_path = os.path.join(args.log_path, "weights/" + model_type)
    model_state_dict = torch.load(weight_path)
    model.load_state_dict(model_state_dict)

    logging.basicConfig(filename=args.log_path + "/val_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("val img nums {}".format(data_length))
    # val
    val_losses = LossAverageMeter()
    val_confmat = ConfusionMatrix(args.num_classes)
    model.eval()
    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(val_dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            output = model(image_batch)
            if isinstance(output, tuple):
                output = output[0]
            val_confmat.update(label_batch.flatten(), output.argmax(1).flatten())
            loss_type = losses[0]
            loss = loss_type(output, label_batch.long())
            val_losses.update(loss.item(), image_batch.shape[0])

    valacc_global, valacc, valiu, valRec, valPre, valF1 = val_confmat.compute()
    valacc_global = valacc_global.item() * 100
    valaver_row_correct = ['{:.2f}'.format(i) for i in (valacc * 100).tolist()]
    valiou = ['{:.2f}'.format(i) for i in (valiu * 100).tolist()]
    valmiou = valiu.mean().item() * 100
    valF1 = valF1.item() * 100,
    valRec = valRec.item() * 100,
    valPre = valPre.item() * 100
    logging.info('\n'+"losses.avg:" + str(val_losses.avg) + "\n" +
                 "miou:" + str(valmiou) + "\n" +
                 "acc_global:" + str(valacc_global) + "\n" +
                 "aver_row_correct:" + str(valaver_row_correct[0]) + "-" + str(valaver_row_correct[1]) + "\n" +
                 "iou:" + str(valiou[0]) + "-" + str(valiou[1]) + "\n" +
                 "f1:" + str(valF1) + "\n" +
                 "Rec:" + str(valRec) + "\n" +
                 "Rre:" + str(valPre) + "\n")
    logging.info("Val Finished")
