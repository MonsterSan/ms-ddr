import argparse
import torch
import time
import os
import logging
import sys

from lib.models.ddrnet import ddrnet_23, ddrnet_silm
from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2
from lib.models.bisenetv1_without_arm import BiSeNetV1_without_Arm
from lib.models.bisenetv1_without_ffm import BiSeNetV1_without_ffm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='bisenetv1_without_ffm', help='model name')
parser.add_argument('--log_path', type=str,
                    default='./run/bisenetv1_without_ffm_20240111_215947', help='log path')
parser.add_argument('--checkpoint_type', type=str,
                    default='best_miou', help="best_miou or last or min_loss")
parser.add_argument('--img_size', type=tuple,
                    default=(512, 512), help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
args = parser.parse_args()

if __name__ == '__main__':
    if args.model == 'ddrnet_silm':
        model = ddrnet_silm(args.num_classes)
    elif args.model == 'ddrnet':
        model = ddrnet_23(args.num_classes)
    elif args.model == 'bisenetv1':
        model = BiSeNetV1(args.num_classes)
    elif args.model == 'bisenetv1_without_arm':
        model = BiSeNetV1_without_Arm(args.num_classes)
    elif args.model == 'bisenetv1_without_ffm':
        model = BiSeNetV1_without_ffm(args.num_classes)
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)
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

    logging.basicConfig(filename=args.log_path + "/fps_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    model.eval()
    all_time = 0
    data = torch.randn(1, 3, args.img_size[0], args.img_size[1]).cuda()
    for i in range(10000):
        print("{}/{}".format(i + 1, 10000))
        start_time = time.time()
        _ = model(data)
        end_time = time.time()
        all_time += (end_time - start_time)
    total_time = all_time / 10000
    fps = 1 / total_time
    logging.info("fps: {}".format(fps))
