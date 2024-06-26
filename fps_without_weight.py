import argparse
import torch
import time
import logging
import sys
import os

from lib.models.ddrnet import ddrnet_23, ddrnet_silm
from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2
from lib.models.bisenetv1_global2taspp import BiSeNetV1_global2taspp
from lib.models.bisenetv1_global2taspp_ffm2fammul import BiSeNetV1_global2taspp_ffm2fammul



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='bisenetv1_global2taspp_ffm2fammul', help='model name')
parser.add_argument('--img_size', type=tuple,
                    default=(512, 512), help='input patch size of network input')
parser.add_argument('--channels', type=int,
                    default=3, help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
args = parser.parse_args()

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"]= "1"

    if args.model == 'ddrnet_silm':
        model = ddrnet_silm(args.num_classes)
    elif args.model == 'ddrnet':
        model = ddrnet_23(args.num_classes)
    elif args.model == 'bisenetv1':
        model = BiSeNetV1(args.num_classes)
    elif args.model == 'bisenetv1_global2taspp_noarm_ffmnoatten':
        model = BiSeNetV1_global2taspp(args.num_classes)
    elif args.model == 'bisenetv1_global2taspp':
        model = BiSeNetV1_global2taspp(args.num_classes)
    elif args.model == 'bisenetv1_global2taspp_ffm2fammul':
        model = BiSeNetV1_global2taspp_ffm2fammul(args.num_classes)



    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)

    else:
        model = None
        raise KeyError("unknown model: {}".format(args.model))

    model.to('cuda')

    logging.basicConfig(filename="fps_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    model.eval()
    all_time = 0
    data = torch.randn(1, args.channels, args.img_size[0], args.img_size[1]).cuda()
    for i in range(10000):
        if (i + 1) % 1000 == 0:
            print("{}/{}".format(i + 1, 10000))
        start_time = time.time()
        _ = model(data)
        end_time = time.time()
        all_time += (end_time - start_time)
    total_time = all_time / 10000
    fps = 1 / total_time
    logging.info("fps: {}".format(fps))
