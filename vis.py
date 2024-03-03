import argparse
import os
import random
import torch
import logging
import sys
from PIL import Image
from torchvision import transforms

from lib.models.ddrnet import ddrnet_silm
from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2
from lib.models.bisenetv1_noarm import BiSeNetV1_noarm
from lib.models.bisenetv1_noarmglobal import BiSeNetV1_noarmglobal
from lib.models.bisenetv1_noarm_global2aspp import BiSeNetV1_noarm_global2aspp
from lib.models.bisenetv1_noarm_global2taspp import BiSeNetV1_noarm_global2taspp
from lib.models.bisenetv1_noarm_global2taspp_ffm2mix import BiSeNetV1_noarm_global2taspp_ffm2mix
from lib.models.bisenetv1_noarm_global2taspp_ffm2mix_v2 import BiSeNetV1_noarm_global2taspp_ffm2mix_v2
from lib.models.bisenetv1_ffm2mix import BiSeNetV1_ffm2mix
from lib.models.crackformer import crackformer

from torch.nn.modules.loss import CrossEntropyLoss
from lib.losses.ohem_cross_entropy_loss import OhemCrossEntropyLoss

from lib.utils.loss_avg_meter import LossAverageMeter
from lib.utils.confusion_matrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='crackformer', help='model name')
parser.add_argument('--img_path',type=str,
                    default="./images/2195.jpg",help='image path')
parser.add_argument('--log_path', type=str,
                    default='./run/crackformer_20240302_092120', help='log path')
parser.add_argument('--checkpoint_type', type=str,
                    default='best_miou', help="best_miou or last or min_loss")
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
    # create model
    if args.model == 'ddrnet':
        model = ddrnet_silm(args.num_classes)
        losses = [OhemCrossEntropyLoss()]
        loss_weights = [1]
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)
        losses = [CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss()]
        loss_weights = [1, 1, 1, 1, 1]
    elif args.model == 'crackformer':
        model = crackformer(2)
        losses = [CrossEntropyLoss()]
        loss_weights = [1]
    elif 'bisenetv1' in args.model:
        if args.model == 'bisenetv1':
            model = BiSeNetV1(args.num_classes)
        elif args.model == 'bisenetv1_ffm2mix':
            model = BiSeNetV1_ffm2mix(args.num_classes)
        elif args.model == 'bisenetv1_noarm':
            model = BiSeNetV1_noarm(args.num_classes)
        elif args.model == 'bisenetv1_noarmglobal':
            model = BiSeNetV1_noarmglobal(args.num_classes)
        elif args.model == 'bisenetv1_noarm_global2taspp':
            model = BiSeNetV1_noarm_global2taspp(args.num_classes)
        elif args.model == 'bisenetv1_noarm_global2taspp_ffm2mix':
            model = BiSeNetV1_noarm_global2taspp_ffm2mix(args.num_classes)
        elif args.model == 'bisenetv1_noarm_global2taspp_ffm2mix_v2':
            model = BiSeNetV1_noarm_global2taspp_ffm2mix_v2(args.num_classes)
        elif args.model == 'bisenetv1_noarm_global2aspp':
            model = BiSeNetV1_noarm_global2aspp(args.num_classes)
        else:
            raise KeyError("unknown model: {}".format(args.model))

        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
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

    logging.basicConfig(filename=args.log_path + "/vis_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # img find
    name = os.path.basename(args.img_path)
    name = name.replace(".jpg", "")
    img = Image.open(args.img_path)
    if img.size != (512,512):
        img = img.resize((512,512))
    data = img.convert("RGB")
    trans = transforms.ToTensor()
    data = trans(data)
    log_path = args.log_path
    new_png_name = name + "_vis.png"

    # vis
    model.eval()
    with torch.no_grad():
        image = data.cuda()
        image = image.unsqueeze(0)
        output = model(image)
        output = output.squeeze(0)
        output = output.view(2,512,512).argmax(0)
        output = torch.where(output == 1, 255, output)
        output = output.to(torch.uint8)
        print(output.shape)
        output = output.to("cpu")
        crack = output.numpy()
        out_img = Image.fromarray(crack)
        out_img.save(os.path.join(log_path,new_png_name))
    logging.info("vis Finished")
