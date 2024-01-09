import argparse
import logging
import os
import datetime
import sys
import torch
import time
import random
import numpy as np
import math

from lib.datasets.dataset_crack import CrackDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR

from lib.models.ddrnet import ddrnet_23, ddrnet_silm
from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2

from torch.nn.modules.loss import CrossEntropyLoss
from lib.losses.ohem_cross_entropy_loss import OhemCrossEntropyLoss

from lib.utils.loss_avg_meter import LossAverageMeter
from lib.utils.confusion_matrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='ddrnet', help='model name')
parser.add_argument('--dataset_root', type=str,
                    default='/home/user/data/lumianliefeng/Crack_Forest_paddle', help='dataset root directory')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,
                    default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
parser.add_argument('--log_path', type=str,
                    default='./run', help='run path')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == "__main__":
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create dataloader
    train_dataset = CrackDataset(args.dataset_root, args.img_size, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    data_length = len(train_dataset)
    max_iterations = data_length // args.batch_size * args.max_epochs

    val_dataset = CrackDataset(args.dataset_root, args.img_size, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # create model
    if args.model == 'ddrnet':
        model = ddrnet_silm(args.num_classes)
        losses = [OhemCrossEntropyLoss()]
        loss_weights = [1]
    elif args.model == 'bisenetv1':
        model = BiSeNetV1(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1]
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)
        losses = [CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss()]
        loss_weights = [1, 1, 1, 1, 1]
    else:
        model = None
        raise KeyError("unknown model: {}".format(args.model))
    model.to('cuda')

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    lr = PolynomialLR(optimizer=optimizer, total_iters=args.max_epochs * data_length, power=0.9)

    # config log
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_path = os.path.join(args.log_path, args.model + '_' + now_time)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(filename=args.log_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("model {}".format(args.model))
    logging.info("train img nums {}".format(data_length))
    logging.info("{} iterations per epoch. {} max iterations "
                 .format(data_length // args.batch_size, data_length // args.batch_size * args.max_epochs))

    trainlog_path = os.path.join(args.log_path, 'train.txt')
    vallog_path = os.path.join(args.log_path, 'val.txt')

    # metric
    iter_num = 0
    min_loss = 100
    best_miou = 0

    # train
    start_time = time.time()
    for epoch in range(args.max_epochs):
        model.train()
        train_losses = LossAverageMeter()
        train_confmat = ConfusionMatrix(args.num_classes)
        for batch_idx, sampled_batch in enumerate(train_dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            if not isinstance(outputs, tuple):
                outputs = [outputs]
            train_confmat.update(label_batch.flatten(), outputs[0].argmax(1).flatten())
            loss_list = [loss_weights[i] * losses[i](outputs[i], label_batch.long()) for i in range(len(outputs))]
            #print(loss_list)
            main_loss = loss_list[0]
            loss = main_loss.clone()
            for i in range(1, len(loss_list)):
                loss += loss_list[i]
            train_losses.update(main_loss.item(), image_batch.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr.step()

            iter_num += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            progress = iter_num / max_iterations
            remaining_time = elapsed_time * (1 / progress - 1)
            remaining_time_minutes = math.floor(remaining_time / 60)
            remaining_time_seconds = math.floor(remaining_time - remaining_time_minutes * 60)

            if iter_num % 500 == 0:
                logging.info('[train]iteration %d / %d\tloss: %f\tmain_loss:%f\tremaining time : %d minutes  %d seconds'
                             % (iter_num, max_iterations, loss.item(), main_loss.item(), remaining_time_minutes,
                                remaining_time_seconds))

        trainacc_global, trainacc, trainiu, trainRec, trainPre, trainF1 = train_confmat.compute()
        trainacc_global = trainacc_global.item() * 100
        trainaver_row_correct = ['{:.1f}'.format(i) for i in (trainacc * 100).tolist()]
        trainiou = ['{:.1f}'.format(i) for i in (trainiu * 100).tolist()]
        trainmiou = trainiu.mean().item() * 100
        trainF1 = trainF1.item() * 100,
        trainRec = trainRec.item() * 100,
        trainPre = trainPre.item() * 100
        with open(trainlog_path, "a") as lpath:
            lpath.write(str(epoch) + "\t" + str(train_losses.avg) + "\t" + str(trainmiou) + "\t" + str(
                trainacc_global) + "\t" + str(trainaver_row_correct[0]) + "-" + str(
                trainaver_row_correct[1]) + "\t" + str(trainiou[0]) + "-" + str(trainiou[1]) + "\t" + str(
                trainF1) + "\t" + str(trainRec) + "\t" + str(trainPre) + "\n")
        if epoch >= 10:
            torch.save(model.state_dict(), os.path.join(args.log_path, 'latest.pth'))

        logging.info("evaluating")
        model.eval()
        val_losses = LossAverageMeter()
        val_confmat = ConfusionMatrix(args.num_classes)
        with torch.no_grad():
            sum_main_loss = 0
            sum_loss = 0
            num = 0
            for batch_idx, sampled_batch in enumerate(val_dataloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                if not isinstance(outputs, tuple):
                    outputs = [outputs]

                val_confmat.update(label_batch.flatten(), outputs[0].argmax(1).flatten())
                loss_list = [loss_weights[i] * losses[i](outputs[i], label_batch.long()) for i in range(len(outputs))]
                main_loss = loss_list[0]
                loss = main_loss
                for i in range(1, len(loss_list)):
                    loss += loss_list[i]
                val_losses.update(main_loss.item(), image_batch.shape[0])
                sum_main_loss += main_loss.item()
                sum_loss += loss.item()
                num += 1

            current_time = time.time()
            elapsed_time = current_time - start_time
            progress = iter_num / max_iterations
            remaining_time = elapsed_time * (1 / progress - 1)
            remaining_time_minutes = math.floor(remaining_time / 60)
            remaining_time_seconds = math.floor(remaining_time - remaining_time_minutes * 60)
            logging.info('[ val ]iteration %d\tmain_loss : %f, loss: %f\tremaining time : %d minutes  %d seconds'
                         % (
                         iter_num, sum_main_loss / num, sum_loss / num, remaining_time_minutes, remaining_time_seconds))
        valacc_global, valacc, valiu, valRec, valPre, valF1 = val_confmat.compute()
        valacc_global = valacc_global.item() * 100
        valaver_row_correct = ['{:.1f}'.format(i) for i in (valacc * 100).tolist()]
        valiou = ['{:.1f}'.format(i) for i in (valiu * 100).tolist()]
        valmiou = valiu.mean().item() * 100
        valF1 = valF1.item() * 100,
        valRec = valRec.item() * 100,
        valPre = valPre.item() * 100
        with open(vallog_path, "a") as lpath:
            lpath.write(
                str(epoch) + "\t" + str(val_losses.avg) + "\t" + str(valmiou) + "\t" + str(valacc_global) + "\t" + str(
                    valaver_row_correct[0]) + "-" + str(valaver_row_correct[1]) + "\t" + str(valiou[0]) + "-" + str(
                    valiou[1]) + "\t" + str(valF1) + "\t" + str(valRec) + "\t" + str(valPre) + "\n")

        if val_losses.avg < min_loss:
            min_loss = val_losses.avg
            torch.save(model.state_dict(), os.path.join(args.log_path, 'lowest_loss.pth'))
        if valmiou > best_miou:
            best_miou = valmiou
            torch.save(model.state_dict(), os.path.join(args.log_path, 'best_miou.pth'))
    new_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print("Training Finished at " + new_time_str)
