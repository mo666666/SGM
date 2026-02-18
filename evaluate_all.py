from __future__ import print_function, division, absolute_import
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import timm

import pretrainedmodels
import pretrainedmodels.utils

from utils_data import SubsetImageNet

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--input_dir', metavar='DIR', default="path_to_dataset",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnetamobile',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.add_argument('--default', action='store_true',
                    help="Evaluation ends")
parser.set_defaults(preserve_aspect_ratio=True)



def main(arch, args):


    # create model（default 模式下用当前循环的 arch，否则用 args.arch）
    if arch in ["convit_base","tnt_s_patch16_224","visformer_small"]:
        model = timm.create_model(arch, pretrained=True)
    elif args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        model = pretrainedmodels.__dict__[arch](num_classes=1000,
                                               pretrained=args.pretrained)
    else:
        model = pretrainedmodels.__dict__[arch]()




    # Data loading code
    valdir = os.path.join(args.input_dir)

    scale = 1.0
    if arch =="pdarts" or  arch == "convit_base" or arch == "tnt_s_patch16_224" or arch == "visformer_small":
        net = pretrainedmodels.__dict__["resnet18"](num_classes=1000,
                                                         pretrained=args.pretrained)
        val_tf = pretrainedmodels.utils.TransformImage(
            net,
            scale=scale,
            preserve_aspect_ratio=args.preserve_aspect_ratio
        )
    else:
        val_tf = pretrainedmodels.utils.TransformImage(
            model,
            scale=scale,
            preserve_aspect_ratio=args.preserve_aspect_ratio
        )
        model.cuda()

    val_set = SubsetImageNet(root=valdir, transform=val_tf)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    top1_avg = validate(val_loader, model, criterion)

    del model,val_loader,criterion,val_set

    return top1_avg



def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        device = torch.device("cuda:0")

        end = time.time()
        for i, raw_data in enumerate(val_loader):
            input = raw_data[0]
            target = raw_data[1]
            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        print('* ASR {:.3f}'.format(100-top1.avg))

        return 100-top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    if args.default:
        arch_list = ["vgg16_bn","vgg19_bn","resnet152","densenet201","senet154","convit_base","tnt_s_patch16_224","visformer_small","inceptionv4","inceptionresnetv2"]
    else:
        arch_list = [args.arch,]
    top1_avg_list = list()

    for arch in arch_list:
        top1_avg_list.append(main(arch, args))

    # 将 top1_avg_list 结果写入 CSV（保存在 input_dir 下）
    csv_path = os.path.join(args.input_dir, 'top1_avg_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'top1_avg'])
        for arch, top1_avg in zip(arch_list, top1_avg_list):
            writer.writerow([arch, top1_avg])
    print('Results saved to {}'.format(csv_path))


