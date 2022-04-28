import os
import time
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import resnet
from args import args


# define the main function
def main():
    global args,  best_prec1
    best_acc1, best_acc5 = 0.0, 0.0

    # set a seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # # check the save dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define the model
    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if args.cuda:
        model.cuda()
        cudnn.benchmark = True
    print(model.module)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found ar '{}'".format(args.resume))

    # preparing data
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.datadir, train=True,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, 4),
                             transforms.ToTensor(),
                             normalize,
                         ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.datadir, train=False,
                         transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize,
                          ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally choose a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    # evaluate model before training
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # train the model
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('current le{:.5e}'.format(optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion)

        # remember best prec@1 and save the checkpoint
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        print(' best Prec@1 {}, Prec@5 {}'
              .format(best_acc1, best_acc5))

    save_checkpoint({
        'epoch': epoch +1,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc1,
    }, filename = os.path.join(args.save_dir, 'model{}.th'.format(args.prune_rate)))


# define train function
def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            target = target.cuda()
            input = input.cuda()
        target_var = Variable(target)
        input_var = Variable(input)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        losses.update(loss.item(), n)
        top1.update(acc1.item(), n)
        top5.update(acc5.item(), n)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


# define validate function
def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        if args.cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, top5.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch):
    update_list = [150, 250, 320]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    prune_rate_all = [0.0, 0.5]
    best_acc_all = []
    for prun_rate in prune_rate_all:
        args.prune_rate = prun_rate
        print('current prune rate: {}'.format(args.prune_rate))
        main()