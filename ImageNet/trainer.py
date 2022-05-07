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
import resnet
from args import args
import shutil
import gc


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

    input_size = 224

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # alternative adam optimizer
    # =============================================================================
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)
    # =============================================================================

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            train_acc_1 = checkpoint['train_acc_1']
            train_acc_5 = checkpoint['train_acc_5']
            val_acc_1 = checkpoint['val_acc_1']
            val_acc_5 = checkpoint['val_acc_5']
            optimizer.load_state_dict(checkpoint['optimizer'])
            # =============================================================================
            # # if can not run, please use the following code
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.cuda()
            # =============================================================================
            print("=> loaded checkpoint '{}' (epoch {}) acc1{} acc5{}"
                  .format(args.resume, checkpoint['epoch'], val_acc_1, val_acc_5))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    print('==> Using Pytorch Dataset')
    import torchvision
    import accimage
    traindir = os.path.join(args.data, 'train') 
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    torchvision.set_image_backend('accimage')

    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    train_acc_1 = []
    val_acc_1 = []
    train_acc_5 = []
    val_acc_5 = []

    # evaluate pretrained model before fine-tuning/training
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        # train for one epoch
        train_prec1, train_prec5 = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        test_prec1, test_prec5 = validate(val_loader, model, criterion)
        val_acc_1.append(test_prec1)
        val_acc_5.append(test_prec5)

        # save the checkpoint
        if (epoch + 1) % args.checkpoint == 0:
            save_file_path = os.path.join(args.result_path,'imagenetmodel.pth')
            states = {
                'epoch': epoch + 1,  
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'train_acc_1': train_acc_1,
                'train_acc_5': train_acc_5,                
                'val_acc_1': val_acc_1,
                'val_acc_5': val_acc_5,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)        


def train(train_loader, model, criterion, optimizer, epoch):
    # batch_time = AverageMeter()
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
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5)) 
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        gc.collect()
        
    return top1.avg, top5.avg        
        
        
def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_top1, best_top5

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            target = target.cuda()
            input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    if top1.avg > best_top1:
        best_top1 = top1.avg
        
    if top5.avg > best_top5:
        best_top5 = top5.avg        

    print('Best Prec@1: {:.2f}%\n'.format(best_top1))
    print('Best Prec@5: {:.2f}%\n'.format(best_top5))
    
    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        

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
    update_list = [30, 60, 80, 95]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.cuda().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
