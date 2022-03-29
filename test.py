""" This program executes the test function on test dataset """

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from trainer import Trainer
from dataset.dbpedia import DBpediaDataset 
from config import *
from utils import accuracy, AverageMeter
from models.classifier import initilize_model
import mlflow

import shutil
import time
import random
import numpy as np
import copy
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


""" Define argparser to read the command line arguments """
parser = argparse.ArgumentParser(description='multimodal fusion')
parser.add_argument('--dataroot', default='/home/vision/BMW_task/hmtc', help='root directory of dataset')
parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--seq_len', type=int, default=50, help='Video length')
parser.add_argument('--num_classes', type=int, default=219, help='Number of categories')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Number of workers')
parser.add_argument('--embed_dim', type=int, default=300, help='Embed dimensions')
parser.add_argument('--pretrain', type=bool, default=False, help='Load pretrained weights')


def main():
    """ Begin the execution """
    args = parser.parse_args()

    model = initilize_model(vocab_size=577154,
                                      embed_dim=300,
                                      learning_rate=0.25,
                                      dropout=0.5)
    model.load_state_dict(torch.load('save_model/dbpedia_best.pt'))
    model.cuda()
    #summary(model, [(1,3,16,112,112),(1,3,16,112,112)])

    """ define loss function (criterion) and optimizer """
    criterion = nn.CrossEntropyLoss().cuda()                         
    cudnn.benchmark = True

    """ Prepare and load the dataset """                                                                                                                                                                    
    test_set = DBpediaDataset(args.dataroot, np_test, np_tst_lbls, transform=None)                                                                                                                                                                   
    test_loader =  torch.utils.data.DataLoader(test_set, batch_size=64,
                                               shuffle=True, num_workers=args.nworkers)

    start_time= time.time()
    epoch=0

    """ Call to the test function """   
    prec1 = test(test_loader, model, criterion, epoch)
    print('Top Precision:',prec1)

    """ Compute the execution time """       
    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)


def test(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_freq = 10
    use_gpu = torch.cuda.is_available()

    """ set to validation mode """
    model.eval()

    end = time.time()
    with torch.no_grad():
      for i, (input_text, target) in enumerate(test_loader):
        
          if use_gpu:
               input_text = torch.autograd.Variable(input_text.long().cuda())
               target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))

          else:
               input_text = torch.autograd.Variable(input_text.long())
               target = torch.autograd.Variable(target.long())

          """ compute output """
          output = model(input_text)
          loss = criterion(output, target)

          """ measure accuracy and record loss """
          prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
          losses.update(loss.data, input_text.size(0))
          top1.update(prec1, input_text.size(0))
          top5.update(prec5, input_text.size(0))

          """ measure elapsed time """
          batch_time.update(time.time() - end)
          end = time.time()

          if i % print_freq == 0:
              print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     i, len(test_loader), batch_time=batch_time, loss=losses,
                     top1=top1, top5=top5))


      print(' Test:{0} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(epoch, top1=top1, top5=top5))

      """ write results to the log """
      results = open('logs/valid_log.txt', 'a')
      results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
            .format(epoch, loss=losses, top1=top1, top5=top5))
      results.close()

      return top1.avg



if __name__ == '__main__':
   main()
