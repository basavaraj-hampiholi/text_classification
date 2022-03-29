import torch
import torch.nn as nn
import time
from utils import accuracy, AverageMeter

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Trainer:
	""" This class contains the methods for training and validating the fusion model """

	def __init__(self, fuse_model, criterion, optimizer, mlflow):
	    self.batch_time = AverageMeter()
	    self.data_time = AverageMeter()
	    self.losses = AverageMeter()
	    self.top1 = AverageMeter()
	    self.top5 = AverageMeter()

	    self.model = fuse_model
	    self.loss_ce = criterion
	    self.optim = optimizer
	    self.print_freq = 10
	    self.use_gpu = torch.cuda.is_available()
	    self.mlflow = mlflow

	def train(self, train_loader, epoch):
	    """ This function trains the model with training set of data """

	    """ set to training mode """
	    self.model.train()

	    end = time.time()
	    for i, (input_text, target) in enumerate(train_loader):
	        #print(input_text.size(), target.size())
	        #exit(1)

	        self.data_time.update(time.time() - end)

	        if self.use_gpu:
	           input_text = torch.autograd.Variable(input_text.long().cuda())
	           target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))

	        else:
	           input_text = torch.autograd.Variable(input_text.long())
	           target = torch.autograd.Variable(target.long())

	        """ compute output """
	        output = self.model(input_text)
	        loss = self.loss_ce(output, target)

	        """ measure accuracy and record loss """
	        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
	        self.losses.update(loss.data, input_text.size(0))
	        self.top1.update(prec1, input_text.size(0))
	        self.top5.update(prec5, input_text.size(0))



	        """ compute gradient and do SGD step """
	        self.optim.zero_grad()
	        loss.backward()
	        self.optim.step()

	        """ measure elapsed time """
	        self.batch_time.update(time.time() - end)
	        end = time.time()


	        if i % self.print_freq == 0:
	            print('Epoch: [{0}][{1}/{2}]\t'
	                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
	                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
	                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
	                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
	                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
	                   epoch, i, len(train_loader), batch_time=self.batch_time,
	                   data_time=self.data_time, loss=self.losses, top1=self.top1, top5=self.top5))

	            self.mlflow.log_metric("Train Prec1", self.top1.avg.item())
	            self.mlflow.log_metric("Train Prec5", self.top5.avg.item())
	            self.mlflow.log_metric("Train Loss", self.losses.avg.item())

	    results = open('logs/train_log.txt', 'a')
	    results.write('Epoch: [{0}][{1}/{2}]\t'
	          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
	          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
	          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
	           epoch, i, len(train_loader), loss=self.losses,
	           top1=self.top1, top5=self.top5))
	    results.close()


	def validate(self, val_loader, epoch):
	    """This function validates the model validation set of data"""

	    """ set to validation mode """
	    self.model.eval()

	    end = time.time()
	    with torch.no_grad():
	      for i, (input_text, target) in enumerate(val_loader):
	        
	          if self.use_gpu:
	             input_text = torch.autograd.Variable(input_text.long().cuda())
	             target = torch.autograd.Variable(target.long().cuda())
	          else:
	             input_text = torch.autograd.Variable(input_text.long())
	             target = torch.autograd.Variable(target.long())

	          """ call the model and compute output """
	          output = self.model(input_text)
	          loss = self.loss_ce(output, target)

	          """ measure accuracy and record loss """
	          prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
	          self.losses.update(loss.data, input_text.size(0))
	          self.top1.update(prec1, input_text.size(0))
	          self.top5.update(prec5, input_text.size(0))

	          """ measure elapsed time """
	          self.batch_time.update(time.time() - end)
	          end = time.time()

	          if i % self.print_freq == 0:
	              print('Test: [{0}/{1}]\t'
	                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
	                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
	                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
	                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
	                     i, len(val_loader), batch_time=self.batch_time, loss=self.losses,
	                     top1=self.top1, top5=self.top5))
	              self.mlflow.log_metric("Val Prec1", self.top1.avg.item())
	              self.mlflow.log_metric("Val Prec5", self.top5.avg.item())
	              self.mlflow.log_metric("Val Loss", self.losses.avg.item())

	    print(' Epoch:{0} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
	            .format(epoch, top1=self.top1, top5=self.top5))

	    """ write results to the log """
	    results = open('logs/valid_log.txt', 'a')
	    results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
	            .format(epoch, loss=self.losses, top1=self.top1, top5=self.top5))
	    results.close()

	    return self.top1.avg