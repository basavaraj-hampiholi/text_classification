import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchsummary import summary

from trainer import Trainer
from dataset.dbpedia import DBpediaDataset 
from utils import save_checkpoint, adjust_learning_rate
from models.classifier import initilize_model
import mlflow
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from config import *
import time
import argparse
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


""" Define argparser to read the command line arguments """
parser = argparse.ArgumentParser(description='multimodal fusion')
parser.add_argument('--dataroot', default='/home/vision/BMW_task/hmtc', help='root directory of dataset')
parser.add_argument('--lr', type=float, default=0.05, help='base learning rate')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--num_classes', type=int, default=219, help='Number of categories')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Number of workers')
parser.add_argument('--embed_dim', type=int, default=300, help='Embed dimensions')
parser.add_argument('--pretrain', type=bool, default=True, help='Load pretrained weights')

def main():
    """ This is main method of the program. Begins the execution """

    args = parser.parse_args()

    model = initilize_model(vocab_size=577154,
                                      embed_dim=args.embed_dim,
                                      learning_rate=0.25,
                                      dropout=0.5)

    model.cuda()

    """ define loss function (criterion) and optimizer """
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=1e-4, momentum=0.9, nesterov=True)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15,25])                       
    cudnn.benchmark = True


    train_set = DBpediaDataset(args.dataroot, np_train, np_trn_lbls, transform=None)    
    valid_set = DBpediaDataset(args.dataroot, np_valid, np_val_lbls, transform=None)    

                                                                                                                                                                          
    # """ Loading dataset into dataloader """
    train_loader =  torch.utils.data.DataLoader(train_set, batch_size=args.bs,
                                               shuffle=True, num_workers=args.nworkers)

    val_loader =  torch.utils.data.DataLoader(valid_set, batch_size=args.bs,
                                               shuffle=True, num_workers=args.nworkers)


    start_time= time.time()

    best_prec1=0

    """ Create on object of Trainer class """
    with mlflow.start_run():
        trainer_obj = Trainer(model, criterion, optimizer, mlflow)
        for epoch in range(0, args.epochs):

            mlflow.log_param("lr", args.lr)

            """ train on trainset """
            trainer_obj.train(train_loader, epoch)

            """ evaluate on validation set """
            prec1 = trainer_obj.validate(val_loader, epoch)
            print('Top Precision:',prec1)

            """ adjusts the learning rate after specified n epochs in scheduler (see above-[15,25]) 
                This is user defined and can be changed according the training curve """
            scheduler.step()


            """ remember best prec@1 and save the checkpoint with best weights"""
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(model.state_dict(), is_best, ckpt_path)
            mlflow.pytorch.log_model(model, "models")

    
    """ Compute and display the total time taken to train and test the model """
    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)


if __name__ == '__main__':
   main()
