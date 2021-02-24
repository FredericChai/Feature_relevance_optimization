from __future__ import print_function
import os
import argparse
import math
import random
import time

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import recall_score,accuracy_score,average_precision_score,precision_score,roc_auc_score

from model import VGG 
from utils import *

parser = argparse.ArgumentParser(description = 'Pytorch VGG on PET-CT image')
#path of dataset
parser.add_argument('--dataset',default = '/mnt/HDD1/Frederic/ite_sparse_classification/SSF_ISIC17/',type = str,help = 'path of dataset')
#configuration 
parser.add_argument('--epochs',default = 100,type=int,help='epochs for each sparse model to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='set epoch number to restart the model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')                 
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, default=20,help='Set epochs to decrease learning rate.')
parser.add_argument('--gamma', type=float, default=0.8, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
#Checkpoints
parser.add_argument('-c', '--checkpoint', default='pruned_checkpoint/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the checkpoint to be optimized')
#Architechture for VG G
parser.add_argument('--arch',default='vgg16',type=str,help='VGG Architechture to load the checkpoint')

#Meter for measure()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate_path',type =str,help = 'path of best model to evaluate the result')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--source_checkpoint', default = '', help = 'path to save the pruned parameters as checkpoint')
parser.add_argument('--sparse_ratio',type = float,help = 'percentage of weights remain')
parser.add_argument('--sparse_epoch',default = 3, type =int, help= 'epochs for sparse trainning the model')
parser.add_argument('-b' ,'--batchsize', type = int,  help = 'ratio for zooming out the weight')
parser.add_argument('--random_zoom',dest = 'random_zoom',action = 'store_true',help = 'use the random zoom method')
# parser.add_argument('--evaluate_only',dest = 'evaluate_only',action = 'store_true',help='if in this mode. directly load checkpoint and ')

# parser.add_argument('--norm_activation',dest = 'norm_activation',action = 'store_true')
# parser.add_argument('--filter_prune',dest = 'filter_prune',action = 'store_true')


#load all args
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

#Define the id of device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

#Set Random seed for 
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)



def main():      
    start_epoch = args.start_epoch 
    best_acc = 0 
    best_auc = 0
    best_spe = 0 
    best_sen = 0
    #control the path of result
    #result_path: /mnt/HDD1/Frederic/ensemble_baseline/checkpoint
    result_path = os.path.join(args.dataset,'skENL_checkpoint/'+args.arch+'-checkpoint-epoch'+str(args.epochs))     
    make_path(result_path) 
    #load the data
    dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'ISIC17_new/SK/')
    #build model with defined architechture
    NN = load_model(args.arch,numOfClasses)
    NN = NN.cuda() 
    cudnn.benchmark = True
    # print('  Total parameters: %.2f' % (sum(p.numel() for p in NN.parameters())))
    target  = np.loadtxt('./ByRelevance/resnet101_pretrained-0.2-epoch300/sparse_ite_0/target.txt',dtype=int)
    targ = np.asarray(target)
    #loss funcion and optimizer
    # # cs_w = torch.tensor([1000,1500],dtype=torch.float32).cuda()
    # cs_w = torch.tensor([2778,1972],dtype=torch.float32).cuda()
    # cs_w = cs_w/cs_w.sum()
    # cs_w = 1/cs_w
    # cs_w = cs_w/cs_w.sum()
    # # cs_w.cuda()
    # print(cs_w)
    # criterion = nn.CrossEntropyLoss(cs_w)

    cs_w = torch.tensor([1,5],dtype=torch.float32).cuda()
    # cs_w = cs_w.expand(2)
    criterion = nn.BCEWithLogitsLoss(pos_weight = cs_w)
    optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)

    title = 'imageclef-'+args.arch
    #load checkpoint
    logger = Logger(os.path.join(result_path, 'log.txt'),title = title)
    logger.set_names(['T_loss','V_loss','t_acc','v_acc','v_auc','v_spe','v_sen'])
    # Train and validate model
    for epoch in range(start_epoch,args.epochs): 
        # adjust the learning rate when the epoch is in the schdule    
        adjust_learning_rate(optimizer,epoch)
        for parameter_group in optimizer.param_groups:
            current = parameter_group['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current ))

        train_loss, train_acc = train(dataloaders['train'], NN, criterion, optimizer, epoch,use_cuda)
        val_loss,top1,val_acc,spe,sen,auc = validate(testDataloaders, NN, criterion, epoch,use_cuda,targ)
        val_acc= round(val_acc,3)
        spe =round(spe,3)
        sen =round(sen,3)
        auc = round(auc,3)
        print('acc:{} spe:{} sen:{} auc:{}'.format(val_acc,spe,sen,auc))
        print("best_acc {},best_auc:{},best_spe:{},best_sen:{}".format(best_acc,best_auc,best_spe,best_sen))
        # append logger file
        logger.append([train_loss, val_loss, train_acc, val_acc,spe,sen,auc])

        # save model
        is_best = val_acc > best_acc 
        best_acc = max(val_acc, best_acc)
        best_auc = max(auc,best_auc)
        best_spe = max(spe, best_spe)
        best_sen = max(sen,best_sen)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': NN.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=result_path)

    logger.close()
    logger.plot()
    savefig(os.path.join(result_path, 'log.eps'))
    print('Best acc:',best_acc)
    #evaluate the model based on the best model on val set
    evaluate_checkpoint = torch.load(os.path.join(result_path,'model_best.pth.tar'))
    NN.load_state_dict(evaluate_checkpoint['state_dict'])
    optimizer.load_state_dict(evaluate_checkpoint['optimizer'])
    best_epoch = evaluate_checkpoint['epoch']
    report,predict,target,accuracy = evaluate(testDataloaders, NN, criterion, class_names, use_cuda)
    #write down classfication report
    writer = open(os.path.join(result_path,'classification_report.txt'),'w')
    writer.write(report+'\n'+str(accuracy)+'\n')
    writer.write('best_epoch:'+str(best_epoch)+'\n')
    writer.write('best_acc:'+str(best_acc))
    writer.close()
    # write down predict result
    predict_writer = open(os.path.join(result_path,'best_predict.txt'),'w')
    predict_writer.write(predict)
    predict_writer.close()
    #write down target result
    target_writer = open(os.path.join(result_path,'target.txt'),'w')
    target_writer.write(target)
    target_writer.close()

def train(trainloader,model,criterion,optimizer,epoch,use_cuda):
    #train mode
    model.train()
    #metrics of the model
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()


    with tqdm(total = len(trainloader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)

            y_batch = args.batchsize
            y_onehot = torch.FloatTensor(targets.shape[0],2).cuda()

            y_onehot.zero_()  
            targets= targets.unsqueeze(dim=1)
            # print(targets.shape)
            y_onehot.scatter_(1,targets,1)
            loss = criterion(outputs, y_onehot)

            # measure accuracy and record loss
            prec1, prec5 = accuracy_top1(outputs.data, targets.data, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()   
            pbar.set_description('loss: %.4f top1: %.4f' % (loss.view(-1).data.tolist()[0],top1.avg))
            pbar.update(1)
    return (losses.avg, top1.avg)

def validate(testloader, model, criterion, epoch, use_cuda,targ):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pred_score = []
    pred = []
    labels = []
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        torch.no_grad()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        y_onehot = torch.FloatTensor(targets.shape[0],2).cuda()

        y_onehot.zero_()  
        targets= targets.unsqueeze(dim=1)
        # print(targets.shape)
        y_onehot.scatter_(1,targets,1)
        loss = criterion(outputs, y_onehot)
        # loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy_top1(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        out = nn.Sigmoid()(outputs)
        # record prediction and target 
        pred_score += out[:,1].tolist()
        _,output = torch.max(outputs.data,1)
        pred += output.tolist()
        labels += targets.tolist()

    pred_score = np.asarray(pred_score)
    accuracy = accuracy_score(labels,pred)
    precision = precision_score(labels,pred)
    recall = recall_score(labels,pred)
    print(pred)
    # print(targ)
    # print(pred_score)
    AUC = roc_auc_score(targ,pred_score)
    return losses.avg, top1.avg, accuracy,precision,recall,AUC


def evaluate(testloader,model,criterion,class_names,use_cuda):

    #evaluate mode
    model.eval()
    pred = []
    targ = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        torch.no_grad()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        # loss = criterion(outputs, targets)
        
        # record prediction and target 
        _,output = torch.max(outputs.data,1)
        pred += output.tolist()
        targ += targets.tolist()

    pred_str = ' '.join([str(i) for i in pred])
    targ_str = ' '.join([str(i) for i in targ])
    # sensitivity, F1-score
    report = classification_report(targ,pred,digits = 4,target_names =class_names)
    accuracy = accuracy_score(targ,pred)

    return report,pred_str,targ_str,accuracy
 


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch>10 and epoch%15==0:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def load_data(path):
    #Load data and augment train data
    data_transforms = {
        #Augment the trainning data
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), #crop the given image
            transforms.RandomHorizontalFlip(),  #horizontally flip the image
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Scale and normalize the validation data
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Scale and normalize the validation data
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

    data_dir = path
    # testData_dir = '/mnt/HDD1/Frederic/ensemble_baseline/TestImage/'

    image_datasets = {
            x : datasets.ImageFolder(os.path.join(data_dir,x),
                                 data_transforms[x])
            for x in ['train','val','test']
        }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],     
                                                        batch_size=args.batchsize, 
                                                        shuffle=True,
                                                        num_workers=4) 
                                                    for x in ['train','val']}

    testImageLoaders = torch.utils.data.DataLoader(image_datasets['test'],batch_size=12,shuffle=False)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    numOfClasses = len(class_names)

    return dataloaders,testImageLoaders,dataset_sizes,class_names,numOfClasses


def load_model(model_arch,numOfClasses):

    if model_arch.endswith('vgg16_pretrained'):
        NN = models.vgg16(pretrained = True)
        num_features = NN.classifier[6].in_features  
        NN.classifier[6] = nn.Linear(num_features,numOfClasses)  #change last fc layer and keep all other layer if used pretrained model
    elif model_arch.endswith('resnet101_pretrained') or model_arch.endswith('resnet101') :
        NN = models.resnet101(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet18_pretrained') or model_arch.endswith('resnet18'):
        NN = models.resnet18(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch =='resnet34_pretrained':
        NN = models.resnet34(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch =='resnet34':
        NN = models.resnet34(pretrained = False)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet50_pretrained') or model_arch.endswith('resnet50'):
        NN = models.resnet50(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet152_pretrained'):
        NN = models.resnet152(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet152'):
        NN = models.resnet152(pretrained = False)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('densenet121_pretrained'):
        NN =  models.densenet121(pretrained=True)
        num_ftrs = NN.classifier.in_features
        NN.classifier = nn.Linear(num_ftrs, numOfClasses)


    return NN
if __name__ == '__main__':
    main()
