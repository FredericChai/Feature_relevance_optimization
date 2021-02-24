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
from sklearn.metrics import recall_score,accuracy_score

from model import VGG 
from utils import *

parser = argparse.ArgumentParser(description = 'Pytorch VGG on PET-CT image')
#path of dataset
parser.add_argument('--dataset',default = '/mnt/HDD1/Frederic/ite_sparse_classification/CriteriaCompare/',type = str,help = 'path of dataset')
#configuration 
parser.add_argument('--epochs',default = 300,type=int,help='epochs for each sparse model to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='set epoch number to restart the model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')                 
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, default=200,help='Set epochs to decrease learning rate.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
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
parser.add_argument('--evaluate_mode',type =int,default = 0,help = 'path of best model to evaluate the result')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--source_checkpoint', default = '', help = 'path to save the pruned parameters as checkpoint')
parser.add_argument('--sparse_ratio',type = float,help = 'percentage of weights remain')
parser.add_argument('--sparse_epoch',default = 3, type =int, help= 'epochs for sparse trainning the model')
parser.add_argument('--zoom_ratio', type = float,  help = 'ratio for zooming out the weight')
parser.add_argument('--random_zoom',dest = 'random_zoom',action = 'store_true',help = 'use the random zoom method')
parser.add_argument('--probability_integrate',default = 0,type = int,help='use to control evaluate mode')
# parser.add_argument('--evaluate_only',dest = 'evaluate_only',action = 'store_true',help='if in this mode. directly load checkpoint and ')
parser.add_argument('--evaluate_path', default = '', help = 'path to save the pruned parameters as checkpoint')

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
 

    if args.evaluate_mode==1:
            # if use evaluate path, turn to evaluate mode         
            write_probability_integrate_result(args.dataset)

    if not args.sparse_ratio is None:
        #load the data
        dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'Data_new')
        for ite in range(args.sparse_epoch):
            best_acc = 0
            start_epoch = 0
            #sparse_model_path: /mnt/HDD1/Frederic/ensemble_baseline/pruned_checkpoint/0.4-vgg16-epoch...
            #result_path: /mnt/HDD1/Frederic/ensemble_baseline/pruned_checkpoint/0.4-vgg16-epoch.../sparse_ite_0
            sparse_model_path = '{}ByRelevance/{}-{}-epoch{}'.format(args.dataset,
                                                                    args.arch,args.sparse_ratio,args.epochs)
            result_path = '{}ByRelevance/{}-{}-epoch{}/sparse_ite_{}'.format(args.dataset,
                                                                    args.arch,
                                                                    args.sparse_ratio,
                                                                    args.epochs,
                                                                    ite)
            make_path(result_path) 
            sparse_checkpoint_path =  os.path.join(result_path,'sparse_checkpoint.pth.tar')
            if ite ==0: #prune the checkpoint from trained model
                source_checkpoint = args.source_checkpoint
                print('loading normal checkpoint:' + source_checkpoint)
                state_dict = prune(source_checkpoint,args.sparse_ratio,args.arch)
                state = {'sparse_ratio':args.sparse_ratio,
                         'state_dict':state_dict,
                         'architechture':args.arch,
                         'lr':args.lr}
                print('prune normal checkpoint to sparse one:' + sparse_checkpoint_path)
                torch.save(state,sparse_checkpoint_path)
            if ite>0:
                source_checkpoint = os.path.join(sparse_model_path+'/sparse_ite_'+str(ite-1),'model_best.pth.tar')
                print('loading last sparse checkpoint:' + source_checkpoint)
                state_dict = prune(source_checkpoint,args.sparse_ratio,args.arch)
                state = {'sparse_ratio':args.sparse_ratio,
                         'state_dict':state_dict,
                         'architechture':args.arch,
                         'lr':args.lr}
                torch.save(state,sparse_checkpoint_path) 
            #build model with defined architechture

            NN = load_model(args.arch,numOfClasses)
            NN = torch.nn.DataParallel(NN).cuda() 
            cudnn.benchmark = True

            #loss funcion and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)
            # optimizer = optim.Adam(NN.parameters(), lr=args.lr, betas =(0.9,0.99))
            title = 'PET-CT-'+args.arch
            #load checkpoint
            checkpoint = torch.load(sparse_checkpoint_path)
            print('loading checkpoint :'+sparse_checkpoint_path)
            NN.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(result_path, 'log.txt'),title = title)
            logger.set_names(['Learning Rate','Trainning Loss','Valid Loss','Train Acc','Valid Acc'])

            # Train and validate model
            for epoch in range(start_epoch,args.epochs): 
                # adjust_learning_rate(optimizer,epoch)
                # adjust_learning_rate(optimizer,epoch,args.lr,args.schedule,args.gamma)
                for parameter_group in optimizer.param_groups:
                    current = parameter_group['lr']
                print('\nSparse iteration: [%d | %d]Epoch: [%d | %d] LR: %f' % (ite+1,args.sparse_epoch,epoch + 1, args.epochs, current))

                train_loss, train_acc = train(dataloaders['train'], NN, criterion, optimizer, epoch,use_cuda)
                val_loss, val_acc = validate(testDataloaders, NN, criterion, epoch,use_cuda)

                # append logger file
                logger.append([args.lr, train_loss, val_loss, train_acc, val_acc])

                # save model
                is_best = val_acc > best_acc 
                best_acc = max(val_acc, best_acc)
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
            evaluate_checkpoint = torch.load(os.path.join(result_path,'model_best.pth.tar'))
            NN.load_state_dict(evaluate_checkpoint['state_dict'])
            optimizer.load_state_dict(evaluate_checkpoint['optimizer'])
            best_epoch = evaluate_checkpoint['epoch']

            report,predict,target = evaluate(testDataloaders, NN, criterion, class_names, use_cuda)
            #write down classfication report
            writer = open(os.path.join(result_path,'classification_report.txt'),'w')
            writer.write(report+'\n')
            writer.write('best_epoch:'+str(best_epoch)+'\n')
            writer.write('best_acc:'+str(best_acc))
            # writer.write('train_schedule: epoch 0-%d,lr:%f   epoch %d-%d,lr:%f  ' % 
            #             (args.schedule,args.lr,args.schedule,args.epochs,args.lr*args.gamma*args.gamma))
            writer.close()
            # write down predict result
            predict_writer = open(os.path.join(result_path,'best_predict.txt'),'w')
            predict_writer.write(predict)
            predict_writer.close()
            #write down target result
            target_writer = open(os.path.join(result_path,'target.txt'),'w')
            target_writer.write(target)
            target_writer.close()  

        write_probability_integrate_result(args.dataset)
    # else use the normal train method

 
       
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
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
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
     # if use sparse iterative train method, then set the sparse ratio


def write_probability_integrate_result(path):
      #path to store the result
    target_label = path+'Data_new/'
    one_target  = np.loadtxt(target_label+'target.txt',dtype = int)
    #use k to store the index of final result
    k=0
    #the final result is used to store the final result
    final_result = np.zeros((len(one_target),30),dtype = float) 
    result_path = args.dataset+'pruned_CP/'+\
                         args.arch+'-'+str(args.sparse_ratio)+'-epoch'+str(args.epochs)+'/'
    make_path(result_path)
    writer =  open(result_path+'/Integrate_probability_result_WeightVote.txt','a')
    for i in range(args.sparse_epoch):

        source = result_path+'sparse_ite_'+str(i)
        print('begin to evaluate, create path for store result,source from {}'.format(source))
        #load data
        dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'Data_new')      
        #build model with defined architechture
        NN = load_model(args.arch,numOfClasses)
        NN = torch.nn.DataParallel(NN).cuda() 
        cudnn.benchmark = True
        #evaluate the model based on the best model on val set
        evaluate_checkpoint = torch.load(os.path.join(source,'model_best.pth.tar'))
        NN.load_state_dict(evaluate_checkpoint['state_dict'])
        NN.eval()
        #the predict is the probability of the ouput ,in this case, each sample will have a 1001*7 length array
        predict= Probability_evaluate(testDataloaders, NN, class_names, use_cuda) 
        #integrate result from 
        predictor  = np.asarray(predict,dtype = float)
 
        final_result +=predictor  
        print('ite:',i)   
        vote = np.argmax(final_result,axis=1)
        report = classification_report(one_target,vote,digits = 4, labels = range(30))
        print(report)
        score = accuracy_score(one_target,vote)
        print('Final Accuracy:',score)     
        writer.write(report+'\n')
        writer.write('Accuracy:'+str(score)+'\n')
        writer.write('sparse_epoch: {} \n ======================== \n'.format(str(i+1)))
    writer.flush()
    writer.close()


def validate(testloader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

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
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

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
        loss = criterion(outputs, targets)
        
        # record prediction and target 
        _,output = torch.max(outputs.data,1)
        pred += output.tolist()
        targ += targets.tolist()

    pred_str = ' '.join([str(i) for i in pred])
    targ_str = ' '.join([str(i) for i in targ])
    # sensitivity, F1-score
    report = classification_report(targ,pred,digits = 4,target_names =class_names)

    return report,pred_str,targ_str
 
def Probability_evaluate(testloader,model,class_names,use_cuda):

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
        
        # record prediction and target 
        # _,output = torch.max(outputs.data,1)
        pred += outputs.tolist()
        # targ += targets.tolist()

    # pred_str = ' '.join([str(i) for i in pred])
    # targ_str = ' '.join([str(i) for i in targ])
    # sensitivity, F1-score
    # report = classification_report(targ,pred,digits = 4, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],target_names =class_names)

    return pred

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def load_data_NoTest(path):

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
    }

    data_dir = path
    image_datasets = {
        x : datasets.ImageFolder(os.path.join(data_dir,x),
                             data_transforms[x])
        for x in ['train','val']
    }

    dataloaders = {x: torch.utils.data.DataLoader(
                                                image_datasets[x],     
                                                batch_size=32, 
                                                shuffle=True,
                                                num_workers=1) 
                                                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    numOfClasses = len(class_names)

    return dataloaders,dataset_sizes,class_names,numOfClasses

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
                                                        batch_size=24, 
                                                        shuffle=True,
                                                        num_workers=0) 
                                                    for x in ['train','val']}

    testImageLoaders = torch.utils.data.DataLoader(image_datasets['test'],batch_size=12,shuffle=False)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    numOfClasses = len(class_names)

    return dataloaders,testImageLoaders,dataset_sizes,class_names,numOfClasses


def adjust_learning_rate(optimizer,epoch,learning_rate,schedule,gamma):
    if epoch == schedule:
        new_learning_rate = gamma*learning_rate
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = new_learning_rate


def load_model(model_arch,numOfClasses):

    if model_arch.endswith('vgg16'):
        NN = VGG.vgg16(num_classes = numOfClasses)
    elif model_arch.endswith('vgg16_pretrained'):
        NN = models.vgg16(pretrained = True)
        num_features = NN.classifier[6].in_features  
        NN.classifier[6] = nn.Linear(num_features,numOfClasses)  #change last fc layer and keep all other layer if used pretrained model)
    elif model_arch.endswith('resnet18_pretrained') or model_arch.endswith('resnet18'):
        NN = models.resnet18(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet34'):
        NN = models.resnet34(pretrained = False)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet34_pretrained'):  
        NN = models.resnet34(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)   

    elif model_arch.endswith('resnet50_pretrained'):  
        NN = models.resnet50(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet50'):
        NN = models.resnet50(pretrained = False)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)

    elif model_arch.endswith('resnet101'):
        NN = models.resnet101(pretrained = False)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif model_arch.endswith('resnet101_pretrained'):
        NN = models.resnet101(pretrained = True)
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

    return NN

if __name__ == '__main__':
    main()
