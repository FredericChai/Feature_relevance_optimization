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
import torch.nn.functional as F

from model import VGG,resnet34 
from utils import *

parser = argparse.ArgumentParser(description = 'Pytorch VGG on PET-CT image')
#path of dataset
parser.add_argument('--dataset',default = '/mnt/HDD1/Frederic/ite_sparse_classification/CriteriaCompare/',
                    type = str,help = 'path of dataset')
#configuration 
parser.add_argument('--epochs',default = 100,type=int,help='epochs for each sparse model to run')
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
parser.add_argument('--evaluate_path',type =str,help = 'path of best model to evaluate the result')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--source_checkpoint', default = '', help = 'path to save the pruned parameters as checkpoint')
parser.add_argument('--sparse_ratio',type = float,help = 'percentage of weights remain')
parser.add_argument('--sparse_epoch',default = 1, type =int, help= 'epochs for sparse trainning the model')
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

module_name =[]
features_in_hook = []
features_out_hook = []

def main():      
    start_epoch = args.start_epoch 
    best_acc = 0 
    #control the path of result
    #result_path: /mnt/HDD1/Frederic/ensemble_baseline/checkpoint
    result_path = os.path.join(args.dataset,'checkpoint/'+args.arch+'-checkpoint-epoch'+str(args.epochs))     
    make_path(result_path) 
    #load the data
    dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'Data_hook')
    #build model with defined architechture
    NN = load_model(args.arch,numOfClasses)
    NN = torch.nn.DataParallel(NN).cuda() 
    cudnn.benchmark = True
    # print('  Total parameters: %.2f' % (sum(p.numel() for p in NN.parameters())))

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)

    # title = 'imageclef-'+args.arch
    # logger = Logger(os.path.join(result_path, 'log.txt'),title = title)
    # logger.set_names(['Learning Rate','Trainning Loss','Valid Loss','Train Acc','Valid Acc'])

    NN.load_state_dict(torch.load(args.source_checkpoint)["state_dict"])
    # Train and validate model
    
    print('begin to hook feature')
    # train_loss, train_acc = hook(dataloaders['train'], NN, criterion, optimizer, epoch,use_cuda)
    dataset_centre = cal_mean_dataset(dataloaders['val'])
    avg_dist, stacked_map_dist = cal_similarity(dataloaders['val'], NN,use_cuda,dataset_centre)
    writer.open('./dist_report.txt','a')/
    writer.write("""==========================
                    Network: {}  Dataset:{} 
                    Checkpoint:{} 
                    ==============
                    Avg dist: {}
                    Whole dist: {}
                    dist type: {}
                    =======================
                """.format(args.arch,args.dataset,args.source_checkpoint,avg_dist,'euclidiean'))


def cal_mean_dataset(testloader):
    # model.eval()
    img_list = []
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.numpy()
        img_list.append(inputs)

    imgs = np.asarray(img_list)
    three_channel_mean = np.mean(imgs,axis=(0,1,2))
    # three_channel_std = np.std(imgs,axis=(0,1,2))
    return three_channel_mean

def cal_similarity(testloader,model,use_cuda,centre):
    # model.eval()
    #centre is the 224*224 matrix of whole datasets
    centre = torch.from_numpy(centre)
    centre = centre.cuda()
    centre_size = centre.shape[0]
    num_layers = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        torch.no_grad()
        stacked_map_dist = 0 
        out,feature= model(inputs) # feature is the activation 
        for stack_feature_map in feature:
            num_layers+=1
            # relu1,layer1,layer2,layer3,layer4 = feature[0],feature[1],feature[2],feature[3],feature[4]
            dim0=stack_feature_map.shape[0]
            dim1=stack_feature_map.shape[1]
            dim2=stack_feature_map.shape[2]
            dim3=stack_feature_map.shape[3]
            #feature map is [8,64,112,112]
            num_feature = dim0*dim1
            stack_feature_map = stack_feature_map.view(num_feature,dim2,dim3)
            map_dist = 0
            if stack_feature_map = 
            for feature_map in stack_feature_map:
                if map_dist
                map_dist += F.pairwise_distance(feature_map,centre,p=2)
                print('map_dist:',map_dist)
            avg_map_dist = map_dist/num_feature
            stacked_map_dist += avg_map_dist

        avg_dist = stacked_map_dist/num_layers

    return avg_dist,stacked_map_dist

# def get_acitvation(name):
#     def hook(module,fea_in,fea_out):
#         activation[name]=fea_out.detach
#         return hook


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
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    if model_arch.endswith('resnet34') or model_arch.endswith('resnet34') :
        NN = resnet34(pretrained = False)
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
