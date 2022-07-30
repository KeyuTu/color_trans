from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record_test/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))

torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
random.seed(777)
torch.backends.cudnn.deterministic = True
#exit()
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def main():
    #print('lkkkkkkkkkkkkkkk')

    criterion = nn.CrossEntropyLoss().cuda()
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0
    #F1_load_dir='./record/'+args.dataset+'/MME/'+args.source+'_'+args.target+'/'+'F1_iter_model_MME_'+args.source+'_to_'+args.target+'_best_val.pth.tar'
    #G_load_dir='./record/'+args.dataset+'/MME/'+args.source+'_'+args.target+'/'+'G_iter_model_MME_'+args.source+'_to_'+args.target+'_best_val.pth.tar' 
    #F1_load_dir='./record_eposide_data_augment/office_home/'+args.source+'_'+args.target+'/'+'F1_iter_model_MME_'+args.source+'_to_'+args.target+'_best_val.pth.tar'
    #G_load_dir='./record_eposide_data_augment/office_home/'+args.source+'_'+args.target+'/'+'G_iter_model_MME_'+args.source+'_to_'+args.target+'_best_val.pth.tar'         
    root_saved="/gdata2/lijj/DA_ours/record_eposide_SD_0.83/office_home/new_no_IFSL_nolossIFSL_nosource_usecross_nofixmatch_compare76.9_0.5_num_12/multi/model/"
    F1_load_dir=root_saved+'F1_iter_model_MME_'+args.source+'_to_'+args.target+'_best_val.pth.tar'
    G_load_dir=root_saved+'G_iter_model_MME_'+args.source+'_to_'+args.target+'_best_val.pth.tar'
    F1.load_state_dict(torch.load(F1_load_dir),strict=False)
    G.load_state_dict(torch.load(G_load_dir),strict=False)    
    for step in range(1):
        G.zero_grad()
        F1.zero_grad()
        #zero_grad_all()
        #if step % args.log_interval == 0:
            #print(log_train)
        #if step % args.save_interval == 0 and step > 0:
        loss_test, acc_test = test(target_loader_test)
        #loss_val, acc_val = test(target_loader_val)


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    print('lllllllllllllll')
    target_unlabel={}
    label_old=[]
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            with torch.no_grad():
                im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
                name=data_t[2]          
            feat = G(im_data_t)
            output1,_ = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            prob_peos = output1.data.max(1)[0]
            prob_soft=F.softmax(output1,1)
            #print(output1.shape)
            #exit()
            #print(F.softmax(output1,1))
            #print(output1.shape)
            #exit(0)
            #print(pred1)
            #print(prob_peos)
            #print(name[0])
            #exit()
            label=int(pred1.cpu().numpy())
            prob_peos=prob_peos.cpu().numpy()[0]
            #print(prob_soft[0][label].cpu().numpy())
            #exit()
            prob_soft=prob_soft[0][label].cpu().numpy()
            #print(type(label))
            #exit()
            #if batch_idx>10:
                #exit()
            if str(label) not in label_old:
                label_old.append(str(label))
                target_unlabel[str(label)]=[[],[],[],[],[]]
            target_unlabel[str(label)][0].append(name[0])
            target_unlabel[str(label)][1].append(prob_peos)  
            target_unlabel[str(label)][2].append(prob_soft)             
            #exit()
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
        #print(target_unlabel['0'])
        #exit()
        #print(
        for key in target_unlabel.keys():
            prob_out=sorted(target_unlabel[key][1],reverse=True)
            #print(prob_out)
            #exit()
            ori_prob_numpy=np.array(target_unlabel[key][1])
            #print(ori_prob_numpy.shape)
            #print(prob_out[0])
            #exit()
            for ljj_index in range(len(prob_out)):
                index=np.where(ori_prob_numpy==prob_out[ljj_index])[0][0]
                target_unlabel[key][3].append(index)
        for key in target_unlabel.keys():
            prob_out=sorted(target_unlabel[key][2],reverse=True)
            #print(prob_out)
            #exit()
            ori_prob_numpy=np.array(target_unlabel[key][2])
            #print(ori_prob_numpy.shape)
            #print(prob_out[0])
            #exit()
            for ljj_index in range(len(prob_out)):
                index=np.where(ori_prob_numpy==prob_out[ljj_index])[0][0]
                target_unlabel[key][4].append(index)                
            #print(index)
            #print(prob_out)
           # print(target_unlabel[key][1])
            #exit()
        np.save('./peusde_multi_3shot_IFSL/'+args.source+'_'+args.target+'_unlabel_3shot_stage1',target_unlabel)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    print(100. * correct / size)
    return test_loss.data, 100. * float(correct) / size

print('llllllll')
main()
print('ooooooooooooo')
exit()