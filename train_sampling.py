from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor
# from model.basenet import Predictor_deep_ifsl as Predictor_deep
from model.basenet import Predictor_deep_ifsl_reconstruction as Predictor_deep
from model.pretrain_external import Pretrain

from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset_eposide_SA import return_dataset
from utils.loss import entropy, CrossEntropyKLD
# from utils.loss import adentropy_ori as adentropy
from utils.loss import adentropy_MME_ifsl_reconstruction as adentropy
from utils.criterion_distill import DistillKL
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50100, metavar='N',
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
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
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
parser.add_argument('--reconstruction', action='store_false', default=True,
                    help='using reconstruction in F1')
parser.add_argument('--patience', type=int, default=100, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--threshold', type=float, default=0.95, help='loss weight')
parser.add_argument('--w_kld', type=float, default=0.1, help='loss weight')
parser.add_argument('--k_per', type=float, default=0.25, help='chose the pse-label')
parser.add_argument('--NCE_weight', type=float, default=0.05, help='NCE_weight')
parser.add_argument('--random_sampling', type=str, default=True,
                    help='None')
parser.add_argument('--fixmatch', type=str, default=True,
                    help='using fixmatch method')
args = parser.parse_args()

args.save_check = True
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
    
use_gpu = torch.cuda.is_available()
NCE_scale=7
NCE_weight = args.NCE_weight
unlabel_NCE_weight = args.NCE_weight

NCE_loss=1
B_Twin_loss=0
kd_T=4
KL_weight=0.01
fixed_weight=0.0
num_instance_1=12
weight_IFSL_logits=0.5
criterion_div = DistillKL(kd_T)
Use_IFSL=False
import time
 
localtime = time.asctime( time.localtime(time.time()) )
localtime=localtime.replace(' ','_')
if NCE_loss:
    record_dir = '/gdata2/tuky/test/%s_%s/' % (args.source, args.target) + localtime
    # record_dir = '/gdata2/tuky/test/%s_%s_%s_shot/threshold%s_kper%s_NCE_weight%s/random_sampling_%s_fixmacth_%s/%s' % (
    #     args.dataset, args.net, args.num, args.threshold, args.k_per, args.NCE_weight, args.random_sampling, args.fixmatch, args.method)+localtime+'_'+args.source+'_'+args.target
if B_Twin_loss:
    record_dir = '/gdata2/tuky/DA_ours/record_eposide_SD/SD_IFSL_barlow_twins_2_augment_no_KL_fixed_fc_fine_source_NCE/%s/%s' % (args.dataset, args.method)+localtime+'_'+args.source+'_'+args.target+'_threshold_'+str(args.threshold)+'_unlabel_NCE_'+str(unlabel_NCE_weight)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))
#os.system('cp -r ./data ' +record_dir)
os.system('cp -r ./utils ' +record_dir)
os.system('cp -r ./model ' +record_dir)
os.system('cp -r ./loaders ' +record_dir)
os.system('cp  ./*.py ' +record_dir)
os.system('cp  ./*.sh ' +record_dir)
import sys
#sys.path.append("/home/lijunjie/Few_shot/CAN/fewshot-CAN-master/torchFewShot/utils/")
from utils.logger.logger import Logger
import os.path as osp
sys.stdout = Logger(osp.join(record_dir, 'log_train.txt'))
#exit()
torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
random.seed(777)
torch.backends.cudnn.deterministic = True
#exit()
G_model_dir="./record/office_home/MME/G_iter_model_MME_Art_to_Clipart_best_val.pth.tar"
F_model_dir="./record/office_home/MME/F1_iter_model_MME_Art_to_Clipart_best_val.pth.tar"

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
    bs = 24
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
    bs = 32
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
    #F1.load_state_dict(torch.load("./record/office_home/MME/F1_iter_model_MME_Art_to_Clipart_best_val.pth.tar"))                    
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
im_data_tu_pesu = torch.FloatTensor(1)
im_data_tu_48 = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
gt_labels_cls=torch.LongTensor(1)
gt_labels_un_pesu = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
im_data_tu_pesu = im_data_tu_pesu.cuda() 
im_data_tu_48 = im_data_tu_48.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
gt_labels_cls=gt_labels_cls.cuda()
gt_labels_un_pesu = gt_labels_un_pesu.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_tu_pesu = Variable(im_data_tu_pesu)
im_data_tu_48 = Variable(im_data_tu_48)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
gt_labels_un_pesu = Variable(gt_labels_un_pesu)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)
gt_labels_cls_ljj = Variable(gt_labels_cls)
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
def train(num_instance_2=12):
    #print(args.reconstruction)
    #exit()
    criterion_un = CrossEntropyKLD(num_class=len(class_list), mr_weight_kld=args.w_kld)    
    args.reconstruction=False
    if args.reconstruction:
        pretrain_model=Pretrain(args,G_model_dir, F_model_dir,args.net,len(class_list),args.T, True)
    else:
        pretrain_model=0
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    NL=nn.NLLLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    #data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0
    best_acc_test_truth=0
    best_acc_val=0
    bn = nn.BatchNorm1d(512, affine=False).cuda()
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        #if step % len_train_target == 0:
            #data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        # data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        data_s[2] = data_s[2][0]
        data_s[3] = data_s[3][0]
        data_s[4] = data_s[4][0]
        data_s[5] = data_s[5][0]
        data_s[6] = data_s[6][0]
        data_s[7] = data_s[7][0]
        with torch.no_grad():
            im_data_s.resize_(data_s[2].size()).copy_(data_s[2])
            gt_labels_s.resize_(data_s[4].size()).copy_(data_s[4])
            im_data_t.resize_(data_s[3].size()).copy_(data_s[3])
            gt_labels_t.resize_(data_s[4].size()).copy_(data_s[4])
            gt_labels_cls.resize_(data_s[5].size()).copy_(data_s[5])        
        
            random_samples=args.random_sampling
            if random_samples=='True':
                #print("sampling is random!")
                im_data_tu_48.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
                # im_data_tu.resize_(data_s[6].size()).copy_(data_s[6])
            else:
                im_data_tu_48.resize_(data_s[7].size()).copy_(data_s[7])
                # im_data_tu.resize_(data_s[6].size()).copy_(data_s[6])
            im_data_tu.resize_(data_s[6].size()).copy_(data_s[6])                
            # im_data_tu_pesu.resize_(data_s[7].size()).copy_(data_s[7])
            # gt_labels_un_pesu.resize_(data_s[8].size()).copy_(data_s[8])
        zero_grad_all()
        #if 
        im_data_tu_48_simple=im_data_tu_48[:,:3,:,:]
        im_data_tu_48_complex=im_data_tu_48[:,3:,:,:]
        ranmdom_fuse=np.random.uniform(0,1,1)[0]
        if True:
            if ranmdom_fuse > 0.5:
                weight_fuse = np.random.uniform(0,0.4,1)[0]
                im_data_t_temp = (1-weight_fuse)*im_data_t+weight_fuse*im_data_tu[:,3:]
                # im_data_t_temp=im_data_tu[24:]
                im_data_t.resize_(im_data_t_temp.size()).copy_(im_data_t_temp)   
        if False:
            #print(step)
            if ranmdom_fuse>0.5:
                weight_fuse=np.random.uniform(0,0.4,1)[0]
                im_data_s_temp=(1-weight_fuse)*im_data_s+weight_fuse*im_data_tu[:,3:]
                #im_data_t_temp=im_data_tu[24:]
                im_data_s.resize_(im_data_s_temp.size()).copy_(im_data_s_temp) 
        #print(im_data_s.shape)
        num_instance = num_instance_2
        base_com = False
        if base_com:
            im_data_s_complex = im_data_s[:,:3,:,:]
            im_data_s_simple = im_data_s[:num_instance,3:,:,:]
            data = torch.cat((im_data_s_simple,im_data_s_complex, im_data_t,im_data_tu_48_simple,im_data_tu_48_complex), 0)            
        else:
            im_data_s_complex = im_data_s[:num_instance,:3,:,:]
            im_data_s_simple = im_data_s[:,3:,:,:]
            data = torch.cat((im_data_s_complex,im_data_s_simple, im_data_t,im_data_tu_48_simple,im_data_tu_48_complex), 0)

        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        target_3 = torch.cat((gt_labels_s[:num_instance],gt_labels_s, gt_labels_t), 0)        
        output = G(data)
        source_mix = False
        if source_mix:
            if ranmdom_fuse>0.7:
                output[:24]=output[:24]+output[24:48]
            #output=output[:144]
        #reconstuct_feature=pretrain_model.get_restuction_features(data)

        out1,ifsl_out,x = F1(output,pretrain_model,data)
        source_complex=out1[num_instance:24+num_instance]
        target_complex=out1[num_instance+24:48+num_instance]

        Fixmatch = args.fixmatch
        if Fixmatch=='True':
            # print("use fixmatch")
            # exit()
            pseudo_label = torch.softmax(out1[2*bs+num_instance:2*bs+2*bs+num_instance].detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()   
            loss_u = criterion_un(out1[4*bs+num_instance:6*bs+num_instance], targets_u, mask)
            loss_u_ifsl = criterion_un(ifsl_out[4*bs+num_instance:6*bs+num_instance], targets_u, mask)

        out1_ori=out1
        ifsl_out_ori=ifsl_out
        x_feature=x
        unlabel_feature=x[2*bs+num_instance:]
        out1=out1[:2*bs+num_instance]
        ifsl_out=ifsl_out[:2*bs+num_instance]
        x_s_c=x[:num_instance*2]
        #x_complex=x[24:48]
        x=x[num_instance:2*bs+num_instance]
        if Fixmatch=='True':
            if Use_IFSL:
                loss = criterion(out1[:2*bs+num_instance], target_3)+loss_u+0.5*criterion(ifsl_out[:2*bs+num_instance], target_3)+0.5*loss_u_ifsl#+fixed_weight*criterion(out3[:48+12],target_3)
            else:
                loss = criterion(out1[:2*bs+num_instance], target_3)+loss_u
        elif Use_IFSL:
            loss = criterion(out1[num_instance:2*bs+num_instance], target)+0.5*criterion(ifsl_out[num_instance:2*bs+num_instance], target)
        else:
            loss = criterion(out1[:2*bs+num_instance], target_3)
        ifsl=True
        unlabel_NCE=True
        using_KL=False
        source_NCE=True
        using_logit_NCE=True
        use_style=True
        out_NCE_loss=True
        out_source_NCE=False
        out_unlabel_NCE=True
        using_logit_IFSL_NCE=False
        consistence_aglign=False        
        if consistence_aglign:
            source_complex_soft=source_complex
            target_complex_soft=target_complex
            S_T_attention=F.softmax(torch.mm(source_complex_soft,target_complex_soft.transpose(0,1).contiguous()))
            T_S_attention=F.softmax(torch.mm(target_complex_soft,source_complex_soft.transpose(0,1).contiguous()))
            #print(S_T_attention.sum(1))
            #exit()
            S_reconstruction=torch.mm(S_T_attention,target_complex_soft)
            T_reconstruction=torch.mm(T_S_attention,source_complex_soft)
            #S_reconstruction=torch.log(S_reconstruction)
            #T_reconstruction=torch.log(T_reconstruction)  
            loss=loss+criterion(source_complex, target[:bs])+criterion(target_complex_soft, target[bs:])
        if using_KL:
            loss_KL_cls = KL_weight*criterion_div(out1[:bs], out1[bs:2*bs])
            loss=loss+loss_KL_cls

        # entrance
        if using_logit_NCE:
            out1_soft=F.softmax(out1_ori)
            #out1_soft=F.normalize(out1_ori)
            #out1_soft=x_feature
            if out_NCE_loss:
                #print(out1_soft.shape)
                out1_x=out1_soft[num_instance:2*bs+num_instance]
                NCE_2=torch.mm(out1_x, out1_x.transpose(0, 1).contiguous())
                unit_1 = torch.eye(2*bs).cuda()

                if use_style:
                    NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                    gt_labels_cls_cross=torch.cat([gt_labels_cls+bs,gt_labels_cls])
                    loss = loss+NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross) 
                else:
                    S_C=NCE_2[:bs,bs:]
                    C_S=NCE_2[bs:,:bs]
                    #gt_labels_cls_cross=torch.cat([gt_labels_cls+24,gt_labels_cls])
                    loss = loss+NCE_weight*criterion(NCE_scale*S_C, gt_labels_cls)+NCE_weight*criterion(NCE_scale*C_S, gt_labels_cls)                     
            if out_source_NCE:
                #print(out1_soft.shape)
                out1_x_s_c=out1_soft[:num_instance*2]
                NCE_2=torch.mm(out1_x_s_c,out1_x_s_c.transpose(0,1).contiguous())
                #print(NCE_2.shape)
                #exit()
                unit_1=torch.eye(num_instance*2).cuda()
                #print(unit_1)
                #exit()
                NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                #print(NCE_2.shape)
                #exit()
                #print(gt_labels_cls)
                #print(gt_labels_cls.shape)
                gt_labels_cls_cross = torch.cat([gt_labels_cls[:num_instance]+num_instance,gt_labels_cls[:num_instance]])
                                    
                loss = loss+unlabel_NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross)                 
            if out_unlabel_NCE:
                #print(out1_soft.shape)
                out1_unlabel_feature=out1_soft[2*bs+num_instance:]
                #print(out1_unlabel_feature.shape)
                NCE_2=torch.mm(out1_unlabel_feature,out1_unlabel_feature.transpose(0,1).contiguous())
                #print(NCE_2.shape)
                #exit()
                if use_style:
                    unit_1 = torch.eye(4*bs).cuda()
                    #print(unit_1)
                    #exit()
                    NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                    gt_labels_cls_cross = torch.cat([gt_labels_cls,gt_labels_cls+bs])
                    gt_labels_cls_cross_96 = torch.cat([gt_labels_cls_cross+2*bs,gt_labels_cls_cross])
                    loss = loss+unlabel_NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross_96) 
                else:
                    S_C=NCE_2[:2*bs,2*bs:]
                    C_S=NCE_2[2*bs:,:2*bs]
                    gt_labels_cls_cross=torch.cat([gt_labels_cls,gt_labels_cls+bs])
                    #print(gt_labels_cls)
                    #exit()
                    #gt_labels_cls_cross=torch.cat([gt_labels_cls+24,gt_labels_cls])
                    loss = loss+NCE_weight*criterion(NCE_scale*S_C, gt_labels_cls_cross)+NCE_weight*criterion(NCE_scale*C_S, gt_labels_cls_cross)                     
        # using_logit_IFSL_NCE=False
        if using_logit_IFSL_NCE:
            out1_soft=F.softmax(ifsl_out_ori)
            if out_NCE_loss:
                #print(out1_soft.shape)
                out1_x=out1_soft[num_instance:48+num_instance]
                NCE_2=torch.mm(out1_x,out1_x.transpose(0,1).contiguous())
                #print(NCE_2.shape)
                #exit()
                unit_1 = torch.eye(48).cuda()
                #print(unit_1)
                #exit()
                NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                #print(NCE_2.shape)
                #exit()
                #print(gt_labels_cls)
                #print(gt_labels_cls.shape)
                gt_labels_cls_cross=torch.cat([gt_labels_cls+24,gt_labels_cls])
                loss = loss+weight_IFSL_logits*NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross) 
                if out_source_NCE:
                    #print(out1_soft.shape)
                    out1_x_s_c=out1_soft[:num_instance*2]
                    NCE_2=torch.mm(out1_x_s_c,out1_x_s_c.transpose(0,1).contiguous())
                    #print(NCE_2.shape)
                    #exit()
                    unit_1=torch.eye(num_instance*2).cuda()
                    #print(unit_1)
                    #exit()
                    NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                    #print(NCE_2.shape)
                    #exit()
                    #print(gt_labels_cls)
                    #print(gt_labels_cls.shape)
                    gt_labels_cls_cross=torch.cat([gt_labels_cls[:num_instance]+num_instance,gt_labels_cls[:num_instance]])
                                        
                    loss = loss+weight_IFSL_logits*unlabel_NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross)                 
                if out_unlabel_NCE:
                    #print(out1_soft.shape)
                    out1_unlabel_feature=out1_soft[48+num_instance:]
                    #print(out1_unlabel_feature.shape)
                    NCE_2=torch.mm(out1_unlabel_feature,out1_unlabel_feature.transpose(0,1).contiguous())
                    #print(NCE_2.shape)
                    #exit()
                    unit_1=torch.eye(96).cuda()
                    #print(unit_1)
                    #exit()
                    NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                    #print(NCE_2.shape)
                    #exit()
                    #print(gt_labels_cls)
                    #print(gt_labels_cls.shape)
                    gt_labels_cls_cross=torch.cat([gt_labels_cls,gt_labels_cls+24])
                    gt_labels_cls_cross_96=torch.cat([gt_labels_cls_cross+48,gt_labels_cls_cross])                    
                    loss = loss+weight_IFSL_logits*unlabel_NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross_96)                      
        if False:
            if NCE_loss:
                NCE_2=torch.mm(x,x.transpose(0,1).contiguous())
                unit_1=torch.eye(48).cuda()
                NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                gt_labels_cls_cross=torch.cat([gt_labels_cls+24,gt_labels_cls])
                loss = loss+NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross)
                if source_NCE:
                    NCE_2=torch.mm(x_s_c,x_s_c.transpose(0,1).contiguous())
                    unit_1=torch.eye(num_instance*2).cuda()
                    NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                    gt_labels_cls_cross=torch.cat([gt_labels_cls[:num_instance]+num_instance,gt_labels_cls[:num_instance]])
                                        
                    loss = loss+unlabel_NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross)                 
                if unlabel_NCE:
                    NCE_2=torch.mm(unlabel_feature,unlabel_feature.transpose(0,1).contiguous())
                    unit_1=torch.eye(96).cuda()
                    NCE_2=NCE_2*(1-unit_1)+(-100000)*unit_1
                    gt_labels_cls_cross=torch.cat([gt_labels_cls,gt_labels_cls+24])
                    gt_labels_cls_cross_96=torch.cat([gt_labels_cls_cross+48,gt_labels_cls_cross])                    
                    loss = loss+unlabel_NCE_weight*criterion(NCE_scale*NCE_2, gt_labels_cls_cross_96)                    
            if B_Twin_loss:
                f_a=x[:24]
                f_b=x[24:]
                f_a_norm=(f_a-f_a.mean(0))/f_a.std(0)
                f_b_norm=(f_a-f_b.mean(0))/f_b.std(0)
                if False:
                    c = bn(f_a).T @ bn(f_b)
                    c.div_(24)
                c=(torch.mm(f_a_norm.transpose(0,1).contiguous(),f_b_norm))/24
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/16)
                off_diag = off_diagonal(c).pow_(2).sum().mul(1/16)
                #c_diff=(c-unit_1).pow(2)
                loss_ba = on_diag + 0.004 * off_diag
                loss = loss+NCE_weight*loss_ba
                if ifsl:
                    f_a_ifsl=ifsl_out[:24]
                    f_b_ifsl=ifsl_out[24:]
                    f_a_norm_ifsl=(f_b_ifsl-f_a_ifsl.mean(0))/f_a_ifsl.std(0)
                    f_b_norm_ifsl=(f_a_ifsl-f_b_ifsl.mean(0))/f_b_ifsl.std(0)
                    #print(f_a_norm.shape)
                    #exit(0)
                    if False:
                        c = bn(f_a).T @ bn(f_b)
                    #print(c.shape)
                        c.div_(24)
                    c_ifsl=(torch.mm(f_a_norm_ifsl.transpose(0,1).contiguous(),f_b_norm_ifsl))/24
                    on_diag_ifsl = torch.diagonal(c_ifsl).add_(-1).pow_(2).sum().mul(1/16)
                    off_diag_ifsl = off_diagonal(c_ifsl).pow_(2).sum().mul(1/16)
                    #c_diff=(c-unit_1).pow(2)
                    loss_ba_ifsl = on_diag_ifsl + 0.004 * off_diag_ifsl 
                    loss = loss+NCE_weight*loss_ba_ifsl    
                              
                              
        loss.backward(retain_graph=True)
        #optimizer_g.step()
        #optimizer_f.step()
        #zero_grad_all()
        if not args.method == 'S+T':
            output = G(im_data_tu_48_simple)
            if args.method == 'ENT':
                loss_t = entropy(F1,pretrain_model,im_data_tu_48_simple, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1,pretrain_model,im_data_tu_48_simple, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             step, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test,pretrain_model)
            loss_val, acc_val = test(target_loader_val,pretrain_model)
            G.train()
            F1.train()
            best_val_acheive=False
            best_test_acheive=False
            if acc_test>best_acc_test_truth:
                best_acc_test_truth = acc_test
                best_acc_val=acc_val
                #best_acc_test = acc_test
                best_test_acheive=True                
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                best_val_acheive=True
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('current acc test %f current acc val %f' % (best_acc_test_truth,
                                                        best_acc_val))            
            print('best acc test %f best acc val %f' % (best_acc_test,
                                                        acc_val))
            #print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step,
                                                         best_acc_test,
                                                         acc_val))
            G.train()
            F1.train()
            if args.save_check:
                print('saving model')
                if best_val_acheive:
                    torch.save(G.state_dict(),
                            os.path.join(record_dir,
                                            "G_iter_model_{}_{}_"
                                            "to_{}_best_val.pth.tar".
                                            format(args.method, args.source,
                                                args.target)))
                    torch.save(F1.state_dict(),
                           os.path.join(record_dir,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_best_val.pth.tar".
                                        format(args.method, args.source,
                                               args.target)))
                #if best_test_acheive:
                    #torch.save(G.state_dict(),
                            #os.path.join(record_dir,
                                            #"G_iter_model_{}_{}_"
                                           # "to_{}_best_test.pth.tar".
                                           # format(args.method, args.source,
                                              #  args.target)))
                    #torch.save(F1.state_dict(),
                          # os.path.join(record_dir,
                                        #"F1_iter_model_{}_{}_"
                                        #"to_{}_best_test.pth.tar".
                                       # format(args.method, args.source,
                                         #      args.target)))

def test(loader,pretrain_model):
    G.eval()
    F1.eval()
    test_loss = 0
    test_loss_ifsl=0
    test_loss_fuse=0
    test_loss_fixed=0    
    correct = 0
    correct_ifsl = 0
    correct_fuse = 0 
    correct_fixed = 0    
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    output_all_ifsl = np.zeros((0, num_class))  
    output_all_fuse = np.zeros((0, num_class))  
    output_all_fixed = np.zeros((0, num_class))     
    #criterion_ifsl = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()    
    confusion_matrix = torch.zeros(num_class, num_class)
    confusion_matrix_ifsl = torch.zeros(num_class, num_class)    
    confusion_matrix_fuse = torch.zeros(num_class, num_class)
    confusion_matrix_fixed = torch.zeros(num_class, num_class)    
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            with torch.no_grad():
                im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1,output_ifsl,_ = F1(feat,pretrain_model,im_data_t)
            output3=output1
            output1_fuse=output1+output_ifsl
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            output_all_ifsl = np.r_[output_all_ifsl, output_ifsl.data.cpu().numpy()]     
            output_all_fuse = np.r_[output_all_fuse, output1_fuse.data.cpu().numpy()]   
            output_all_fixed = np.r_[output_all_fixed, output3.data.cpu().numpy()]             
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            pred1_ifsl = output_ifsl.data.max(1)[1] 
            pred1_fuse = output1_fuse.data.max(1)[1]    
            pred1_fixed = output3.data.max(1)[1]             
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
            for t_ifsl, p_ifsl in zip(gt_labels_t.view(-1), pred1_ifsl.view(-1)):
                confusion_matrix_ifsl[t_ifsl.long(), p_ifsl.long()] += 1
            correct_ifsl += pred1_ifsl.eq(gt_labels_t.data).cpu().sum()
            test_loss_ifsl += criterion(output_ifsl, gt_labels_t) / len(loader)
            for t_f, p_f in zip(gt_labels_t.view(-1), pred1_fuse.view(-1)):
                confusion_matrix_fuse[t_f.long(), p_f.long()] += 1
            correct_fuse += pred1_fuse.eq(gt_labels_t.data).cpu().sum()
            test_loss_fuse += criterion(output1_fuse, gt_labels_t) / len(loader)  
            for t_fixed, p_fixed in zip(gt_labels_t.view(-1), pred1_fixed.view(-1)):
                confusion_matrix_fixed[t_fixed.long(), p_fixed.long()] += 1
            correct_fixed += pred1_fixed.eq(gt_labels_t.data).cpu().sum()
            test_loss_fixed += criterion(output3, gt_labels_t) / len(loader)            
    #print('\nTest set: Average loss: {:.4f}, '
    #      'Accuracy: {}/{} F1 ({:.0f}%)\n'.
    #      format(test_loss, correct, size,
    #             100. * correct / size))
    print('acc_ori: {:.4f}, '
          'acc_ifsl: {:.4f}, '
          'acc_ifsl: {:.4f}, '
          'acc_fixed: {:.4f}, '.
          format(100. * float(correct) / size, 100. * float(correct_ifsl) / size, 100. * float(correct_fuse) / size,100. * float(correct_fixed) / size)) 
    print('Accuracy_ori: {}, '
          'Accuracy_ifsl: {}, '
          'Accuracy_fuse: {}, '
          'Accuracy_fixed: {}, '
          'size: {}, '.
          format(correct, correct_ifsl,correct_fuse,correct_fixed,size))     
    return test_loss.data, 100. * float(correct) / size


train(num_instance_2=num_instance_1)
