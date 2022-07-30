import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent

def adentropy_MME(F1, feat, lamda, eta=1.0):
    #print(feat.shape)
    #exit()
    out_t1,out_t1_ifsl ,_= F1(feat, reverse=True, eta=eta)
    if len(out_t1.shape)>2:
        out_t1=out_t1.mean(2).mean(2)
    out_t1 = F.softmax(out_t1)#.reshape(out_t1.shape[0],out_t1.shape[1],-1)
    out_t1_ifsl = F.softmax(out_t1_ifsl)#.reshape(out_t1.shape[0],out_t1.shape[1],-1)    
    #print(out_t1.sum(1))
    #exit()
   # out_t1=out_t1.transpose(1,2).contiguous().reshape(-1,out_t1.shape[1])
    #print(out_t1.shape)
    #exit()
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))+lamda * torch.mean(torch.sum(out_t1_ifsl *
                                              (torch.log(out_t1_ifsl + 1e-5)), 1))
    return loss_adent
    
def adentropy_MME_ifsl_reconstruction(F1,pretrain_model,image_data, feat, lamda,param_fc2=None, eta=1.0):
    #print(feat.shape)
    #exit()
    if param_fc2 is None:
        out_t1,out_t1_ifsl ,_=F1(feat,pretrain_model,image_data, reverse=True, eta=eta)
    else:
        out_t1,out_t1_ifsl ,_=F1(feat,pretrain_model,image_data, param_fc2,reverse=True, eta=eta)
    #out_t1,out_t1_ifsl ,_= F1(feat, reverse=True, eta=eta)
    if len(out_t1.shape)>2:
        out_t1=out_t1.mean(2).mean(2)
    out_t1 = F.softmax(out_t1)#.reshape(out_t1.shape[0],out_t1.shape[1],-1)
    out_t1_ifsl = F.softmax(out_t1_ifsl)#.reshape(out_t1.shape[0],out_t1.shape[1],-1)    
    #print(out_t1.sum(1))
    #exit()
   # out_t1=out_t1.transpose(1,2).contiguous().reshape(-1,out_t1.shape[1])
    #print(out_t1.shape)
    #exit()
    use_IFSL=False
    if use_IFSL:
        loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))+lamda * torch.mean(torch.sum(out_t1_ifsl *
                                              (torch.log(out_t1_ifsl + 1e-5)), 1))
    else:
        loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))        
    return loss_adent
    
def adentropy(F1, feat, lamda,param_fc2=None, eta=1.0):
    if param_fc2 is None:
        out_t1,_ = F1(feat, reverse=True, eta=eta)
    else:
        out_t1,_,out_t1_transformer,_,out_t1_fuse,_= F1(feat, param_fc2,reverse=True, eta=eta)
    #out_t1,_ = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    if param_fc2 is not None:
        out_t1_transformer = F.softmax(out_t1_transformer) 
        out_t1_fuse = F.softmax(out_t1_fuse)    
        loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))+lamda * torch.mean(torch.sum(out_t1_transformer *
                                              (torch.log(out_t1_transformer + 1e-5)), 1))+lamda * torch.mean(torch.sum(out_t1_fuse *
                                              (torch.log(out_t1_fuse + 1e-5)), 1))
    else:
        loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))        
    return loss_adent

def adentropy_ori(F1, feat, lamda,param_fc2=None, eta=1.0):
    if param_fc2 is None:
        out_t1,_ = F1(feat, reverse=True, eta=eta)
    else:
        out_t1,_ = F1(feat, param_fc2,reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent
    
class CrossEntropyKLD(object):
    def __init__(self, num_class=126, mr_weight_kld=0.1):
        self.num_class = num_class 
        self.mr_weight_kld = mr_weight_kld

    def __call__(self, pred, label, mask):
        # valid_reg_num = len(label)
        logsoftmax = F.log_softmax(pred, dim=1)

        kld = torch.sum(-logsoftmax/self.num_class, dim=1)
        ce = (F.cross_entropy(pred, label, reduction='none')*mask).mean()
        kld = (self.mr_weight_kld*kld*mask).mean()

        ce_kld = ce + kld

        return ce_kld    