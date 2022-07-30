from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from model.detr import TransformerHELayer

#class GradReverse(Function):
    #def __init__(self, lambd):
        #self.lambd = lambd

    #def forward(self, x):
        #return x.view_as(x)

    #def backward(self, grad_output):
        #return (grad_output * -self.lambd)
       
class MultiLinearClassifier(nn.Module):
    def __init__(self, n_clf, feat_dim, n_way, sum_log=True, permute=False, shapes=None, loss_type="softmax"):
        super(MultiLinearClassifier, self).__init__()
        self.n_clf = n_clf#8
        self.feat_dim = int(feat_dim/n_clf)#128
        self.n_way = n_way#5
        self.sum_log = sum_log#true
        #print(self.n_clf,self.feat_dim)
        #print(self.n_way)
        #print(self.sum_log)
        #print(loss_type,permute)
        #exit()
        #exit()        
        self.softmax = nn.Softmax(dim=2)
        self.permute = permute
        self.shapes = shapes
        if self.permute:
            self.clfs = nn.ModuleList([self.create_clf(loss_type, shapes[i], n_way).cuda() for i in range(n_clf)])
        else:
            self.clfs = nn.ModuleList([self.create_clf(loss_type, self.feat_dim, n_way).cuda() for i in range(n_clf)])

    def create_clf(self, loss_type, in_dim, out_dim):
        if loss_type == "softmax":
            return nn.Linear(in_dim, out_dim)
        elif loss_type == "dist":
            return distLinear(in_dim, out_dim, True)

    def forward(self, X):
        # X is n_clf * N * feat_dim
        #print(X.shape)#(8 4 128)
        #exit()
        if self.permute:
            N = X[0].shape[0]
        else:
            N = X.shape[1]
        resp = torch.zeros(self.n_clf, N, self.n_way).cuda()
        for i in range(self.n_clf):
            resp[i] = self.clfs[i](X[i])
        #print(resp.shape)#(8 4 5)
        proba = self.softmax(resp)
        #print(proba.shape,self.sum_log)#(8 4 5) true
        #exit()
        if self.sum_log:
            log_proba = torch.log(proba)
            sum_log_proba = log_proba.mean(dim=0)
            scores = sum_log_proba
        else:
            mean_proba = proba.mean(dim=0)
            log_proba = torch.log(mean_proba)
            scores = log_proba
        return scores        
class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()

def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=False)
        model_alexnet.load_state_dict(torch.load("/gdata2/tuky/DA/pretrain_model/alexnet-owt-4df8aa71.pth"))
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x,pretrain_model=None,data=None, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out,x_out,x
class Predictor_ori(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_ori, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out

class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        x_f1=x
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out,x_f1
        
class Predictor_deep_reconstruction(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep_reconstruction, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        x_f1=x
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out,x_f1
    def classifier(self,x):
        x_out = self.fc2(x) / self.temp 
        return x_out

        
class Predictor_deep_ifsl(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep_ifsl, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.n_splits=8
        self.feat_dim=512
        self.preprocess_after_split="l2n"
        self.clf = MultiLinearClassifier(self.n_splits, self.feat_dim, num_class, True, False, None, "softmax").cuda()
        self.num_class = num_class
        self.temp = temp
    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        # Sequentially cut into n_splits parts
        #print(self.feat_dim,self.n_splits)
        split_dim = int(self.feat_dim / self.n_splits)
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
        #print(x.shape)
        #exit()
        #print(preprocess_method)
        #exit()
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            if preprocess_method == "l2n":
                split_features[i] = F.normalize(x[:, start_idx:end_idx])
            elif preprocess_method == "none":
                split_features[i] = x[:, start_idx:end_idx]
            elif preprocess_method == "cl2n":
                split_features[i] = F.normalize(x[:, start_idx:end_idx] - center[:, start_idx:end_idx])
            '''
            if preprocess:
                if preprocess_method != "dl2n":
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
            '''
        return split_features 
    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        x_f1=x
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        split_support = self.get_split_features(x, preprocess=True, center=None,
                                                preprocess_method=self.preprocess_after_split)
        #print(split_support.shape)
        #exit()
        logits = self.clf(split_support) / (self.temp*1.5)                                               
        x_out = self.fc2(x) / self.temp
        return x_out,logits,x_f1
        
class Predictor_deep_ifsl_reconstruction(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep_ifsl_reconstruction, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        
        #self.fc3 = nn.Linear(512, num_class, bias=False)
        self.n_splits=8
           
        self.preprocess_after_split="l2n"
        self.concat=False
        self.logits=False
        self.transformer=False


            
        if self.logits:
            self.feat_dim=num_class
            self.n_splits=2
        else:
            self.feat_dim=512
        if self.transformer:
            self.transformer=TransformerHELayer(d_model=self.feat_dim, dim_feedforward=2048)            
        if self.concat:
            if self.logits:
                self.fc2 = nn.Linear(512, num_class, bias=False)
                self.fc3_con = nn.Linear(num_class, num_class, bias=False)
            else:
                self.fc2_con = nn.Linear(512, num_class, bias=False)
        else:
            self.fc2 = nn.Linear(512, num_class, bias=False)
        if self.concat:
            self.clf1 = MultiLinearClassifier(self.n_splits, self.feat_dim, num_class, True, False, None, "softmax").cuda()
            #self.clf1  
        else:
            self.clf = MultiLinearClassifier(self.n_splits, self.feat_dim, num_class, True, False, None, "softmax").cuda()
        self.num_class = num_class
        self.temp = temp
    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        # Sequentially cut into n_splits parts
        #print(self.feat_dim,self.n_splits)
        split_dim = int(self.feat_dim / self.n_splits)
        if False:
            try:
                split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
            except:
               print(self.n_splits,'l')
               print(x.shape[0])
               print(split_dim)
               print('error')
               exit() 
        else:
            split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
        #print(x.shape)
        #exit()
        #print(preprocess_method)
        #exit()
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            if preprocess_method == "l2n":
                split_features[i] = F.normalize(x[:, start_idx:end_idx])
            elif preprocess_method == "none":
                split_features[i] = x[:, start_idx:end_idx]
            elif preprocess_method == "cl2n":
                split_features[i] = F.normalize(x[:, start_idx:end_idx] - center[:, start_idx:end_idx])
            '''
            if preprocess:
                if preprocess_method != "dl2n":
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
            '''
        return split_features 
    def forward(self, x, pretrain_model,image_data,reverse=False, eta=0.1):
        x = self.fc1(x)
        x_f1=x
        #if self.concat:
           # x_normal,reconstuction_feature=pretrain_model.get_transform_features(x,image_data)

        if reverse:
            x = grad_reverse(x, eta)
        #x = F.normalize(x)
        if self.concat:
            if self.logits:
                x = F.normalize(x)
                x= self.fc2(x) / self.temp
                x_normal,reconstuction_feature=pretrain_model.get_transform_features(x,image_data)        
                split_support = self.get_split_features(x_normal, preprocess=True, center=None,
                                                preprocess_method=self.preprocess_after_split)            
            else:
                x_normal,reconstuction_feature=pretrain_model.get_transform_features(x,image_data)        
                split_support = self.get_split_features(x_normal, preprocess=True, center=None,
                                                preprocess_method=self.preprocess_after_split)
        else:
            x = F.normalize(x)
            split_support = self.get_split_features(x, preprocess=True, center=None,
                                                preprocess_method=self.preprocess_after_split)            
        #print(split_support.shape)
        #print(reconstuction_feature.shape)
        #exit()
        if self.concat:
            #split_support=torch.cat((split_support, reconstuction_feature), dim=2)      
            split_support=split_support+reconstuction_feature     
            logits = self.clf1(split_support) / (self.temp*1.5) 
        else:
            logits = self.clf(split_support) / (self.temp*1.5) 

        if self.concat:
            if self.logits:
                x_concat=split_support.transpose(0,1).contiguous().reshape(split_support.shape[1],-1)
                x_concat=F.normalize(x_concat)
                x_out = self.fc3_con(x_concat) / self.temp
            else:
                x_concat=split_support.transpose(0,1).contiguous().reshape(split_support.shape[1],-1)
                x_concat=F.normalize(x_concat)
                x_out = self.fc2_con(x_concat) / self.temp                
        else:
            x_out = self.fc2(x) / self.temp
            #x_out_3 = self.fc3(x) / self.temp
        return x_out,logits,x#,x_out_3        
class Predictor_deep_transformer(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep_transformer, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        #self.fc2_source = nn.Linear(512, num_class, bias=False)
        #self.fc2_target = nn.Linear(512, num_class, bias=False)         
        self.fc2_transformer = nn.Linear(512, num_class, bias=False)
        self.fc2_fuse = nn.Linear(1024, num_class, bias=False)        
        self.num_class = num_class
        self.temp = temp
        self.transformer=TransformerHELayer(d_model=512, dim_feedforward=2048)
        #self.transformer=TransformerHELayer(d_model=512, dim_feedforward=2048)
        #for key in self.transformer.named_parameters():
            #print(key)
            #exit()
    def forward(self, x,f_weight=None,target=None, reverse=False, eta=0.1):
        if f_weight is not None:
            x_transformer=self.transformer(query=x.unsqueeze(0), key=f_weight.unsqueeze(0), value=f_weight.unsqueeze(0),target=target)
            x_transformer = self.fc1(x_transformer.squeeze(0)) / self.temp    

        x = self.fc1(x)
        x_ori=x
        x_transformer_ori=x_transformer
        if reverse:
            x = grad_reverse(x, eta)
            if f_weight is not None:
                x_transformer = grad_reverse(x_transformer, eta)
        #print(x.shape,x_transformer.shape)
        #print(torch.cat([x,x_transformer],1).shape)
        #exit()
 
        if f_weight is not None:
            x_fuse=F.normalize(torch.cat([x,x_transformer],1))    
            x_transformer = F.normalize(x_transformer)
        x = F.normalize(x)
        x_no_reverse = F.normalize(x_ori)   
        #print(x_no_reverse.shape)
        #exit()
        x_source=x_no_reverse[:24,:]
        x_target=x_no_reverse[24:,:] 
        #print(x_source.shape)
        #print(x_targe.shape)
        #exit()
        #print(x.shape,x_transformer_out.shape)
        #exit()
        #print(x.unsqueeze(0).shape,f_weight.unsqueeze(0).shape)
        #if f_weight is not None:
            #x_transformer=self.transformer(query=x.unsqueeze(0), key=f_weight.unsqueeze(0), value=f_weight.unsqueeze(0),target=target)
            #x_transformer_out = self.fc2(x_transformer_out.squeeze(0)) / self.temp
            #return x_transformer_out,x_transformer.squeeze(0)
        #print(x_transformer.shape)
        #exit()
        #print('not use transformer')
        #x_fuse=F.normalize(0.5*(x+x_transformer))
        x_out = self.fc2(x) / self.temp
        #x_source_out = self.fc2_source(x_source) / self.temp 
        #x_targe_out = self.fc2_target(x_target) / self.temp
        #x_source_target_out = self.fc2_source(x_target) / self.temp 
        #x_targe_source_out = self.fc2_target(x_source) / self.temp          
        if f_weight is not None:        
            x_transformer_out = self.fc2_transformer(x_transformer) / self.temp
            x_fuse_out = self.fc2_fuse(x_fuse) / self.temp
        if f_weight is not None:            
            return x_out,x_ori,x_transformer_out,x_transformer_ori,x_fuse_out,x_fuse#,x_source_out,x_targe_out,x_source_target_out,x_targe_source_out      
        else:
            return x_out,x      
class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out
