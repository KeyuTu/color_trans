import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor
from model.basenet import Predictor_deep_reconstruction as Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset_eposide import return_dataset
#import tqdm
#from dataloader.dataset_loader import DatasetLoader as Dataset
from torch.utils.data import DataLoader


class Pretrain():
    def __init__(self, args,G_model_dir, F_model_dir,net,num_class,temp, init_model=True,use_logits=False):
        #self.dataset = dataset
        #self.method = method
        #self.model_name = model
        self.init_model = init_model
        self.G_model_dir=G_model_dir
        self.F_model_dir=F_model_dir
        self.net=net
        self.num_class=num_class
        #self.num_classes = 351
        #self.inc=inc
        self.temp=temp
        self.batch=1
        self.n_splits=8
        self.softmax = torch.nn.Softmax(dim=1)
        self.source_name=args.source
        self.target_name=args.target
        #print(args)
        #exit()
        self.args=args
        self.normalize_before_center=True
        #print(args)
        #exit()
        im_data_t = torch.FloatTensor(1)
        gt_labels_t = torch.LongTensor(1)

        im_data_t = im_data_t.cuda()
        gt_labels_t = gt_labels_t.cuda()

        self.im_data_t = Variable(im_data_t)
        self.gt_labels_t = Variable(gt_labels_t)
        
        self.source_loader, self.target_loader, self.target_loader_unl, self.target_loader_val, \
                    self.target_loader_test, class_list = return_dataset(args,False)
        #print('lkk','enter pretrain')
        #exit()
        #print(self.method)
        #exit()
        #if self.method in ["simpleshot", "simpleshotwide"]:
        self.simpleshot_init()
        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.F1.parameters():
            param.requires_grad = False   
        #print('lll')
        #exit()
        self.logits=use_logits
        if self.logits:
            self.n_splits=2
        self.pretrain_features=self.get_pretrained_class_mean(True)
        #print(self.pretrain_features.shape)
        #exit()
        self.pretrain_features = torch.from_numpy(self.pretrain_features).cuda()
        self.pretrain_features_mean = self.pretrain_features.mean(dim=0)
        #print(self.pretrain_features.shape)
        #print(self.pretrain_features_mean.shape)
        #exit()
    def simpleshot_init(self):
        simple_shot_dir = "/home/yfchen/ljj_code/ifsl-master/simpleshot/models/results/"#mini/softmax/"
        #model_name = self.model_name.lower()
        #print(model_name)
        if self.net == 'resnet34':
            self.G = resnet34()
            
            #G.load_state_dict(torch.load("./record/office_home/MME/G_iter_model_MME_Art_to_Clipart_best_val.pth.tar"))
            inc = 512
        elif self.net == "alexnet":
            self.G = AlexNetBase()
            inc = 4096
        elif self.net == "vgg":
            self.G = VGGBase()
            inc = 4096
        else:
            raise ValueError('Model cannot be recognized.')
        if "resnet" in self.net:
            self.F1 = Predictor_deep(num_class=self.num_class,
                                inc=inc)
            #F1.load_state_dict(torch.load("./record/office_home/MME/F1_iter_model_MME_Art_to_Clipart_best_val.pth.tar"))                    
        else:
            self.F1 = Predictor(num_class=self.num_class, inc=inc,
                           temp=args.T)            
        
        #if self.dataset == "tiered":
            #self.num_classes = 351
        #else:
            #self.num_classes = 64
        #self.image_size = 84
        #if model_name == "wideres":
            #self.batch_size = 128
            #self.feat_dim = 640
        #else:
            #self.batch_size = 128
            #self.feat_dim = 512

        #if model_name == "resnet10":
            #model_abbr = "resnet"
        #elif model_name == "wideres":
            #model_abbr = "wrn"
        #if self.dataset == "cross" or self.dataset == "miniImagenet":
            #model_dir = os.path.join(simple_shot_dir, "mini/softmax", model_name, "model_best.pth.tar")
            # model_dir = "/model/1154027137/ifsl_mini_pretrain/ifsl_mini/ss_" + model_abbr + "_mini.tar"
        #elif self.dataset == "tiered":
            #model_dir = os.path.join(simple_shot_dir, "tiered", model_name, "model_best.pth.tar")
            # model_dir = "/model/1154027137/ifsl_tiered_pretrain/ifsl_tiered/ss_" + model_abbr + "_tiered.tar"

        def remove_module_from_param_name(params_name_str):
            split_str = params_name_str.split(".")[1:]
            params_name_str = ".".join(split_str)
            return params_name_str

        if self.init_model:
            #if model_name == "resnet10":
                #model = ResNet10(num_classes=self.num_classes, remove_linear=False)
            #elif model_name == "wideres":
                #model = WideRes28(num_classes=self.num_classes, remove_linear=False)

            G_checkpoint = torch.load(self.G_model_dir)
            self.G.load_state_dict(G_checkpoint)
            
            
            F_checkpoint = torch.load(self.F_model_dir)
            self.F1.load_state_dict(F_checkpoint,strict=False)            
            #print('load strict is True')
            #exit()
            #model_dict = model.state_dict()
            #model_params = checkpoint['state_dict']
            #model_params = {remove_module_from_param_name(k): v for k, v in model_params.items()}
            #model_params = {k: v for k, v in model_params.items() if k in model_dict}
            #model_dict.update(model_params)
            #model.load_state_dict(model_dict)
            self.F1=self.F1.cuda()
            self.G=self.G.cuda()
            self.G.eval()
            self.F1.eval()            
            #model.eval()
            #self.model = model

    def nn_preprocess(self, data, center=None, preprocessing="l2n"):
        #print(preprocessing)
        #exit()
        #print(data.shape,center.shape)
        #exit()
        if preprocessing == "none":
            return data
        elif preprocessing == "l2n":
            return F.normalize(data)
        elif preprocessing == "cl2n":
            if self.normalize_before_center:
                data = F.normalize(data)
            #print(data.shape)
            #print(center.shape)
            #exit()
            centered_data = data - center
            return F.normalize(centered_data)

    def get_restuction_features(self, embedding,image_data,use_ori=False):
        feat_dim = int(embedding.shape[1] / self.n_splits)
        d_feature = torch.zeros(self.n_splits, embedding.shape[0], feat_dim).cuda()
        with torch.no_grad():
            if use_ori:
                G_featutes=self.G(image_data)
                F_logits,F_features = self.F1(G_featutes) 
                  
                prob = self.softmax(F_logits)    
            else:
                #G_featutes=embedding
                if self.logits:            
                    prob = self.softmax(embedding)                
                else:
                    G_featutes = F.normalize(embedding)
                    F_logits = self.F1.classifier(G_featutes)             
                    prob = self.softmax(F_logits)
            for i in range(self.n_splits):
                start = i * feat_dim
                stop = start + feat_dim
                #print(self.pretrain_features.shape)
                #exit()
                d_feature[i] = torch.mm(prob, self.pretrain_features)[:, start:stop] # d_feature意义何在？
            
        return d_feature            

    def get_transform_features(self, embedding, image_data):
        # print(support.shape, query.shape)#(5,512)(75,512)
        # print(support_ori, query_ori)
        # exit()
        embedding_t = self.get_restuction_features(embedding, image_data)#(8,5,64)#重构特征

        embedding_size = embedding.shape[0]
        if self.logits:
            embedding_t = self.softmax(embedding_t)
        else:
            embedding_t = F.normalize(embedding_t)
        #pmean_embedding = self.pretrain_features_mean.expand((embedding_size, embedding.shape[1]))

        #embedding = self.nn_preprocess(embedding, pmean_embedding, preprocessing="cl2n")#特征归一化然后减均值
        
        #split_support = self.get_split_features(embedding, preprocess=True, center=pmean_embedding,
                                                #preprocess_method=self.preprocess_after_split)

        return embedding, embedding_t
    def get_pretrained_class_mean(self, normalize=False):
        if normalize:
            pre = "norm_"
        else:
            pre = ""
        if self.logits:
            save_folder= "pretrain/%s_logits/%s/%s/" % (self.args.dataset,self.args.net,self.args.num)   
            save_dir = "pretrain/%s_logits/%s/%s/%s%s_%s_%s_mean.npy" % (self.args.dataset,self.args.net,self.args.num,pre, self.source_name, self.target_name, self.net)
        else:
            save_folder= "pretrain/%s/%s/%s/" % (self.args.dataset,self.args.net,self.args.num)   
            save_dir = "pretrain/%s/%s/%s/%s%s_%s_%s_mean.npy" % (self.args.dataset,self.args.net,self.args.num,pre, self.source_name, self.target_name, self.net)            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if os.path.isfile(save_dir):
            features = np.load(save_dir)
        else:
            # normalize = False
            features = self.get_base_means(normalize=normalize)
            np.save(save_dir, features)
        return features

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def get_base_means(self, normalize=False):
        num_classes = self.num_class
        # save_dir = "pretrain/%s%s_%s_%s_mean.npy" % (pre, self.dataset, self.method, self.model_name)
        # ensure_path("pretrain")
        # self.means_save_dir = osp.join("logs/means", "%s_%s.npy" % (self.args.dataset, str(is_cosine_feature)))
        # Load pretrain set
        num_workers = 8
        #if self.args.debug:
            #num_workers = 0
        #self.trainset = Dataset('train', self.args, dataset=self.dataset, train_aug=False)
        #self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.args.pre_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if self.logits:
            means = torch.zeros(num_classes, num_classes).cuda()
        else:
            means = torch.zeros(num_classes, 512).cuda()
        counts = torch.zeros(num_classes).cuda()
        for epoch in range(1):
            counter=0
            #tqdm_gen = tqdm.tqdm(self.train_loader)
            for batch_idx, data_t in enumerate(self.target_loader_test):
                with torch.no_grad():
                    self.im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                    self.gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
                    #data, _ = [_.cuda() for _ in batch]
                    #label = batch[1]
                #with torch.no_grad():
                    G_features = self.G(self.im_data_t)
                    F_logits,F_features = self.F1(G_features)
                    #print(F_logits.shape)
                    #exit()
                    #print(F_logits.max(1)[1])
                    #print(self.gt_labels_t)
                    #if self.gt_labels_t==F_logits.max(1)[1]:
                        #counter=counter+1
                    #exit(0)
                    #print(normalize)
                    #exit()
                    if self.logits:
                        F_features=F_logits
                        if normalize:
                            F_features = self.softmax(F_features)
                            print('using logits softmax')
                    else:  
                        if normalize:
                            F_features = F.normalize(F_features)                        
                        #print('normal')
                    #print(F_features.shape)
                    #exit()
                    #print(self.gt_labels_t)
                    #exit()
                    for j in range(F_features.shape[0]):
                        means[int(self.gt_labels_t[j].cpu().numpy())] += F_features[j]
                        counts[int(self.gt_labels_t[j].cpu().numpy())] += 1
            #print(counter,'/',len(self.target_loader_test))
            #exit()
        counts = counts.unsqueeze(1).expand_as(means)
        #print(counts.shape)
        #exit()
        means = means / counts
        means_np = means.cpu().detach().numpy()
        # np.save(save_dir, means_np)
        return means_np