import numpy as np
import os
import os.path
from PIL import Image
import random
import torch


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list
def make_dataset_fromlist_eposide(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]

    image_label_path={}
    label_path_label={}
    old_label=[]

    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            if label in old_label:
                image_label_path[label].append(x.split(' ')[0].strip())
            else:
                image_label_path[label]=[x.split(' ')[0].strip()]
                old_label.append(label)

            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
   
    image_index = image_index[selected_list]

    return image_index, label_list,image_label_path

def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    ## 与return_dataset的函数不匹配
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        # 依次读图片
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
        
class Imagelists_VISDA_2_augment(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, transform_2=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.transform_2 = transform_2        
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        img_2=img
        if self.transform is not None:
            img = self.transform(img)
            img_2 = self.transform_2(img_2)
            images_concat=torch.cat([img,img_2],0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return images_concat, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
        
class Imagelists_VISDA_eposide_two_unlabel(object):
    def __init__(self, k_per, bs, image_list_source, image_list_target, image_set_file_unl,
                 image_set_file_dict, root="./data/multi/", transform=None,
                 target_transform=None, unlabel_transform_1=None, unlabel_transform_2=None, test=False, ids=1):

        self.imgs_train_source, self.labels_train_source, self.image_label_path_train_source \
            = make_dataset_fromlist_eposide(image_list_source)
        self.imgs_train_target, self.labels_train_target, self.image_label_path_train_target \
            = make_dataset_fromlist_eposide(image_list_target)
        self.imgs_train_target_un, self.labels_train_target_un, self.image_label_path_train_target_un \
            = make_dataset_fromlist_eposide(image_set_file_unl)

        self.k_per = k_per
        self.bs = bs
        self.imgs = self.imgs_train_source
        self.labels = self.labels_train_source
        self.transform = transform
        self.target_transform = target_transform
        self.unlabel_transform_1 = unlabel_transform_1
        self.unlabel_transform_2 = unlabel_transform_2
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.ids = ids
        label_str = []

        if 'office_home' in image_list_source:
            num_class = 65
        else:
            num_class = 126
        for i in range(np.array(self.labels_train_source).max()+1):
            label_str.append(str(i))
        self.label_str = label_str
        self.label_list_single = list(range(np.array(self.labels_train_source).max()+1))
        self.image_set_file_dict = image_set_file_dict
        self.class_num = torch.Tensor(range(num_class))
        #i=0
        for key in self.image_set_file_dict:
            self.class_num[int(key)] = len(self.image_set_file_dict[key][0])

    def __create_eposide_un__(self, select_class, image_label_path, transform, use_2_transform=True):
        images = []
        images_2 = []
        labels = []
        cls = []
        for idx_way in range(len(select_class)):
            class_id = select_class[idx_way]
            select_ins = random.sample(list(range(len(image_label_path[class_id]))), self.ids)
            paths=[image_label_path[class_id][select_in] for select_in in select_ins]

            for path in paths:
                img = self.loader(os.path.join(self.root, path))
                if use_2_transform:
                    # img_ori = img
                    img_ori = self.unlabel_transform_2(img)
                    images_2.append(img_ori)
                if transform is not None:
                    img = self.unlabel_transform_1(img)
                images.append(img)
                labels.append(int(class_id))
                cls.append(idx_way)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)
        if use_2_transform:
            images_2 = torch.stack(images_2, dim=0)
            images_concat = torch.cat([images, images_2], 1)
            return images_concat, labels, cls
        return images, labels, cls

    def __create_eposide__(self, select_class, image_label_path, transform, use_2_transform=False):
        images = []
        images_2 = []
        labels = []
        cls = []
        for idx_way in range(len(select_class)):
            class_id = select_class[idx_way]
            select_ins = random.sample(list(range(len(image_label_path[class_id]))), self.ids)
            paths=[image_label_path[class_id][select_in] for select_in in select_ins]

            for path in paths:
                img = self.loader(os.path.join(self.root, path))
                if use_2_transform:
                    # img_ori = img
                    img_ori = self.unlabel_transform_1(img)
                    images_2.append(img_ori)
                if transform is not None:
                    img = transform(img)
                images.append(img)
                labels.append(int(class_id))
                cls.append(idx_way)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)  
        if use_2_transform:            
            images_2 = torch.stack(images_2, dim=0)
            images_concat = torch.cat([images, images_2], 1)
            return images_concat, labels, cls
        return images, labels, cls
    
    def __create_eposide__from__dict(self, select_class,image_label_path,image_set_file_dict,k_sel=None,K_per=1.0):
        images = []
        images_2 = []
        labels = []
        cls = []

        for idx_way in range(len(select_class)):
            class_id=select_class[idx_way]
            if k_sel is None:
                if class_id not in image_set_file_dict:
                    num_instanse = 0
                else:
                    num_instanse = int(len(image_set_file_dict[class_id][3])*K_per)
                    if num_instanse == 0:
                        num_instanse = int(len(image_set_file_dict[class_id][3]))
            else:
                num_instanse = int(k_sel)
            if num_instanse == 0:
                class_id = list(image_set_file_dict.keys())[0]
                class_id_pool = image_set_file_dict[class_id][3][:1]
            else:
                class_id_pool = image_set_file_dict[class_id][3][:num_instanse]
            if False:
                try:
                    select_ins = random.sample(class_id_pool, self.ids)
                except:
                   print(class_id_pool, self.ids, 'l')
                   print(num_instanse, int(len(image_set_file_dict[class_id][3])*K_per))
                   print(len(image_set_file_dict[class_id][3]))
                   print('error')
                   exit()
            else:
                select_ins = random.sample(class_id_pool, self.ids)
            paths = [image_set_file_dict[class_id][0][select_in] for select_in in select_ins]
            for path in paths:
                img = self.loader(os.path.join(self.root, path))
                img_ori = img
                #if transform is not None:
                img = self.unlabel_transform_1(img)
                img_ori=self.unlabel_transform_2(img_ori)
                images.append(img)
                images_2.append(img_ori)
                labels.append(int(class_id))
                cls.append(idx_way)
        if num_instanse==0:
            images = torch.stack(images, dim=0)*0
            images_2 = torch.stack(images_2, dim=0)*0
        else:
            images = torch.stack(images, dim=0)
            images_2 = torch.stack(images_2, dim=0)            
        #print(images.shape)
        #exit()
        images_concat = torch.cat([images, images_2], 1)
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)        
        return  images_concat, labels, cls
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)

        select_class_num=random.sample(self.label_list_single, 2*self.bs)
        select_class_num_junheng=list(torch.multinomial(self.class_num, 2*self.bs, replacement=False).numpy())

        select_class=[self.label_str[i] for i in select_class_num] 
        select_class_junheng=[self.label_str[i] for i in select_class_num_junheng]

        images_source,labels_source,cls_source=self.__create_eposide__( select_class[:self.bs],self.image_label_path_train_source,
                                                                        self.transform,use_2_transform=True)
        images_target,labels_target,cls_target=self.__create_eposide__( select_class[:self.bs],self.image_label_path_train_target,
                                                                        self.target_transform,use_2_transform=False)
        # images_target_un,labels_target_un,cls_target_un=self.__create_eposide__( select_class[:24],self.image_label_path_train_target_un,self.transform)
        images_target_un,labels_target_un,cls_target_un=self.__create_eposide__from__dict(select_class[:self.bs], self.image_label_path_train_target_un,
                                                                                           self.image_set_file_dict, k_sel=None, K_per=self.k_per)
        images_target_un_all,labels_target_un_all,cls_target_un_all=\
            self.__create_eposide__from__dict(select_class_junheng,self.image_label_path_train_target_un,self.image_set_file_dict,k_sel=None,K_per=1.0)

        # images_target_un_all, labels_target_un_all, cls_target_un_all=self.__create_eposide_un__(select_class_junheng,self.image_label_path_train_target_un,
        #                                                                                       self.unlabel_transform_1, use_2_transform=True)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            #target = self.target_transform(target)
            target=target
        if not self.test:
            return img, target,images_source,images_target,labels_source,cls_source,images_target_un,images_target_un_all#,images_target_un_peso,labels_target_un_peso
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)