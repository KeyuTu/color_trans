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
    #print(len(image_index))
    #for i in range(len(image_index)):
        #print(image_index[i])
        #if i>10:
           # exit()
    #exit()
    image_label_path={}
    label_path_label={}
    old_label=[]
    #print('enter,eposide loss')
    #exit()
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            #print(x,'llll')
            #exit()
            #print(ind)
            #exit()
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            if label in old_label:
                image_label_path[label].append(x.split(' ')[0].strip())
            else:
                image_label_path[label]=[x.split(' ')[0].strip()]
                old_label.append(label)
            #print(image_label_path)    
            #exit()

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
class Imagelists_VISDA_eposide(object):
    def __init__(self, image_list_source,image_list_target,image_set_file_unl, image_set_file_dict,root="./data/multi/",
                 transform=None, target_transform=None, test=False,ids=1):
        self.imgs_train_source, self.labels_train_source,self.image_label_path_train_source  = make_dataset_fromlist_eposide(image_list_source)
        self.imgs_train_target, self.labels_train_target,self.image_label_path_train_target  = make_dataset_fromlist_eposide(image_list_target)
        self.imgs_train_target_un, self.labels_train_target_un,self.image_label_path_train_target_un  = make_dataset_fromlist_eposide(image_set_file_unl)
        #self.imgs_train_target_un, self.labels_train_target_un,self.image_label_path_train_target_un  = make_dataset_fromlist_eposide(image_set_file_unl)         
        #for key in self.image_label_path_train_source.keys():
            #print(self.image_label_path_train_source[key])
            #print('ooooooooooooooooooooooooooooooooooooooo')
            #print(self.image_label_path_train_target[key])
            #print('ppppppppppppppp')
            #exit()
            
        self.imgs = self.imgs_train_source
        self.labels = self.labels_train_source
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.ids=ids
        label_str=[]
        #print(np.array(self.labels_train_source).max())
        for i in range(np.array(self.labels_train_source).max()+1):
            label_str.append(str(i))
        self.label_str=label_str  
        self.label_list_single=list(range(np.array(self.labels_train_source).max()+1))
        self.image_set_file_dict=image_set_file_dict
        #print(np.unique(np.array(self.labels_train_source)))
        #exit()
        #print(self.label_str)
        #print(self.label_list_single)
        #exit()
    def __create_eposide__(self, select_class,image_label_path,transform):
        images = []
        labels = []
        cls = []
        #print(select_class)
        #exit()
        for idx_way in range(len(select_class)):
            class_id=select_class[idx_way]
            select_ins=random.sample(list(range(len(image_label_path[class_id]))), self.ids)
            #print(select_ins)
            #exit()
            paths=[image_label_path[class_id][select_in] for select_in in select_ins]
            #print(paths)
            #exit()
            for path in paths:
                img = self.loader(os.path.join(self.root,path))
                #print(img)
                #exit()
                if transform is not None:
                    img = transform(img)
                    #print(img.shape)
                    #exit()
                images.append(img)
                labels.append(int(class_id))
                cls.append(idx_way)
        images = torch.stack(images, dim=0)
        
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)        
        return  images,labels,cls  
    def __create_eposide__from__dict(self, select_class,image_label_path,image_set_file_dict,transform,k_sel=None):
        images = []
        labels = []
        cls = []
        #print(select_class)
        #exit()
        #print(image_label_path['0'][1])
        #exit()
        for idx_way in range(len(select_class)):
            class_id=select_class[idx_way]
            if k_sel is None:
                num_instanse=int(len(image_set_file_dict[class_id][3])*1.0)
                #print('enter None')
            else:
                num_instanse=int(k_sel)
                #print('enter pesu')
            class_id_pool=image_set_file_dict[class_id][3][:num_instanse]
            select_ins=random.sample(class_id_pool, self.ids)
            #print(select_ins)
            #print(class_id_pool)
            #exit()
            #print(select_ins)
            #exit()
            paths=[image_set_file_dict[class_id][0][select_in] for select_in in select_ins]
            #print(paths)
            #exit()
            for path in paths:
                img = self.loader(os.path.join(self.root,path))
                #print(img)
                #exit()
                if transform is not None:
                    img = transform(img)
                    #print(img.shape)
                    #exit()
                images.append(img)
                labels.append(int(class_id))
                cls.append(idx_way)
        images = torch.stack(images, dim=0)
        
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)        
        return  images,labels,cls          
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
        select_class_num=random.sample(self.label_list_single, 24)
        #print(select_class_num)
        #exit()
        select_class=[self.label_str[i] for i in select_class_num] 
        #print(select_class,'pppppppppppppp')
        #exit()
        images_source,labels_source,cls_source=self.__create_eposide__( select_class[:24],self.image_label_path_train_source,self.transform)
        images_target,labels_target,cls_target=self.__create_eposide__( select_class[:24],self.image_label_path_train_target,self.target_transform) 
        #images_target_un,labels_target_un,cls_target_un=self.__create_eposide__( select_class[:24],self.image_label_path_train_target_un,self.transform)
        images_target_un,labels_target_un,cls_target_un=self.__create_eposide__from__dict( select_class,self.image_label_path_train_target_un,self.image_set_file_dict,self.transform) 
        #print(type(self.image_set_file_dict))
        #exit()
        #images_target_un_peso,labels_target_un_peso,cls_target_un_peso=self.__create_eposide__from__dict( select_class,self.image_label_path_train_target_un,self.image_set_file_dict,self.transform,k_sel=3)        
        #print(cls_target_un_peso)
        #print(labels_target_un_peso)
        #exit()
        #print(self.transform)
        #print(self.target_transform)
        #exit()
        #print(labels_source)
        #print(labels_target)
        #print(images_source.shape)
        #print(images_target.shape)
        #print('enter data_list.py at 157')
        #exit()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            #target = self.target_transform(target)
            target=target
        if not self.test:
            return img, target,images_source,images_target,labels_source,cls_source,images_target_un#,images_target_un_peso,labels_target_un_peso
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)