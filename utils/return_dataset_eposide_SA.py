import os
import numpy as np
import torch
from torchvision import transforms
# from loaders.data_list import Imagelists_VISDA, return_classlist,Imagelists_VISDA_eposide
from loaders.data_list_two_unlabel import Imagelists_VISDA, return_classlist, Imagelists_VISDA_eposide_two_unlabel,Imagelists_VISDA_2_augment
from .randaugment import RandAugmentMC


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args, use_test_loader=True):
    base_path = '/gdata2/tuky/DATASET/txt/%s' % args.dataset
    root = '/gdata2/tuky/DATASET/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    # image_set_file_dict = np.load('./peusde_multi_baseline_alexnet_1shot/'+args.source+'_'+args.target+'_unlabel_3shot_stage1.npy', allow_pickle=True).item()
    image_set_file_dict = np.load('/gdata2/tuky/peusde_label/peusde_'+str(args.dataset)+'_baseline_'+str(args.net)+'_'+str(args.num)+'shot/' +
                                  args.source + '_' + args.target + '_unlabel_'+str(args.num)+'shot_stage1.npy', allow_pickle=True).item()
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),            
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),            
            #transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    k_per = args.k_per
    source_dataset = Imagelists_VISDA_eposide_two_unlabel(k_per, bs, image_set_file_s, image_set_file_t, image_set_file_unl, image_set_file_dict,
                                                          root=root, transform=data_transforms['train'], target_transform=data_transforms['train'],
                                                          unlabel_transform_1=data_transforms['val'], unlabel_transform_2=data_transforms['self'])
    # source_dataset = Imagelists_VISDA_eposide_two_unlabel(k_per, bs, image_set_file_s,image_set_file_t,image_set_file_unl,image_set_file_dict,
                                                          # root=root,transform=data_transforms['train'],target_transform=data_transforms['train'],
                                                          # unlabel_transform_1=data_transforms['val'],unlabel_transform_2=data_transforms['self'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA_2_augment(image_set_file_unl, root=root,transform=data_transforms['val'],
                                                    transform_2=data_transforms['self'])
    if use_test_loader:
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    else:
        target_dataset_test = Imagelists_VISDA(image_set_file_s, root=root,
                                           transform=data_transforms['test'])        
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=1,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(1, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    if use_test_loader:
        target_loader_test = \
            torch.utils.data.DataLoader(target_dataset_test,
                                        batch_size=bs * 2, num_workers=3,
                                        shuffle=True, drop_last=True)
    else:
        target_loader_test = \
            torch.utils.data.DataLoader(target_dataset_test,
                                        batch_size=1, num_workers=3,
                                        shuffle=True, drop_last=True)        
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
