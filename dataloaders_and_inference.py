from torchvision import datasets, transforms
import torch
import numpy as np
from tqdm import tqdm

STL10_MEAN = [x / 255.0 for x in [113.9, 112.2, 103.7]]
STL10_STD = [x / 255.0 for x in [66.4, 65.4, 69.2]]
ImageNet_MEAN = [0.485, 0.456, 0.406]
ImageNet_STD = [0.229, 0.224, 0.225]
PLAYGROUND_MEAN = 3*[0.5]
PLAYGROUND_STD = 3*[0.5]

STL10_PATH = 'Set path to data here'
CIFAR10_PATH = 'Set path to data here'
SVHN_PATH = 'Set path to data here'
CIFAR100_PATH = 'Set path to data here'
IMAGENET_PATH = 'Set path to data here'

def Return_Dataloaders(dataset,batch_size,data_augmentation=False,cutout=False,normalize_data=True,train_set_indicator=None,download=True,test_set_4_training_when_splitting=True):
    # test_set_4_training_when_splitting only affects when  train_set_indicator is not None. Then it dermines whether the indicator corresponds to the training or test set, meaning whether training should be done of a portion of the original training or test sets.
    assert train_set_indicator is None or dataset=='stl10','Unimplemented yet.'
    NEW_TRANSFORMATIONS_SET = True
    # Image Preprocessing
    if dataset in ['svhn','cifar10','cifar100']:
        normalize = transforms.Normalize(mean=PLAYGROUND_MEAN,std=PLAYGROUND_STD)#When using the pretrained model from pytorch-playground
    elif dataset == 'stl10':
        normalize = transforms.Normalize(mean=STL10_MEAN,std=STL10_STD)
    elif dataset=='ImageNet':
        normalize = transforms.Normalize(mean=ImageNet_MEAN,std=ImageNet_STD)
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    if data_augmentation=='flipOnly':
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    elif data_augmentation=='cropOnly':
        assert dataset=='stl10','Unsupported yet'
        train_transform.transforms.append(transforms.RandomCrop(96, padding=12))
    elif data_augmentation:
        if dataset == 'stl10':
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.RandomCrop(96, padding=12))
        else:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.ToTensor())
    if normalize_data:
        train_transform.transforms.append(normalize)
    if dataset=='ImageNet':
        test_transform = transforms.Compose([transforms.Scale(256),transforms.CenterCrop(224),transforms.ToTensor()]+([normalize] if normalize_data else []))
    else:
        test_transform = transforms.Compose([transforms.ToTensor()]+([normalize] if normalize_data else []))

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=CIFAR10_PATH,train=True,transform=train_transform,download=download)
        test_dataset = datasets.CIFAR10(root=CIFAR10_PATH,train=False,transform=test_transform,download=download)
    elif dataset == 'stl10':
        test_dataset = datasets.STL10(root=STL10_PATH, split='test', transform=test_transform, download=download)
        if train_set_indicator is None:
            train_dataset = datasets.STL10(root=STL10_PATH,split='train',transform=train_transform,download=download)
        else:
            if test_set_4_training_when_splitting:
                train_dataset = datasets.STL10(root=STL10_PATH,split='test',transform=train_transform,download=download)
                test_dataset = torch.utils.data.Subset(test_dataset,np.argwhere(np.logical_not(train_set_indicator)).reshape([-1]))
            else:
                train_dataset = datasets.STL10(root=STL10_PATH,split='train',transform=train_transform,download=download)
    elif dataset=='ImageNet':
        assert train_set_indicator is None
        test_dataset = datasets.ImageNet(root=IMAGENET_PATH,split='val',transform=test_transform,download=download)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=CIFAR100_PATH,train=True,transform=train_transform,download=download)

        test_dataset = datasets.CIFAR100(root=CIFAR100_PATH,train=False,transform=test_transform,download=download)
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN(root=SVHN_PATH,split='train',transform=train_transform,download=download)

        extra_dataset = datasets.SVHN(root=SVHN_PATH,split='extra',transform=train_transform,download=download)

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

        test_dataset = datasets.SVHN(root=SVHN_PATH,split='test',transform=test_transform,download=download)

    NUM_WORKERS = 2 # Changed from 2 to 0, after process used to just hang
    if train_set_indicator is None:
        if dataset=='ImageNet':
            train_loader = None
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=NUM_WORKERS)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, num_workers=NUM_WORKERS,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(np.argwhere(train_set_indicator).reshape([-1])))
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True, num_workers=NUM_WORKERS)#\
    return train_loader,test_loader

def model_inference(loader,model,dataset,transformer=None,return_outputs_dict=False):
    assert dataset in ['stl10','ImageNet','svhn','cifar10','cifar100']
    images_mean = STL10_MEAN if dataset=='stl10' else ImageNet_MEAN if dataset=='ImageNet' else PLAYGROUND_MEAN
    images_STD = STL10_STD if dataset=='stl10' else ImageNet_STD if dataset=='ImageNet' else PLAYGROUND_STD
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct,correct_top5 = 0.,0.
    total = 0.
    if return_outputs_dict:
        outputs_dict = {'logits':[],'error_indicator':[],'top5_error_indicator':[],'GT_labels':[]}
    already_normalized = any([isinstance(t,transforms.transforms.Normalize) for t in loader.dataset.transforms.transform.transforms])
    assert transformer is None or (not already_normalized),'Should not apply transformer on already normalized images.'
    progress_bar = tqdm(loader)
    for images, labels in progress_bar:
        progress_bar.set_description('Testing')
        if dataset=='svhn':
            # SVHN labels are from 1 to 10, not 0 to 9, so subtract 1
            labels = labels.type_as(torch.LongTensor()).view(-1) - 1
        images,labels = images.data.cpu().numpy().transpose((0,2,3,1)),labels.data.cpu().numpy()
        if transformer is not None:
            images,labels = transformer.TransformImages(images=images,labels=labels)
        if not already_normalized:
            images = (images-np.reshape(images_mean,[1,1,1,3]))/np.reshape(images_STD,[1,1,1,3])
        images, labels = torch.from_numpy(images.transpose((0,3,1,2))).type(torch.FloatTensor),torch.from_numpy(labels).type(torch.int64)
        with torch.no_grad():
            pred = model(images)
        if transformer is None:
            cur_logits = 1*pred
        else:
            pred,cur_logits = transformer.Process_Logits(input_logits=pred,reorder_logits=False)
            labels = transformer.Process_NonLogits(labels)
        top_5_predicted = torch.argsort(pred,dim=1,descending=True)[:,:5]
        error_indicator = (top_5_predicted[:,0] != labels.to(top_5_predicted.device)).data.cpu().numpy()
        if return_outputs_dict:
            outputs_dict['logits'].append(cur_logits.data.cpu().numpy())
            outputs_dict['error_indicator'].append(error_indicator)
            outputs_dict['top5_error_indicator'].append(torch.all(top_5_predicted != labels.view(-1,1).to(top_5_predicted.device),dim=1).data.cpu().numpy())
            outputs_dict['GT_labels'].append(labels.data.cpu().numpy())
        total += labels.size(0)
        correct += (np.logical_not(error_indicator)).sum()

    model.train()
    if return_outputs_dict:
        for key in outputs_dict.keys():
            outputs_dict[key] = np.concatenate(outputs_dict[key],0)
        return outputs_dict
    else:
        val_acc = correct.item() / total
        return val_acc
