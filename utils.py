import os
import csv
import re
import numpy as np
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import torch.nn as nn
import Transformations
from shutil import copyfile
import copy

STL10_MEAN = [x / 255.0 for x in [113.9, 112.2, 103.7]]
STL10_STD = [x / 255.0 for x in [66.4, 65.4, 69.2]]
ImageNet_MEAN = [0.485, 0.456, 0.406]
ImageNet_STD = [0.229, 0.224, 0.225]
PLAYGROUND_MEAN = 3*[0.5]
PLAYGROUND_STD = 3*[0.5]

STL10_PATH = '~/data/Databases/stl10/'
CIFAR10_PATH = '~/data/Databases/cifar10/'
SVHN_PATH = '~/data/Databases/SVHN/'
CIFAR100_PATH = '~/data/Databases/CIFAR100/'
IMAGENET_PATH = '~/data/Databases/ImageNet/'
SAVED_LOGITS_FOLDER = '/home/ybahat/data/ErrorDetection/Saved_logits'
SAVED_DRAWN_RANDOMS_FOLDER = 'datasets_partition/'
SAVED_LOGITS_CSV_FILENAME = 'saved_logits_files'

STL10_MODEL_VERSIONS = ['Seed0_no_augm','Seed0_flipOnly','Seed0_flipOnlyHalfData','ensemble','Seed0_untrained','Seed0_no_flip']+['Seed%d'%(i+1) for i in range(4)]


def Locate_Saved_logits(csv_file_path,saved_logits_file_name,model_version,desired_transformations,use_all_saved_transformations=None):
    logits_files_2_load = []
    found_saved_Ts = np.zeros([len(desired_transformations)]).astype(int)
    if os.path.exists(csv_file_path):
        with open(csv_file_path,'r') as csvfile:
            for row in csv.reader(csvfile):
                if any([re.search(name+'_(\d)+\.npy',row[0]) for name in ([(saved_logits_file_name+identifier) for identifier in (['']+['_Seed%d'%(i+1) for i in range(4)])] if model_version=='ensemble' else [saved_logits_file_name])]):
                    saved_Ts = row[1].split(',')
                    if re.search('_Seed\d_',row[0]) is not None and model_version=='ensemble':
                        seed_num = int(re.search('(?<=_Seed)\d',row[0]).group(0))
                        saved_Ts = ['model_seed%d_'%(seed_num)+t for t in saved_Ts]
                    if use_all_saved_transformations is not None:
                        Ts_2_add = [T for T in saved_Ts if T not in desired_transformations and
                                    ((any([re.match(p,T) is not None for p in use_all_saved_transformations]) and 'model_seed' not in T))]
                        found_saved_Ts = np.concatenate([found_saved_Ts,np.zeros([len(Ts_2_add)]).astype(int)])
                        desired_transformations += Ts_2_add
                    if re.search('_Seed\d_',row[0]) is not None and model_version=='ensemble':
                        existing_desired_Ts = [('model_seed%d_'%(seed_num)+T) in saved_Ts for T in desired_transformations]
                    else:
                        existing_desired_Ts = [T in saved_Ts for T in desired_transformations]
                    if any(existing_desired_Ts):
                        logits_files_2_load.append(row[0])
                        found_saved_Ts[np.argwhere(existing_desired_Ts)] = len(logits_files_2_load)
                        print('Using saved logits from file %s'%(logits_files_2_load[-1]))
                        if all(found_saved_Ts>0) and use_all_saved_transformations is None:
                            break
    return logits_files_2_load,found_saved_Ts

def Load_Saved_Logits(logits_files_2_load,found_saved_Ts,desired_transformations,num_classes,models_ensemble_mode=False):
    classifier_output_dict = None
    if len(logits_files_2_load)>0:
        loaded_desired_Ts = np.zeros([len(desired_transformations)]).astype(np.bool)
        unique_files_2_load = sorted([i for i in list(set(list(found_saved_Ts))) if i>0],key=lambda x:logits_files_2_load[x-1]) #Removing the invalid 0 index and sorting as a hack to prevent loading model_seed files first
        for file_num in unique_files_2_load:
            filename_2_load = logits_files_2_load[file_num-1]
            loaded_dict = np.load(os.path.join(SAVED_LOGITS_FOLDER,filename_2_load),allow_pickle=True).item()
            saved_Ts = ['+'.join(T) for T in loaded_dict['Transformations']]
            assert len(saved_Ts)==((loaded_dict['logits'].shape[1]//num_classes)-1),'Saved logits vector length does not match num of saved transformations'
            if re.search('_Seed\d_',filename_2_load) is not None and models_ensemble_mode:
                seed_num = int(re.search('(?<=_Seed)\d',filename_2_load).group(0))
                saved_Ts = ['model_seed%d_'%(seed_num)+t for t in saved_Ts]
            def transformations_match(saved_T,desired_T):
                return saved_T==desired_T
            corresponding_T_indexes = [(desired_T_index,saved_T_index) for saved_T_index,T in enumerate(saved_Ts) for desired_T_index,desired_T in enumerate(desired_transformations) if transformations_match(T,desired_T)]
            if classifier_output_dict is None:
                classifier_output_dict = copy.deepcopy(loaded_dict)
                classifier_output_dict['logits'] = np.zeros(shape=[loaded_dict['logits'].shape[0],(1+len(desired_transformations))*num_classes])
                assert not any(['model_seed' in t for t in saved_Ts]),'Should not use the file containing logits computed by differently seeded models as source for original logits, as those are set to zero there. Solve this if it happens'
                classifier_output_dict['logits'][:,:num_classes] = loaded_dict['logits'][:,:num_classes]
            for desired_T_index,saved_T_index in corresponding_T_indexes:
                if loaded_desired_Ts[desired_T_index]:
                    continue
                loaded_desired_Ts[desired_T_index] = True
                classifier_output_dict['logits'][:,(desired_T_index+1)*num_classes:(desired_T_index+2)*num_classes] = loaded_dict['logits'][:,(saved_T_index+1)*num_classes:(saved_T_index+2)*num_classes]
            if np.all(loaded_desired_Ts):
                break
    return classifier_output_dict

def Return_Dataloaders(dataset,batch_size,data_augmentation=False,normalize_data=True,download=True):
    # Image Preprocessing
    if dataset in ['svhn','cifar10','cifar100']:
        normalize = transforms.Normalize(mean=PLAYGROUND_MEAN,std=PLAYGROUND_STD)#When using the pretrained model from pytorch-playground
    elif dataset == 'stl10':
        normalize = transforms.Normalize(mean=STL10_MEAN,std=STL10_STD)
    elif dataset=='ImageNet':
        normalize = transforms.Normalize(mean=ImageNet_MEAN,std=ImageNet_STD)

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
        train_dataset = datasets.STL10(root=STL10_PATH,split='train',transform=train_transform,download=download)
    elif dataset=='ImageNet':
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
    if dataset=='ImageNet':
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=NUM_WORKERS)
    return train_loader,test_loader

def Model_Inference(loader,model,dataset,transformer=None,return_outputs_dict=False):
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
    model_device = next(model.parameters()).device
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
        images, labels = torch.from_numpy(images.transpose((0,3,1,2))).type(torch.FloatTensor).to(model_device),torch.from_numpy(labels).type(torch.int64).to(model_device)
        with torch.no_grad():
            pred = model(images)
        if transformer is None:
            cur_logits = 1*pred
        else:
            pred,cur_logits = transformer.Process_Logits(input_logits=pred,reorder_logits=False)
            labels = transformer.Process_NonLogits(labels.cpu())
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

def Infer_Logits(classifier_output_dict,desired_Ts,found_saved_Ts,used_classifier,model_version,used_dataset,num_classes,saved_logits_file_name):
    SAVE_LOGITS_2_FILE = True
    remaining_transformations = [T for i,T in enumerate([T.split('+') for T in desired_Ts]) if not found_saved_Ts[i]]
    from STL10 import WideResNet_model
    if used_classifier == 'ResNet18':
        from torchvision.models import resnet18
        model = nn.DataParallel(resnet18(pretrained=True),device_ids=[0]).cuda()
    elif used_classifier=='ResNext':
        from torchvision.models import resnext101_32x8d
        model = nn.DataParallel(resnext101_32x8d(pretrained=True),device_ids=[0]).cuda()
    elif used_classifier == 'AlexNet':
        from torchvision.models import alexnet
        model = nn.DataParallel(alexnet(pretrained=True),device_ids=[0]).cuda()
    elif used_classifier=='SVHN':
        from SVHN import model as SVHN_model
        model = SVHN_model.svhn(n_channel=32,pretrained=True).cuda()
    elif used_classifier in ['CIFAR10','CIFAR100']:
        from CIFAR import model as CIFAR_model
        model = getattr(CIFAR_model,used_dataset)(n_channel=128,pretrained=True).cuda()
    else:
        model,_ = WideResNet_model.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_'+(model_version if model_version in STL10_MODEL_VERSIONS else 'Seed0'))
    data_loader = Return_Dataloaders(used_dataset,batch_size=int(np.maximum(1,np.floor(32/(1+len(remaining_transformations))))),normalize_data=False)
    data_loader = data_loader[1]
    model_seed_transformations = ['model_seed' in '+'.join(t) for t in remaining_transformations]
    assert all(model_seed_transformations) or np.all(np.logical_not(model_seed_transformations)),'Not supporting mixed logits computation'
    predicted_logits_memory_size = len(data_loader.dataset)*(len(remaining_transformations)+1)*num_classes*4
    if SAVE_LOGITS_2_FILE:
        assert predicted_logits_memory_size<4e9,'Computed logits memory size (%.2f GB, %d transformations) is about to exceed the 4GB data limit'%(predicted_logits_memory_size/1e9,len(remaining_transformations))
    if model_seed_transformations[0]:
        assert model_version in STL10_MODEL_VERSIONS
        resulting_dict,actual_computed_transformations = [],[]
        for seed_num in list(set([int(re.search('(?<=model_seed)\d','+'.join(t)).group(0)) for t in remaining_transformations])):
            model,_ = WideResNet_model.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_Seed%d'%(seed_num))
            this_seed_remaining_Ts = [t for t in remaining_transformations if ('model_seed%d'%(seed_num) in t[0])]
            if len(this_seed_remaining_Ts)==1 and this_seed_remaining_Ts[0][0]=='model_seed%d'%(seed_num):#Only computing logits corresponding to original image:
                resulting_dict.append(Model_Inference(loader=data_loader,model=model,return_outputs_dict=True,dataset=used_dataset))
            else:
                transformer = Transformations.Transformer(transformations=[[t.replace('model_seed%d_'%(seed_num),'') for t in t_chain] for t_chain in this_seed_remaining_Ts],min_pixel_value=0,max_pixel_value=1)
                resulting_dict.append(Model_Inference(loader=data_loader,model=model,transformer=transformer,return_outputs_dict=True,dataset=used_dataset))
                if not this_seed_remaining_Ts[0] == [['model_seed%d'%(seed_num)]]:#Due to the way I later concatenate all logits from all resulting_dicts, I'm going to include in the logits those that correspond to original images too. So I'm changing the transfomration list accordingly.
                    this_seed_remaining_Ts = [['model_seed%d'%(seed_num)]]+this_seed_remaining_Ts
            actual_computed_transformations.append(this_seed_remaining_Ts)
        # Converting the list of dicts to the expected resulting_dict structure. Adding zeros to substitute for the missing "non-transformed" logits, that were not computed.
        resulting_dict = {'GT_labels':resulting_dict[0]['GT_labels'],'logits':np.concatenate([np.zeros([len(data_loader.dataset),num_classes])]+[d['logits'] for d in resulting_dict],-1)}
        remaining_transformations = [t for t_list in actual_computed_transformations for t in t_list]
    else:
        transformer = Transformations.Transformer(transformations=remaining_transformations,min_pixel_value=0,max_pixel_value=1)
        resulting_dict = Model_Inference(loader=data_loader,model=model,transformer=transformer,return_outputs_dict=True,dataset=used_dataset)
    assert len(remaining_transformations)==((resulting_dict['logits'].shape[1]//num_classes)-1),'Saved logits vector length does not match num of saved transformations'
    file_counter = 0
    saved_logits_file_name += '_%d.npy'
    while os.path.isfile(os.path.join(SAVED_LOGITS_FOLDER,saved_logits_file_name%(file_counter))):
        file_counter += 1
    saved_logits_file_name = saved_logits_file_name%(file_counter)
    resulting_dict['Transformations'] = remaining_transformations
    if SAVE_LOGITS_2_FILE:
        np.save(os.path.join(SAVED_LOGITS_FOLDER, saved_logits_file_name), resulting_dict)
        trans_list_4_saving = ','.join(['+'.join(t) for t in sorted(remaining_transformations,key=lambda t:''.join(t))])
        if os.path.exists(os.path.join(SAVED_LOGITS_FOLDER, SAVED_LOGITS_CSV_FILENAME+'.csv')):
            copyfile(os.path.join(SAVED_LOGITS_FOLDER, SAVED_LOGITS_CSV_FILENAME+'.csv'),os.path.join(SAVED_LOGITS_FOLDER, SAVED_LOGITS_CSV_FILENAME+'.bkp'))
        with open(os.path.join(SAVED_LOGITS_FOLDER, SAVED_LOGITS_CSV_FILENAME+'.csv'), 'a') as csvfile:
            csv.writer(csvfile).writerow([saved_logits_file_name,trans_list_4_saving])
    if classifier_output_dict is not None:
        assert model_seed_transformations[0] or np.max(np.abs(resulting_dict['logits'][:,:num_classes]-classifier_output_dict['logits'][:,:num_classes]))<np.percentile(np.abs(resulting_dict['logits']),0.01),'Difference in logits of original images between loaded and computed logits.'
        corresponding_T_indexes = [(desired_T_index,saved_T_index) for saved_T_index,T in enumerate(['+'.join(t) for t in remaining_transformations]) for desired_T_index,desired_T in enumerate(desired_Ts) if T==desired_T]
        for desired_T_index,saved_T_index in corresponding_T_indexes:
            classifier_output_dict['logits'][:,(desired_T_index+1)*num_classes:(desired_T_index+2)*num_classes] = resulting_dict['logits'][:,(saved_T_index+1)*num_classes:(saved_T_index+2)*num_classes]
    else:
        classifier_output_dict = resulting_dict
    return classifier_output_dict

def Subset_Partition(full_dict,dataset,partition):
    assert partition in ['train','test']
    partition_file_name = os.path.join(SAVED_DRAWN_RANDOMS_FOLDER,'%s_hyper_partition.npz'%(dataset))
    TRAIN_SET_PORTION = {'stl10':1/8,'svhn':1/8,'ImageNet':1/50,'cifar10':1/10,'cifar100':1/10}
    total_num_samples = full_dict['logits'].shape[0]
    if not os.path.isfile(partition_file_name):
        print('Creating new partition file %s'%(partition_file_name))
        train_set_indicator = np.zeros([total_num_samples]).astype(np.bool)
        train_set_indicator[np.random.permutation(total_num_samples)[:int(total_num_samples*TRAIN_SET_PORTION[dataset])]] = True
        np.savez(partition_file_name,train_set_indicator=train_set_indicator)
    else:
        train_set_indicator = np.load(partition_file_name)['train_set_indicator']
        assert np.sum(train_set_indicator)==int(total_num_samples*TRAIN_SET_PORTION[dataset])
    dict_2_return = {}
    indicator = train_set_indicator if partition=='train' else np.logical_not(train_set_indicator)
    assert any(indicator),'No %s partition in the %s dataset'%(partition,dataset)
    for key in full_dict.keys():
        if isinstance(full_dict[key],np.ndarray) and full_dict[key].shape[0]==total_num_samples:
            dict_2_return[key] = full_dict[key][indicator,...]
        else:
            dict_2_return[key] = full_dict[key]
    return dict_2_return

def RegExp_In_List(input_string,list_of_interest,other_list=None,exact_match=False):
    found_matches = [re.search(m,input_string) for m in list_of_interest]
    found_matches = [m for m in found_matches if m is not None]
    if exact_match:
        found_matches = [m for m in found_matches if np.diff(m.span())[0]==len(input_string)]
    if len(found_matches)==0:
        return False
    elif other_list is None:
        return True
    else:#A match found but there is another list. Check if it better matches a vlaue in another list:
        alternative_matches = [re.search(m, input_string) for m in other_list]
        alternative_matches = [m for m in alternative_matches if m is not None]
        return max([np.diff(m.span()) for m in alternative_matches])[0]<max([np.diff(m.span()) for m in found_matches])[0] if len(alternative_matches)>0 else True
