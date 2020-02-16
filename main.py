import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import Transformations
from utils import *
from confidence_estimation import *
# import example_utils
import csv
import re
# from example_utils import Assign_GPU
import torch
import torch.nn as nn
import copy
from shutil import copyfile
import csv
from tqdm import tqdm

CURVE_TYPE = 'AORC_savelog'#'comparison'#'diff'#'incremental'#'excess','accum_diff','2Dhists','transformations_corr','posteriors','param_search','excess_savelog','num_transformations_performance','AORC'
PAPER_FIG_MODE = False
TOP5 = False
USE_ALL_EXISTING_Ts = None
METHODS_2_COMPARE = ['T_ensemble_MSR','original_MSR','T_MSR_bootstraping','mandelbaum_scores']
TEMPERATURE = {'default':1.}
CLASSIFIER_2_USE = 'AlexNet'#'Wide_ResNet','ELU','ResNet18','ResNext','AlexNet','SVHN','CIFAR10','CIFAR100'
HYPER_PARTITION = 'test'#'train','test'

MODEL_VERSION = 'None'#'None','AT','dist','Seed0_no_augm','flipOnly','flipOnlyHalfData','Seed1','Seed2','ensemble','Seed0_untrained','Seed0_no_flip'
AMIT_MODELS = ['None','AT','dist']
CUTOUT_MODELS = ['Seed0_no_augm','Seed0_flipOnly','Seed0_flipOnlyHalfData','ensemble','Seed0_untrained','Seed0_no_flip']+['Seed%d'%(i+1) for i in range(4)]
EFFECTIVE_POSTERIORS = 'original_posteriors'#'T_ensemble_posteriors','original_posteriors','T_bootstraping_posteriors'
used_dataset = 'ImageNet' if CLASSIFIER_2_USE in ['ResNet18','ResNext','AlexNet'] else 'svhn' if CLASSIFIER_2_USE=='SVHN' else 'cifar10' if CLASSIFIER_2_USE=='CIFAR10' else 'cifar100' if CLASSIFIER_2_USE=='CIFAR100' else 'stl10'
NUM_CLASSES = {'stl10':10,'ImageNet':1000,'svhn':10,'cifar10':10,'cifar100':100}
IM_SIZE = {'stl10':96,'ImageNet':224,'svhn':32,'cifar10':32,'cifar100':32}

if used_dataset=='cifar100':
    # 1 fold, as long as AORC keeps increasing, Ts from best to worse
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['horFlip', 'shift0_2'], ['shift-1_-2'], ['shift0_-3'], ['horFlip', 'shift0_-2'], ['shift1_2'], ['horFlip', 'shift1_-3'], ['shift2_1'], ['horFlip', 'shift4_0'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift2_-3'], ['shift1_3'], ['horFlip', 'shift-3_0'], ['horFlip', 'shift6_0'], ['horFlip', 'shift0_-5'], ['horFlip', 'shift4_4'], ['horFlip', 'shift0_5'], ['horFlip', 'shift-5_-3'], ['shift0_5'], ['horFlip', 'shift7_0'], ['horFlip', 'shift5_-5'], ['rotate3'], ['shift10_10'], ['horFlip', 'shift5_-11'], ['shift9_11'], ['shift-11_-9'], ['vertFlip'], ['horFlip', 'shift11_-11'], ['horFlip', 'shift12_12'], ['shift14_14'], ['horFlip', 'shift14_14']]
elif used_dataset=='cifar10':
    # 1 fold, as long as AORC keeps increasing, Ts from best to worse
    TRANSFORMATIONS_LIST = [['shift1_-1'], ['shift0_-1'], ['shift1_0'], ['horFlip', 'shift0_-1'], ['shift-1_0'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['shift1_1'], ['horFlip', 'shift-2_-1'], ['horFlip', 'shift1_-2'], ['shift-3_1'], ['horFlip', 'shift2_0']]
elif used_dataset=='svhn':
    TRANSFORMATIONS_LIST = [['shift5_0'], ['shift4_0'], ['shift5_4'], ['shift5_2'], ['shift-3_-5'], ['shift-5_-3'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift10_0'], ['shift11_0']]
elif used_dataset=='stl10':
    if CLASSIFIER_2_USE=='ELU':
        # 1 fold, as long as AORC keeps increasing, Ts from best to worse
        TRANSFORMATIONS_LIST = [['gamma12'], ['zoomin95'], ['shift0_-1'], ['horFlip', 'zoomin95'], ['shift0_-5'], ['shift1_0'], ['horFlip', 'shift1_1'], ['shift0_-7'], ['zoomin95', 'shift0_5'], ['horFlip', 'shift0_-1'], ['rotate5'], ['horFlip', 'shift0_-5']]
    else:
        # 1 fold, as long as AORC keeps increasing, Ts from best to worse
        TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift7_-4'], ['shift0_12'], ['horFlip', 'shift-4_-2'], ['shift-9_9'], ['horFlip', 'shift4_1'], ['shift0_14'], ['horFlip', 'shift0_12'], ['horFlip', 'shift12_12'], ['horFlip', 'shift7_0'], ['horFlip', 'shift12_0'], ['gamma8'], ['shift14_14'], ['horFlip', 'shift13_0'], ['horFlip', 'shift14_0'], ['shift0_18'], ['gamma6'], ['horFlip', 'shift19_0'], ['shift0_22'], ['horFlip', 'gamma6'], ['horFlip', 'rotate-1'], ['horFlip', 'rotate1'], ['rotate1'], ['shift22_22'], ['horFlip', 'rotate-7']]
elif used_dataset=='ImageNet':
    TRANSFORMATIONS_LIST = [['gamma9'], ['gamma8'], ['shift0_1'], ['shift0_2'], ['rotate1'], ['rotate4'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['gamma6'], ['horFlip', 'shift1_0'], ['zoomin92'], ['zoomin95', 'gamma8'], ['horFlip', 'shift-1_2'], ['horFlip', 'shift0_5'], ['horFlip', 'shift2_0'], ['rotate5'], ['horFlip', 'shift-5_1'], ['horFlip', 'shift0_10'], ['horFlip', 'shift0_-10'], ['shift0_-10'], ['horFlip', 'zoomin90'], ['rotate7'], ['horFlip', 'rotate-10']]

IAD_FEATURES_FOLDER = '/home/ybahat/data/ErrorDetection/Amit_related/Amit_features/STL10_Amit'#'/home/ybahat/data/Databases/ErrorAndNovelty/STL10_Amit'
FIGURES_FOLDER = '/home/ybahat/PycharmProjects/CFI_private/Error_detection_figures'
PARAMS_SEARCH_FOLDER = '/home/ybahat/PycharmProjects/CFI_private/params_search_figures'
RESULTS_CSV_FOLDER = '/home/ybahat/data/ErrorDetection/saved_results'
# SAVED_LOGITS_FOLDER = '/home/ybahat/data/ErrorDetection/Saved_logits'

assert CLASSIFIER_2_USE in ['Wide_ResNet','ELU','ResNet18','ResNext','AlexNet','SVHN','CIFAR10','CIFAR100']
assert not TOP5 or CLASSIFIER_2_USE in ['ResNet18','ResNext','AlexNet'],'Why usie top-5 accuracy for datasets other than ImageNet?'
num_classes = NUM_CLASSES[used_dataset]
assert all([RegExp_In_List(method,RELIABILITY_SCORE_METHODS+UNCERTAINY_SCORE_METHODS,exact_match=True) for method in METHODS_2_COMPARE]),'Unknown confidence estimation method'

np.random.seed(0)
full_model_name = CLASSIFIER_2_USE+('_%s'%(MODEL_VERSION) if (CLASSIFIER_2_USE=='ELU' or (CLASSIFIER_2_USE=='Wide_ResNet' and MODEL_VERSION in CUTOUT_MODELS)) else '')
if CLASSIFIER_2_USE != 'Wide_ResNet':
    MODEL_VERSION = MODEL_VERSION.replace('ensemble','')# Should not have specific_model=ensemble unless using an ensemble. Might cause truble when looking for saved logits content'
if USE_ALL_EXISTING_Ts is not None:
    desired_Ts = []
    found_saved_Ts = np.zeros([0]).astype(int)
else:
    desired_Ts = ['+'.join(t) for t in TRANSFORMATIONS_LIST]
    if MODEL_VERSION=='ensemble':
        desired_Ts = [m+t for m in (['']+['model_seed%d_'%(i+1) for i in range(4)]) for t in desired_Ts]+['model_seed%d'%(i+1) for i in range(4)]
    found_saved_Ts = np.zeros([len(desired_Ts)]).astype(int)
saved_logits_file_name = 'saved_logits_%s'%(full_model_name)
saved_logits_file_name = saved_logits_file_name.replace('_ensemble','')

logits_files_2_load,found_saved_Ts = Locate_Saved_logits(s.path.join(SAVED_LOGITS_FOLDER,'saved_logits_files.csv'),saved_logits_file_name,MODEL_VERSION,desired_Ts,USE_ALL_EXISTING_Ts)
# logits_files_2_load = []
# with open(os.path.join(SAVED_LOGITS_FOLDER,'saved_logits_files.csv'),'r') as csvfile:
#     for row in csv.reader(csvfile):
#         if any([re.search(name+'_(\d)+\.npy',row[0]) for name in ([(saved_logits_file_name+identifier) for identifier in (['']+['_Seed%d'%(i+1) for i in range(4)])] if MODEL_VERSION=='ensemble' else [saved_logits_file_name])]):
#             saved_Ts = row[1].split(',')
#             if re.search('_Seed\d_',row[0]) is not None and MODEL_VERSION=='ensemble':
#                 seed_num = int(re.search('(?<=_Seed)\d',row[0]).group(0))
#                 saved_Ts = ['model_seed%d_'%(seed_num)+t for t in saved_Ts]
#             if USE_ALL_EXISTING_Ts is not None:
#                 Ts_2_add = [T for T in saved_Ts if T not in desired_Ts and
#                             ((any([re.match(p,T) is not None for p in USE_ALL_EXISTING_Ts]) and 'model_seed' not in T))]
#                 found_saved_Ts = np.concatenate([found_saved_Ts,np.zeros([len(Ts_2_add)]).astype(int)])
#                 desired_Ts += Ts_2_add
#             if re.search('_Seed\d_',row[0]) is not None and MODEL_VERSION=='ensemble':
#                 existing_desired_Ts = [('model_seed%d_'%(seed_num)+T) in saved_Ts for T in desired_Ts]
#             else:
#                 existing_desired_Ts = [T in saved_Ts for T in desired_Ts]
#             if any(existing_desired_Ts):
#                 logits_files_2_load.append(row[0])
#                 found_saved_Ts[np.argwhere(existing_desired_Ts)] = len(logits_files_2_load)
#                 print('Using saved logits from file %s'%(logits_files_2_load[-1]))
#                 if all(found_saved_Ts>0) and USE_ALL_EXISTING_Ts is None:
#                     break

print('Using %d different transformations'%(len(desired_Ts)))
TRANSFORMATIONS_LIST = [T.split('+') for T in desired_Ts]
if 'perT_MSR' in METHODS_2_COMPARE:
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if m!='perT_MSR']+['%s_MSR'%('+'.join(t)) for t in TRANSFORMATIONS_LIST]

if CLASSIFIER_2_USE != 'ELU':
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if m != 'mandelbaum_scores']

# classifier_output_dict = None
# if len(logits_files_2_load)>0:
#     loaded_desired_Ts = np.zeros([len(desired_Ts)]).astype(np.bool)
#     unique_files_2_load = sorted([i for i in list(set(list(found_saved_Ts))) if i>0],key=lambda x:logits_files_2_load[x-1]) #Removing the invalid 0 index and sorting as a hack to prevent loading model_seed files first
#     for file_num in unique_files_2_load:
#         # if file_num==0:
#         #     continue
#         filename_2_load = logits_files_2_load[file_num-1]
#         loaded_dict = np.load(os.path.join(SAVED_LOGITS_FOLDER,filename_2_load),allow_pickle=True).item()
#         saved_Ts = ['+'.join(T) for T in loaded_dict['Transformations']]
#         assert len(saved_Ts)==((loaded_dict['logits'].shape[1]//num_classes)-1),'Saved logits vector length does not match num of saved transformations'
#         if re.search('_Seed\d_',filename_2_load) is not None and MODEL_VERSION=='ensemble':
#             seed_num = int(re.search('(?<=_Seed)\d',filename_2_load).group(0))
#             saved_Ts = ['model_seed%d_'%(seed_num)+t for t in saved_Ts]
#         def transformations_match(saved_T,desired_T):
#             # if 'Wide_ResNet_Seed' in full_model_name:
#             #     return saved_T==('model_seed%d_'%(seed_num)+desired_T)
#             # else:
#             return saved_T==desired_T
#         corresponding_T_indexes = [(desired_T_index,saved_T_index) for saved_T_index,T in enumerate(saved_Ts) for desired_T_index,desired_T in enumerate(desired_Ts) if transformations_match(T,desired_T)]
#         if classifier_output_dict is None:
#             classifier_output_dict = copy.deepcopy(loaded_dict)
#             classifier_output_dict['logits'] = np.zeros(shape=[loaded_dict['logits'].shape[0],(1+len(desired_Ts))*num_classes])
#             if CURVE_TYPE!='param_search':#I allow loading the orignal image logits as 0 for the Seed!=0 case only when used for param_search
#                 assert not any(['model_seed' in t for t in saved_Ts]),'Should not use the file containing logits computed by differently seeded models as source for original logits, as those are set to zero there. Solve this if it happens'
#             classifier_output_dict['logits'][:,:num_classes] = loaded_dict['logits'][:,:num_classes]
#         # TRANSFORMATIONS_LIST = [classifier_output_dict['Transformations'][i] for i in corresponding_T_indexes]#I"m not sure why I need this line - It seems like I'm recreating the same list.
#         # classifier_output_dict['logits'] = np.concatenate([classifier_output_dict['logits'][:,num_classes*(i+1):num_classes*(i+2)] for i in [-1]+corresponding_T_indexes],1)
#         for desired_T_index,saved_T_index in corresponding_T_indexes:
#             if loaded_desired_Ts[desired_T_index]:
#                 continue
#             loaded_desired_Ts[desired_T_index] = True
#             classifier_output_dict['logits'][:,(desired_T_index+1)*num_classes:(desired_T_index+2)*num_classes] = loaded_dict['logits'][:,(saved_T_index+1)*num_classes:(saved_T_index+2)*num_classes]
#         if np.all(loaded_desired_Ts):
#             break
classifier_output_dict = Load_Saved_Logits(logits_files_2_load,found_saved_Ts,desired_Ts,num_classes,models_ensemble_mode=MODEL_VERSION=='ensemble')

if any(found_saved_Ts==0):
    remaining_transformations = [T for i,T in enumerate([T.split('+') for T in desired_Ts]) if not found_saved_Ts[i]]
    # assert not os.path.isfile(os.path.join(SAVED_LOGITS_FOLDER, saved_logits_file_name)),'These logits were already extracted...'
    # if '/ybahat/PycharmProjects/' in os.getcwd():
    #     GPU_2_use = Assign_GPU()
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (GPU_2_use[0])  # Limit to 1 GPU when using an interactive session
    if CLASSIFIER_2_USE == 'ELU':
        import Mandelbaum.elu_network_stl10 as Amit_code
        import reproduce_amit_scores
        model = Amit_code.STL10_Model(amit_model=MODEL_VERSION)
        resulting_dict = model.Evaluate_Model(transformations_list=remaining_transformations,set='val',amitFeatures='mandelbaum_scores' in METHODS_2_COMPARE)
        mandelbaum_scores, mandelbaum_correct_predictions = reproduce_amit_scores.Calc_Amit_scores(model,input_folder=IAD_FEATURES_FOLDER)
        assert np.all(np.logical_not(mandelbaum_correct_predictions) == resulting_dict['error_indicator']), \
            'There seems to be a mismatch between Mandelbaum''s loaded data and the data used for other methods'
        resulting_dict['mandelbaum_scores'] = mandelbaum_scores
        model_seed_transformations = [True]
    else:#Wide_ResNet
        import Wide_ResNet.train as Wide_ResNet_code
        if CLASSIFIER_2_USE == 'ResNet18':
            from torchvision.models import resnet18
            model = nn.DataParallel(resnet18(pretrained=True)).cuda()
        elif CLASSIFIER_2_USE=='ResNext':
            from torchvision.models import resnext101_32x8d
            model = nn.DataParallel(resnext101_32x8d(pretrained=True)).cuda()
        elif CLASSIFIER_2_USE == 'AlexNet':
            from torchvision.models import alexnet
            model = nn.DataParallel(alexnet(pretrained=True)).cuda()
        elif CLASSIFIER_2_USE=='SVHN':
            from SVHN import model as SVHN_model
            model = torch.nn.DataParallel(SVHN_model.svhn(n_channel=32)).cuda()
            model.module.load_state_dict(torch.load(os.path.join('/net/mraid11/export/data/ybahat/Databases/SVHN/pretrained_model','svhn-f564f3d8.pth')))
        elif CLASSIFIER_2_USE in ['CIFAR10','CIFAR100']:
            from CIFAR import model as CIFAR_model
            cifar_type = CLASSIFIER_2_USE[len('CIFAR'):]
            model = torch.nn.DataParallel(getattr(CIFAR_model,used_dataset)(n_channel=128,pretrained=True)).cuda()
        else:
            model,_ = Wide_ResNet_code.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_'+(MODEL_VERSION if MODEL_VERSION in CUTOUT_MODELS else 'Seed0'))
        data_loader = Wide_ResNet_code.Return_Dataloaders(used_dataset,batch_size=int(np.maximum(1,np.floor(32/(1+len(remaining_transformations))))),
                                                     normalize_data=False,download=False)
        data_loader = data_loader[1]
        model_seed_transformations = ['model_seed' in '+'.join(t) for t in remaining_transformations]
        assert all(model_seed_transformations) or np.all(np.logical_not(model_seed_transformations)),'Not supporting mixed logits computation'
        predicted_logits_memory_size = len(data_loader.dataset)*(len(remaining_transformations)+1)*num_classes*4
        assert predicted_logits_memory_size<4e9,'Computed logits memory size (%.2f GB, %d transformations) is about to exceed the 4GB data limit'%(predicted_logits_memory_size/1e9,len(remaining_transformations))
        if model_seed_transformations[0]:
            assert MODEL_VERSION in CUTOUT_MODELS
            resulting_dict,actual_computed_transformations = [],[]
            for seed_num in list(set([int(re.search('(?<=model_seed)\d','+'.join(t)).group(0)) for t in remaining_transformations])):
                model,_ = Wide_ResNet_code.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_Seed%d'%(seed_num))
                this_seed_remaining_Ts = [t for t in remaining_transformations if ('model_seed%d'%(seed_num) in t[0])]
                if len(this_seed_remaining_Ts)==1 and this_seed_remaining_Ts[0][0]=='model_seed%d'%(seed_num):#Only computing logits corresponding to original image:
                    resulting_dict.append(Wide_ResNet_code.test(loader=data_loader,model=model,return_outputs_dict=True,dataset=used_dataset))
                else:
                    transformer = Transformations.Transformer(transformations=[[t.replace('model_seed%d_'%(seed_num),'') for t in t_chain] for t_chain in this_seed_remaining_Ts],min_pixel_value=0,max_pixel_value=1)
                    resulting_dict.append(Wide_ResNet_code.test(loader=data_loader,model=model,transformer=transformer,return_outputs_dict=True,dataset=used_dataset))
                    if not this_seed_remaining_Ts[0] == [['model_seed%d'%(seed_num)]]:#Due to the way I later concatenate all logits from all resulting_dicts, I'm going to include in the logits those that correspond to original images too. So I'm changing the transfomration list accordingly.
                        this_seed_remaining_Ts = [['model_seed%d'%(seed_num)]]+this_seed_remaining_Ts
                actual_computed_transformations.append(this_seed_remaining_Ts)
            # Converting the list of dicts to the expected resulting_dict structure. Adding zeros to substitute for the missing "non-transformed" logits, that were not computed.
            resulting_dict = {'GT_labels':resulting_dict[0]['GT_labels'],'logits':np.concatenate([np.zeros([len(data_loader.dataset),num_classes])]+[d['logits'] for d in resulting_dict],-1)}
            remaining_transformations = [t for t_list in actual_computed_transformations for t in t_list]
        else:
            transformer = Transformations.Transformer(transformations=remaining_transformations,min_pixel_value=0,max_pixel_value=1)
            resulting_dict = Wide_ResNet_code.test(loader=data_loader,model=model,transformer=transformer,return_outputs_dict=True,dataset=used_dataset)
    assert len(remaining_transformations)==((resulting_dict['logits'].shape[1]//num_classes)-1),'Saved logits vector length does not match num of saved transformations'
    file_counter = 0
    saved_logits_file_name += '_%d.npy'
    while os.path.isfile(os.path.join(SAVED_LOGITS_FOLDER,saved_logits_file_name%(file_counter))):
        file_counter += 1
    saved_logits_file_name = saved_logits_file_name%(file_counter)
    resulting_dict['Transformations'] = remaining_transformations
    np.save(os.path.join(SAVED_LOGITS_FOLDER, saved_logits_file_name), resulting_dict)
    trans_list_4_saving = ','.join(['+'.join(t) for t in sorted(remaining_transformations,key=lambda t:''.join(t))])
    copyfile(os.path.join(SAVED_LOGITS_FOLDER, 'saved_logits_files.csv'),os.path.join(SAVED_LOGITS_FOLDER, 'saved_logits_files.bkp'))
    with open(os.path.join(SAVED_LOGITS_FOLDER, 'saved_logits_files.csv'), 'a') as csvfile:
        csv.writer(csvfile).writerow([saved_logits_file_name,trans_list_4_saving])
    if classifier_output_dict is not None:
        assert model_seed_transformations[0] or np.max(np.abs(resulting_dict['logits'][:,:num_classes]-classifier_output_dict['logits'][:,:num_classes]))<np.percentile(np.abs(resulting_dict['logits']),0.01),'Difference in logits of original images between loaded and computed logits.'
        corresponding_T_indexes = [(desired_T_index,saved_T_index) for saved_T_index,T in enumerate(['+'.join(t) for t in remaining_transformations]) for desired_T_index,desired_T in enumerate(desired_Ts) if T==desired_T]
        for desired_T_index,saved_T_index in corresponding_T_indexes:
            classifier_output_dict['logits'][:,(desired_T_index+1)*num_classes:(desired_T_index+2)*num_classes] = resulting_dict['logits'][:,(saved_T_index+1)*num_classes:(saved_T_index+2)*num_classes]
    else:
        classifier_output_dict = resulting_dict
classifier_output_dict['Transformations'] = TRANSFORMATIONS_LIST
classifier_output_dict = example_utils.hyper_partition(classifier_output_dict,used_dataset,HYPER_PARTITION)
GT_labels = classifier_output_dict['GT_labels']
if TOP5:
    per_version_accuracy = [np.mean(np.any(np.argpartition(classifier_output_dict['logits'][:,(i)*num_classes:(i+1)*num_classes],kth=-5,axis=-1)[:,-5:]==GT_labels.reshape([-1,1]),-1)) for i in range(len(TRANSFORMATIONS_LIST)+1)]
else:
    per_version_accuracy = [np.mean(np.argmax(classifier_output_dict['logits'][:,(i)*num_classes:(i+1)*num_classes],-1)==GT_labels) for i in range(len(TRANSFORMATIONS_LIST)+1)]
DISPLAY_SORTED_ACCURACY = True
if DISPLAY_SORTED_ACCURACY:
    print('%sAccuracy on original images: %.3f\nOn transformed images (low to high):'%('Top-5 ' if TOP5 else '',per_version_accuracy[0])+
          ''.join(['\n%s: %.3f'%(t,acc) for acc,t in sorted(zip(per_version_accuracy,TRANSFORMATIONS_LIST))]))
else:
    print('%sAccuracy on original images: %.3f\nOn transformed images:'%('Top-5 ' if TOP5 else '',per_version_accuracy[0])+
          ''.join(['\n%s: %.3f'%(t,per_version_accuracy[i+1]) for i,t in enumerate(TRANSFORMATIONS_LIST)]))

if CURVE_TYPE=='transformations_corr':
    all_posteriors = example_utils.Softmax(classifier_output_dict['logits'],num_classes=num_classes).reshape([classifier_output_dict['logits'].shape[0],len(TRANSFORMATIONS_LIST)+1,num_classes])
    decreasing_acc_order = np.argsort(per_version_accuracy)[::-1]
    all_posteriors = all_posteriors[:,decreasing_acc_order,:]
    transformations_corr = np.mean(np.stack([np.corrcoef(all_posteriors[:,:,c].transpose()) for c in range(num_classes)],-1),-1)
    t_list = [['+'.join(t) for t in [['original']]+TRANSFORMATIONS_LIST][i] for i in decreasing_acc_order]
    plt.imshow(transformations_corr)
    plt.colorbar()
    plt.gca().set_xticks(np.arange(len(t_list)))
    plt.gca().set_yticks(np.arange(len(t_list)))
    plt.gca().set_yticklabels(['(%d) %s'%(i,t) for i,t in enumerate(t_list)])
    # plt.yticks(rotation=90)
    plt.gca().set_xticklabels(['(%d) %.3f'%(i,per_version_accuracy[ord]) for i,ord in enumerate(decreasing_acc_order)])
    plt.xticks(rotation=90)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.title('Transformations normalized correlations and accuracy')
    plt.savefig(os.path.join(FIGURES_FOLDER, 'Transformations_corr_%s.png'%(saved_logits_file_name[:-4])))
    sys.exit(0)
metrics_dict = example_utils.Logits_2_Metrics(classifier_output_dict,num_classes=num_classes,effective_posteriors=EFFECTIVE_POSTERIORS,
                                              desired_metrics=METHODS_2_COMPARE,temperatures=TEMPERATURE,transformations=TRANSFORMATIONS_LIST,top5=TOP5,models_ensemble=MODEL_VERSION=='ensemble')
if TOP5:
    detection_labels = np.all(np.argpartition(metrics_dict[EFFECTIVE_POSTERIORS],axis=-1,kth=-5)[:,-5:]!=GT_labels.reshape([-1,1]),-1)
else:
    detection_labels = np.argmax(metrics_dict[EFFECTIVE_POSTERIORS],-1)!=GT_labels
print('Final classification accuracy %.3f'%(1-np.mean(detection_labels)))
# all_T_detection_labels = np.argmax(classifier_output_dict['logits'].reshape([classifier_output_dict['logits'].shape[0],-1,num_classes]),-1)!=np.expand_dims(GT_labels,-1)
# T_detection_configurations = [int(''.join(c.astype(int).astype(str)),2) for c in all_T_detection_labels]
# different_T_detection_configurations = set(T_detection_configurations)

if CURVE_TYPE not in ['comparison','diff','incremental','excess','accum_diff','param_search','excess_savelog','num_transformations_performance','AORC_savelog','AORC']:
    sys.exit(0)

if 'mandelbaum_scores' in METHODS_2_COMPARE:
    metrics_dict['mandelbaum_scores'] = classifier_output_dict['mandelbaum_scores']
if PCA_method:
    METHODS_2_COMPARE += ['PCA']
curve_type = CURVE_TYPE
if curve_type in ['param_search','num_transformations_performance']:
    curve_type = 'excess'
risk,coverage,AORC = example_utils.Calc_RC(
    detection_scores=[(-1 if example_utils.RegExp_In_List(method,example_utils.RELIABILITY_SCORE_METHODS,example_utils.UNCERTAINY_SCORE_METHODS) else 1)*metrics_dict[method] \
                      for method in METHODS_2_COMPARE],detection_labels=detection_labels,curve_type=curve_type)

if CURVE_TYPE=='num_transformations_performance':
    AORC_ORDER = 'learned' #'increasing','decreasing','arbitrary','learned'
    NUM_LEARNING_FOLDS = 10#30#20#10
    FOLDS_CONFIG = 'bootstrapping'#'mutually_exclusive','random','bootstrapping'
    if AORC_ORDER in ['increasing','decreasing','learned']:
        methods_order = np.argsort(AORC)
        if AORC_ORDER in ['decreasing','learned']:
            # if AORC_ORDER in ['decreasing']:
            methods_order = methods_order[::-1]
    elif AORC_ORDER=='arbitrary':
        methods_order = np.arange(len(AORC))
    # Brinigng the identity transformation to always be first:
    methods_order = [0]+[t for t in methods_order if t>0]
    temp_dict = {}
    chosen_Ts = []
    if AORC_ORDER=='learned':
        reordered_dict = {'logits':classifier_output_dict['logits'][:,:num_classes]}
        temp_dict['logits'] = 1*reordered_dict['logits']
        num_folds = NUM_LEARNING_FOLDS
        if FOLDS_CONFIG=='mutually_exclusive':
            fold_indexes = np.random.randint(0,high=num_folds,size=reordered_dict['logits'].shape[0]).reshape([-1,1])
            valid_fold_samples = np.arange(num_folds).reshape([1,-1])
            valid_fold_samples = np.equal(valid_fold_samples,fold_indexes)
        elif FOLDS_CONFIG=='random':
            valid_fold_samples = np.random.uniform(low=0,high=1,size=[reordered_dict['logits'].shape[0],num_folds])<=0.3#0.3#0.9#0.8,2/3
        else:
            valid_fold_samples = np.round(np.random.uniform(low=0,high=1,size=[reordered_dict['logits'].shape[0],num_folds])*reordered_dict['logits'].shape[0]-0.5).astype(int)
    else:
        num_folds = 1
        reordered_dict = {'logits':np.concatenate([classifier_output_dict['logits'][:,i*num_classes:(i+1)*num_classes] for i in methods_order],1)}
        valid_fold_samples = np.ones([reordered_dict['logits'].shape[0],1]).astype(np.bool)
    per_num_transformations_AORC = []
    for num_Ts in tqdm(range(len(methods_order))):
        cur_AORC = []
        if AORC_ORDER=='learned':
            if num_Ts>0:
                temp_dict['logits'] = np.concatenate([reordered_dict['logits'],classifier_output_dict['logits'][:,methods_order[num_Ts]*num_classes:(methods_order[num_Ts]+1)*num_classes]],1)
        else:
            temp_dict['logits'] = reordered_dict['logits'][:,:num_classes*(1+num_Ts)]
        temp_Ts = [TRANSFORMATIONS_LIST[i-1] for i in (chosen_Ts+[methods_order[num_Ts]]) if i>0]
        for fold_num in range(num_folds):
            # valid_fold_samples = np.random.uniform(low=0,high=1,size=reordered_dict['logits'].shape[0])<0.5#0.9#0.8,2/3
            per_fold_dict = {'logits':temp_dict['logits'][valid_fold_samples[:,fold_num],:]}
            cur_metrics = example_utils.Logits_2_Metrics(per_fold_dict,num_classes=num_classes,effective_posteriors=EFFECTIVE_POSTERIORS,desired_metrics=['T_ensemble_MSR'],
                                                         temperatures=TEMPERATURE,transformations=temp_Ts,top5=TOP5,models_ensemble=MODEL_VERSION=='ensemble',verbosity=False)
            _,_,per_fold_AORC = example_utils.Calc_RC(detection_scores=-1*cur_metrics['T_ensemble_MSR'],detection_labels=detection_labels[valid_fold_samples[:,fold_num]],curve_type='excess')
            cur_AORC.append(per_fold_AORC)
        if AORC_ORDER=='learned' and num_Ts>0:
            # if np.mean([cur_AORC[i]>=per_num_transformations_AORC[-1][i] for i in range(num_folds)])>=0.6:#0.7:#0.7:#0.5:#0.7
            if np.mean(cur_AORC)>=np.mean(per_num_transformations_AORC[-1]):
                reordered_dict['logits'] = 1*temp_dict['logits']
                chosen_Ts.append(methods_order[num_Ts])
                per_num_transformations_AORC.append(cur_AORC)
            else:
                pass
                # print('Skipping %s as only %.2f of folds found it usefull'%(TRANSFORMATIONS_LIST[methods_order[num_Ts]-1],np.mean([cur_AORC[i]>per_num_transformations_AORC[-1][i] for i in range(num_folds)])))
        else:
            chosen_Ts.append(methods_order[num_Ts])
            per_num_transformations_AORC.append(cur_AORC)
    plt.plot([np.mean(p) for p in per_num_transformations_AORC])
    if AORC_ORDER=='learned':
        print('Learned optimal transformations list (%d):'%(len(chosen_Ts)-1))
        print('TRANSFORMATIONS_LIST =',[TRANSFORMATIONS_LIST[i-1] for i in chosen_Ts[1:]])
    plt.xlabel('# Transformations used')
    plt.ylabel('Excess AORC')
    plt.title('%s by %s (acc.=%.3f) on %s'%('Amit model: %s,'%(MODEL_VERSION) if CLASSIFIER_2_USE=='ELU' else '%s%s,'%('Top-5 ' if TOP5 else '',CLASSIFIER_2_USE),EFFECTIVE_POSTERIORS,
                                            1-detection_labels.mean(),HYPER_PARTITION))
    plt.savefig(os.path.join(FIGURES_FOLDER,'%sModel_%s%s%s_%s_on_%s_Methods_%s.png'%('Num_Ts_',full_model_name,'top5' if TOP5 else '','modelEns' if MODEL_VERSION=='ensemble' else '',
                                                                                      'Effective_%s'%(EFFECTIVE_POSTERIORS),HYPER_PARTITION,'+'.join([m[:4] for m in METHODS_2_COMPARE])[:40])), bbox_inches = 'tight',pad_inches = 0.05)
    sys.exit(0)
if CURVE_TYPE=='param_search':
    RADIUS_PLOT = True
    NORMALIZE_SHIFT_SIZE = True
    OFF_AND_ON_AXIS = False
    transformation_regexp = USE_ALL_EXISTING_Ts[0]
    transformation_name = re.search('^(\w)*',transformation_regexp).group(0)
    num_param_dims = 1+len(re.findall('_',transformation_regexp))
    per_method_params = [re.search('(?<='+transformation_name+').*(?=_MSR)',method).group(0) for method in METHODS_2_COMPARE if method!='original_MSR']
    per_method_params = [[int(p) for p in params.split('_')] for params in per_method_params]
    if NORMALIZE_SHIFT_SIZE and transformation_name=='shift' and RADIUS_PLOT:
        per_method_params = [[p/IM_SIZE[used_dataset] for p in param] for param in per_method_params]
    assert not any([all([p==0 for p in params]) for params in per_method_params])
    neutral_params = {'shift':0,'zoomin':100,'rotate':0,'gamma':10}
    per_method_params = [[neutral_params[transformation_name]]*num_param_dims]+per_method_params
    params_range = [(min([p[i] for p in per_method_params]),max([p[i] for p in per_method_params])) for i in range(num_param_dims)]
    # def
    filename_2_save = os.path.join(PARAMS_SEARCH_FOLDER,'%sModel_%s%s_%s_Trans_%s.npz'%(CURVE_TYPE+'_',full_model_name,'top5' if TOP5 else '',
                                                                                        'Effective_%s'%(EFFECTIVE_POSTERIORS),transformation_name))
    if num_param_dims==1:
        unique_params,mapping_2_unique,unique_counter = np.unique(per_method_params,return_inverse=True,return_counts=True)
        per_param_AORC = np.zeros([len(unique_params)])
        for AORC_i,unique_i in enumerate(mapping_2_unique):
            per_param_AORC[unique_i] += AORC[AORC_i]
        per_param_AORC /= unique_counter
        # params_order = np.argsort(np.concatenate(per_method_params))
        # plt.plot(np.concatenate([per_method_params[i] for i in params_order]),[AORC[i] for i in params_order])
        plt.plot(unique_params,per_param_AORC)
        np.savez(filename_2_save,optional_param=unique_params,per_param_AORC=per_param_AORC)
    elif RADIUS_PLOT and transformation_name=='shift':
        corresponding_radii = [np.sqrt(p[0]**2+p[1]**2) for p in per_method_params]
        optional_radii,mapping_2_optional,per_radii_counter = np.unique(corresponding_radii,return_inverse=True,return_counts=True)
        if OFF_AND_ON_AXIS:
            off_axis_translations = [(p[0]!=0 and p[1]!=0) for p in per_method_params]
        else:
            off_axis_translations = [False for p in per_method_params]
        per_radii_AORC = np.zeros([len(optional_radii)])
        for AORC_i,radii_i in enumerate(mapping_2_optional):
            per_radii_AORC[radii_i] += AORC[AORC_i]
        per_radii_AORC /= per_radii_counter
        on_axis_radii_i = sorted(list(set([m for i,m in enumerate(mapping_2_optional) if not off_axis_translations[i]])))
        plt.plot(optional_radii[on_axis_radii_i],per_radii_AORC[on_axis_radii_i])
        if OFF_AND_ON_AXIS:
            off_axis_radii_i = sorted(list(set([m for i,m in enumerate(mapping_2_optional) if off_axis_translations[i]])))
            plt.plot(optional_radii[off_axis_radii_i],per_radii_AORC[off_axis_radii_i])
            plt.legend(['On axis','Off axis'])
        np.savez(filename_2_save,optional_param=optional_radii,per_param_AORC=per_radii_AORC)
    else:
        params_matrix = np.zeros([params_range[i][1]-params_range[i][0]+1 for i in range(num_param_dims)])
        counter = np.zeros_like(params_matrix)
        for i,method in enumerate(METHODS_2_COMPARE):
            params_matrix[per_method_params[i][0]-params_range[0][0],per_method_params[i][1]-params_range[1][0]] += AORC[i]
            counter[per_method_params[i][0]-params_range[0][0],per_method_params[i][1]-params_range[1][0]] += 1
        params_matrix /= counter
        plt.imshow(params_matrix.transpose(),'gray',extent=[params_range[0][0],params_range[0][1],params_range[1][0],params_range[1][1]])
        plt.colorbar()
    plt.title('%s: %s'%(full_model_name,transformation_name))
    plt.savefig(os.path.join(PARAMS_SEARCH_FOLDER,'%s%sModel_%s%s_%s_Trans_%s.png'%(CURVE_TYPE+'_','radius_' if RADIUS_PLOT else '',full_model_name,'top5' if TOP5 else '',
                                                                                    'Effective_%s'%(EFFECTIVE_POSTERIORS),transformation_name)))
    sys.exit(0)

example_utils.Create_RC_figure(risk=risk, coverage=coverage, score=AORC,methods_names=METHODS_2_COMPARE,temperature_dict=TEMPERATURE,comparison_fig=False,
                               curve_type=CURVE_TYPE,title='%s by %s (acc.=%.3f) on %s'%('Amit model: %s,'%(MODEL_VERSION) if CLASSIFIER_2_USE=='ELU' else '%s%s,'%('Top-5 ' if TOP5 else '',CLASSIFIER_2_USE),EFFECTIVE_POSTERIORS,
                                                                                         1-detection_labels.mean(),HYPER_PARTITION),paper_mode=PAPER_FIG_MODE)
plt.savefig(os.path.join(FIGURES_FOLDER,'%sModel_%s%s%s_%s_on_%s_Methods_%s.png'%((CURVE_TYPE+'_') if CURVE_TYPE is not None else '',full_model_name,'top5' if TOP5 else '','modelEns' if MODEL_VERSION=='ensemble' else '',
                                                                                  'Effective_%s'%(EFFECTIVE_POSTERIORS),HYPER_PARTITION,'+'.join([m[:4] for m in METHODS_2_COMPARE])[:40])), bbox_inches = 'tight',pad_inches = 0.05)

if 'savelog' in CURVE_TYPE:
    fields_dict_keys = ['dataset','classifier','method','AORC','effective_posteriors','classification_accuracy','top-5','transformations','model_ensemble','hyper_partition','eAURC']
    first_csv_writing = not os.path.isfile(os.path.join(RESULTS_CSV_FOLDER,'AORC_results.csv'))
    if not first_csv_writing:
        copyfile(os.path.join(RESULTS_CSV_FOLDER,'AORC_results.csv'),os.path.join(RESULTS_CSV_FOLDER,'AORC_results_backup.csv'))
        with open(os.path.join(RESULTS_CSV_FOLDER,'AORC_results_backup.csv'),'r',newline='') as f_read:
            csv_reader = csv.DictReader(f_read)
            existing_field_names = csv_reader.fieldnames
            missing_fields = [field for field in fields_dict_keys if field not in existing_field_names]
            if len(missing_fields)>0:
                print('New fields added - rewriting file!')
                with open(os.path.join(RESULTS_CSV_FOLDER,'AORC_results.csv'),'w',newline='') as f_write:
                    csv_writer = csv.DictWriter(f_write,fieldnames=fields_dict_keys)
                    csv_writer.writeheader()
                    for i,row in enumerate(csv_reader):
                        # if row['dataset'] in ['cifar10','ImageNet']:
                        #     continue
                        csv_writer.writerow(row)
    with open(os.path.join(RESULTS_CSV_FOLDER,'AORC_results.csv'),'a',newline='') as f:
        csv_writer = csv.DictWriter(f,fieldnames=fields_dict_keys)
        if first_csv_writing:
            csv_writer.writeheader()
        for i,method in enumerate(METHODS_2_COMPARE):
            csv_writer.writerow(dict(zip(fields_dict_keys,
                                         [used_dataset,full_model_name,method,AORC[i],EFFECTIVE_POSTERIORS,1-detection_labels.mean(),TOP5,','.join(['+'.join(t) for t in sorted(TRANSFORMATIONS_LIST,key=lambda t:''.join(t))]),
                                          MODEL_VERSION=='ensemble',HYPER_PARTITION,'AORC' not in CURVE_TYPE])))

