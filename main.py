import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import Transformations
import example_utils
import csv
import re
from example_utils import Assign_GPU
import torch
import torch.nn as nn
import copy
from shutil import copyfile
import csv
from tqdm import tqdm

# Available measures (methods):
CURVE_TYPE = 'AORC_savelog'#'comparison'#'diff'#'incremental'#'excess','accum_diff','2Dhists','transformations_corr','posteriors','param_search','excess_savelog','num_transformations_performance','AORC'
PAPER_FIG_MODE = False
TOP5 = False
USE_ALL_EXISTING_Ts = ['.*']#
# USE_ALL_EXISTING_Ts = ['shift(-)?\d_(-)?\d','horFlip\+shift(-)?\d_(-)?\d']#,'horFlip$'# Works well for Cutout (wideResNet on STL10)
# USE_ALL_EXISTING_Ts = ['shift(-)?(\d)+_(-)?(\d)+','horFlip\+shift(-)?(\d)+_(-)?(\d)+']
# USE_ALL_EXISTING_Ts = ['shift(-)?(\d)+_(-)?(\d)+']
# USE_ALL_EXISTING_Ts = ['gamma(\d)+$','horFlip\+gamma(\d)+$']
# USE_ALL_EXISTING_Ts = ['gamma(\d)+$']
# USE_ALL_EXISTING_Ts = ['rotate(-)?(\d)+$','horFlip\+rotate(-)?(\d)+$']
# USE_ALL_EXISTING_Ts = ['rotate(-)?(\d)+$']
# USE_ALL_EXISTING_Ts = ['zoomin(\d)+$','horFlip\+zoomin(\d)+$']
# USE_ALL_EXISTING_Ts = ['zoomin(\d)+$']
# USE_ALL_EXISTING_Ts = ['horFlip$','rotate(-)?\d','gamma(\d)+','horFlip\+rotate(-)?\d','horFlip\+gamma(\d)+']# Works well for AlexNet
USE_ALL_EXISTING_Ts = None
METHODS_2_COMPARE = ['original_MSR','perT_MSR','T_ensemble_MSR']#'KLD_horFlip'
METHODS_2_COMPARE = ['T_ensemble_MSR','original_MSR']
METHODS_2_COMPARE = ['T_ensemble_MSR','original_MSR','T_MSR_bootstraping','mandelbaum_scores']#,'weighted_independent_prob_model'],'weighted_independent_prob_model_bootstraping'
# METHODS_2_COMPARE = ['KLD_shift3_0','KST_shift3_0']#,'weighted_independent_prob_model'],'weighted_independent_prob_model_bootstraping'
# METHODS_2_COMPARE = ['KLD_horFlip','KST_horFlip']#,'weighted_independent_prob_model'],'weighted_independent_prob_model_bootstraping'
# METHODS_2_COMPARE = ['T_ensemble_MSR','original_MSR','T_MSR_bootstraping','mandelbaum_scores','KLD_shift3_0','KST_shift3_0']#,'weighted_independent_prob_model'],'weighted_independent_prob_model_bootstraping','model_ensemble_MSR'
TEMPERATURE = {'default':1.}
CLASSIFIER_2_USE = 'AlexNet'#'Cutout','Amit','ResNet18','ResNext','AlexNet','SVHN','CIFAR10','CIFAR100'
USE_TRAIN_SET = False
HYPER_PARTITION = 'test'#'train','test'
# use_saved_logits = True
SPECIFIC_MODEL = 'None'#'None','AT','dist','Seed0_no_augm','flipOnly','flipOnlyHalfData','Seed1','Seed2','ensemble','Seed0_untrained','Seed0_no_flip'
AMIT_MODELS = ['None','AT','dist']
CUTOUT_MODELS = ['Seed0_no_augm','Seed0_flipOnly','Seed0_flipOnlyHalfData','ensemble','Seed0_untrained','Seed0_no_flip']+['Seed%d'%(i+1) for i in range(4)]
EFFECTIVE_POSTERIORS = 'original_posteriors'#'T_ensemble_posteriors','original_posteriors','T_bootstraping_posteriors'
# HIST_SCORES_X = 'Independent_prob_model'#'T_ensemble_var_2_oneHot'
# HIST_SCORES_Y = 'T_ensemble_MSR'#'T_ensemble_MSR'
PCA_method = False
# TEMPERATURE = [0.05,0.9,1.1]#[0.05,0.1,0.5,1,5,10]
SHIFT_SIZE = 5
shifts_list = [-SHIFT_SIZE,0,SHIFT_SIZE]

# shifts_list = [-SHIFT_SIZE,SHIFT_SIZE]
used_dataset = 'ImageNet' if CLASSIFIER_2_USE in ['ResNet18','ResNext','AlexNet'] else 'svhn' if CLASSIFIER_2_USE=='SVHN' else 'cifar10' if CLASSIFIER_2_USE=='CIFAR10' else 'cifar100' if CLASSIFIER_2_USE=='CIFAR100' else 'stl10'
NUM_CLASSES = {'stl10':10,'ImageNet':1000,'svhn':10,'cifar10':10,'cifar100':100}
IM_SIZE = {'stl10':96,'ImageNet':224,'svhn':32,'cifar10':32,'cifar100':32}

shifts_list = [1,2,3]
shifts_list = [-s for s in shifts_list]+[0]+shifts_list
shift_transforms = [['shift%d_%d'%(i,j)] for i in shifts_list for j in shifts_list if not (i==0 and j==0)]#Including diagonal shifts
TRANSFORMATIONS_LIST = [['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]+[['gamma8']]+[['gamma9']]#+[['zoomin95']]#File saved_logits_Cutout_0, Botstrap +0.0005 over higher MSR (200 Resamples)
if used_dataset=='cifar100':
    shifts_list = [1,2,3]
    shifts_list = [-s for s in shifts_list]+[0]+shifts_list
    shift_transforms = [['shift%d_%d'%(i,j)] for i in shifts_list for j in shifts_list if not (i==0 and j==0)]#Including diagonal shifts
    TRANSFORMATIONS_LIST = [['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]+[['gamma9']]+[['gamma11']]
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip', 'shift1_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['shift0_-3'], ['shift1_2'], ['horFlip', 'shift2_-2'], ['horFlip', 'shift1_-3'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift0_-5'], ['shift0_5'], ['rotate3'], ['shift13_13'], ['horFlip', 'shift14_14']]
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip', 'shift1_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['shift0_-3'], ['horFlip', 'shift0_-2'], ['horFlip', 'shift1_-2'], ['shift1_2'], ['horFlip', 'shift2_-2'], ['horFlip', 'shift1_-3'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift2_-3'], ['shift1_3'], ['horFlip', 'shift-3_0'], ['horFlip', 'shift0_4'], ['shift0_3'], ['horFlip', 'shift0_-5'], ['horFlip', 'shift5_-5'], ['rotate3'], ['shift3_5'], ['shift-9_11'], ['shift11_9'], ['horFlip', 'shift5_-11'], ['shift-11_-9'], ['horFlip', 'shift-11_11'], ['horFlip', 'shift12_12'], ['horFlip', 'shift14_14']]
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['horFlip', 'shift0_2'], ['shift-1_-2'], ['horFlip', 'shift-2_0'], ['horFlip', 'shift0_-2'], ['shift1_2'], ['horFlip', 'shift2_-2'], ['horFlip', 'shift0_-3'], ['shift1_3'], ['horFlip', 'shift-3_0'], ['horFlip', 'shift3_-3'], ['horFlip', 'shift0_-5'], ['shift0_5'], ['horFlip', 'shift7_0'], ['rotate3'], ['shift3_5'], ['shift13_13']]
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip', 'shift1_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['shift-2_-1'], ['horFlip', 'shift0_2'], ['shift-1_-2'], ['shift0_-3'], ['horFlip', 'shift0_-2'], ['shift1_2'], ['horFlip', 'shift1_2'], ['horFlip', 'shift1_-3'], ['horFlip', 'shift3_0'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift-1_-3'], ['horFlip', 'shift2_-3'], ['shift1_3'], ['horFlip', 'shift6_0'], ['horFlip', 'shift2_3'], ['horFlip', 'shift0_-5'], ['horFlip', 'shift4_4'], ['horFlip', 'shift0_5'], ['shift0_5'], ['horFlip', 'shift7_0'], ['horFlip', 'shift5_-5'], ['rotate3'], ['shift9_9'], ['shift13_0'], ['shift-9_-9'], ['horFlip', 'shift-3_-11'], ['horFlip', 'shift0_13'], ['shift-9_11'], ['shift-5_-11'], ['shift11_9'], ['horFlip', 'shift5_-11'], ['shift9_11'], ['horFlip', 'shift9_-11'], ['vertFlip'], ['horFlip', 'shift12_12']]
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['shift1_-2'], ['shift0_-3'], ['horFlip', 'shift0_-2'], ['horFlip', 'shift2_-2'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift2_-3'], ['horFlip', 'shift-1_2'], ['shift1_3'], ['horFlip', 'shift-3_0'], ['horFlip', 'shift0_4'], ['horFlip', 'shift3_-3'], ['horFlip', 'shift2_3'], ['horFlip', 'shift0_-5'], ['shift5_-3'], ['horFlip', 'shift-5_-3'], ['shift0_5'], ['horFlip', 'shift7_0'], ['horFlip', 'shift5_-5'], ['rotate3'], ['horFlip', 'shift0_6'], ['shift10_10'], ['horFlip', 'shift5_-11'], ['horFlip', 'shift12_12'], ['shift13_13'], ['horFlip', 'shift14_14']]
    # 1 fold, as long as AORC keeps increasing, Ts from best to worse
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['horFlip', 'shift0_2'], ['shift-1_-2'], ['shift0_-3'], ['horFlip', 'shift0_-2'], ['shift1_2'], ['horFlip', 'shift1_-3'], ['shift2_1'], ['horFlip', 'shift4_0'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift2_-3'], ['shift1_3'], ['horFlip', 'shift-3_0'], ['horFlip', 'shift6_0'], ['horFlip', 'shift0_-5'], ['horFlip', 'shift4_4'], ['horFlip', 'shift0_5'], ['horFlip', 'shift-5_-3'], ['shift0_5'], ['horFlip', 'shift7_0'], ['horFlip', 'shift5_-5'], ['rotate3'], ['shift10_10'], ['horFlip', 'shift5_-11'], ['shift9_11'], ['shift-11_-9'], ['vertFlip'], ['horFlip', 'shift11_-11'], ['horFlip', 'shift12_12'], ['shift14_14'], ['horFlip', 'shift14_14']]
elif used_dataset=='cifar10':
    # 1 fold, as long as AORC keeps increasing, Ts from best to worse
    TRANSFORMATIONS_LIST = [['shift1_-1'], ['shift0_-1'], ['shift1_0'], ['horFlip', 'shift0_-1'], ['shift-1_0'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['shift1_1'], ['horFlip', 'shift-2_-1'], ['horFlip', 'shift1_-2'], ['shift-3_1'], ['horFlip', 'shift2_0']]
elif used_dataset=='svhn':
    shifts_list = [1,2,3]
    shifts_list = [-s for s in shifts_list]+[0]+shifts_list
    shift_transforms = [['shift%d_%d'%(i,j)] for i in shifts_list for j in shifts_list if not (i==0 and j==0)]#Including diagonal shifts
    TRANSFORMATIONS_LIST = shift_transforms+[['gamma%d'%(g+4)] for g in range(6)]
    TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_2'], ['shift-5_4'], ['shift7_0'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift9_0'], ['shift13_13'], ['shift14_14'], ['shift10_0']]
    TRANSFORMATIONS_LIST = [['shift5_0'], ['shift4_0'], ['shift5_4'], ['shift5_2'], ['shift-3_-5'], ['shift-5_-3'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift10_0'], ['shift11_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-4_-3'], ['shift-5_4'], ['shift6_0'], ['shift5_-5'], ['shift7_0'], ['shift0_-11'], ['shift-7_0'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift12_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-4_-3'], ['shift-5_4'], ['shift6_0'], ['shift5_-5'], ['shift7_0'], ['shift0_-11'], ['shift-7_0'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift12_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-4_-3'], ['shift-5_4'], ['shift6_0'], ['shift5_-5'], ['shift7_0'], ['shift0_-11'], ['shift-7_0'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift12_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-2_-3'], ['shift-4_-3'], ['shift5_-1'], ['shift-4_-4'], ['shift5_1'], ['shift-4_4'], ['shift-5_4'], ['shift-5_5'], ['shift5_-3'], ['rotate-15'], ['shift0_14'], ['shift-5_-5'], ['shift6_0'], ['shift5_-5'], ['shift6_6'], ['shift0_-9'], ['shift7_0'], ['shift0_-11'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift13_0'], ['shift12_0'], ['shift14_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-2_-3'], ['shift-4_-3'], ['shift5_-1'], ['shift-4_4'], ['shift-5_4'], ['shift-5_5'], ['shift5_-3'], ['rotate-13'], ['shift0_14'], ['shift-5_-5'], ['shift6_0'], ['shift5_-4'], ['shift5_-5'], ['shift6_6'], ['shift0_-9'], ['shift7_0'], ['shift0_-11'], ['shift-7_0'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift13_0'], ['shift12_0'], ['shift14_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-2_-3'], ['shift-4_-3'], ['shift5_-1'], ['shift-4_4'], ['shift-5_4'], ['shift-5_5'], ['shift5_-3'], ['rotate-15'], ['shift0_14'], ['shift-5_-5'], ['shift6_0'], ['shift5_-5'], ['shift6_6'], ['shift0_-9'], ['shift7_0'], ['shift0_-11'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift13_0'], ['shift12_0'], ['shift14_0']]
    # TRANSFORMATIONS_LIST = [['shift5_0'], ['shift5_4'], ['shift-2_-3'], ['shift-4_-3'], ['shift5_-1'], ['shift-4_4'], ['shift-5_4'], ['shift-5_5'], ['shift5_-3'], ['rotate-15'], ['shift0_14'], ['shift-5_-5'], ['shift6_0'], ['shift5_-4'], ['shift5_-5'], ['shift6_6'], ['shift0_-9'], ['shift7_0'], ['shift0_-11'], ['shift-7_0'], ['shift-9_0'], ['shift-11_0'], ['shift7_7'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift11_11'], ['shift10_0'], ['shift11_0'], ['shift13_0'], ['shift12_0'], ['shift14_0']]

elif used_dataset=='stl10':
    if CLASSIFIER_2_USE=='Amit':
        shifts_list = [1,2,3]
        # 1 fold, as long as AORC keeps increasing, Ts from best to worse
        TRANSFORMATIONS_LIST = [['gamma12'], ['zoomin95'], ['shift0_-1'], ['horFlip', 'zoomin95'], ['shift0_-5'], ['shift1_0'], ['horFlip', 'shift1_1'], ['shift0_-7'], ['zoomin95', 'shift0_5'], ['horFlip', 'shift0_-1'], ['rotate5'], ['horFlip', 'shift0_-5']]
    else:
        shifts_list = [3,5,7]
        # 1 fold, as long as AORC keeps increasing, Ts from best to worse
        TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift7_-4'], ['shift0_12'], ['horFlip', 'shift-4_-2'], ['shift-9_9'], ['horFlip', 'shift4_1'], ['shift0_14'], ['horFlip', 'shift0_12'], ['horFlip', 'shift12_12'], ['horFlip', 'shift7_0'], ['horFlip', 'shift12_0'], ['gamma8'], ['shift14_14'], ['horFlip', 'shift13_0'], ['horFlip', 'shift14_0'], ['shift0_18'], ['gamma6'], ['horFlip', 'shift19_0'], ['shift0_22'], ['horFlip', 'gamma6'], ['horFlip', 'rotate-1'], ['horFlip', 'rotate1'], ['rotate1'], ['shift22_22'], ['horFlip', 'rotate-7']]
        # shifts_list = [1,3,5,7]
    # shifts_list = [-s for s in shifts_list]+[0]+shifts_list
    # shift_transforms = [['shift%d_%d'%(i,j)] for i in shifts_list for j in shifts_list if not (i==0 and j==0)]#Including diagonal shifts
    # TRANSFORMATIONS_LIST = [['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]+[['gamma8']]+[['gamma9']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-5_6'], ['shift-6_7'], ['horFlip', 'shift7_-3'], ['horFlip', 'shift4_-3'], ['shift-4_5'], ['horFlip', 'shift12_12'], ['shift-9_0'], ['horFlip', 'shift0_18'], ['horFlip', 'rotate4']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift2_6'], ['shift-2_6'], ['horFlip', 'shift-4_6'], ['shift-5_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift5_-2'], ['shift-6_2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift5_2'], ['shift-3_6'], ['shift-5_4'], ['horFlip', 'shift7_-2'], ['horFlip', 'shift7_2'], ['horFlip', 'shift-2_-2'], ['horFlip', 'shift3_-2'], ['shift2_-2'], ['horFlip', 'shift1_6'], ['horFlip', 'shift-3_-6'], ['horFlip', 'shift7_4'], ['horFlip', 'shift4_3'], ['horFlip', 'shift5_4'], ['shift-7_4'], ['shift-4_-3'], ['horFlip', 'shift-3_1'], ['horFlip', 'shift1_-2'], ['shift0_14'], ['horFlip', 'shift-1_-3'], ['horFlip', 'shift-5_-2'], ['horFlip', 'shift-6_6'], ['horFlip', 'shift-2_4'], ['horFlip', 'shift-5_5'], ['horFlip', 'shift-1_-2'], ['horFlip', 'shift0_9'], ['shift7_6'], ['shift-6_-6'], ['shift-1_4'], ['shift3_-2'], ['horFlip', 'shift0_12'], ['shift0_11'], ['shift-3_4'], ['horFlip', 'shift2_-3'], ['shift0_13'], ['horFlip', 'shift6_0'], ['horFlip', 'shift-2_-4'], ['horFlip', 'shift-2_1'], ['shift3_5'], ['shift-3_2'], ['shift4_5'], ['horFlip', 'shift1_-6'], ['horFlip', 'shift-5_7'], ['shift-7_5'], ['horFlip', 'shift7_7'], ['horFlip', 'shift-1_-4'], ['horFlip', 'shift2_5'], ['horFlip', 'shift-6_5'], ['shift7_9'], ['shift6_7'], ['shift-1_2'], ['horFlip', 'shift-7_2'], ['shift-4_1'], ['horFlip', 'shift9_9'], ['horFlip', 'shift6_-6'], ['shift-7_11'], ['horFlip', 'shift12_12'], ['shift-4_-6'], ['shift1_4'], ['horFlip', 'shift1_5'], ['shift1_2'], ['horFlip', 'shift5_3'], ['horFlip', 'shift-4_-5'], ['shift-7_7'], ['horFlip', 'shift9_0'], ['horFlip', 'shift0_11'], ['shift-7_-3'], ['horFlip', 'shift-2_-6'], ['horFlip', 'shift2_2'], ['shift-9_7'], ['shift6_5'], ['horFlip', 'shift1_4'], ['shift2_-6'], ['horFlip', 'shift7_-7'], ['horFlip', 'shift2_7'], ['horFlip', 'shift7_0'], ['shift0_-2'], ['shift0_4'], ['horFlip', 'shift-2_3'], ['shift7_11'], ['shift-6_-7'], ['horFlip', 'shift1_-3'], ['horFlip', 'shift5_0'], ['shift6_2'], ['horFlip', 'shift-6_2'], ['horFlip', 'shift-6_4'], ['horFlip', 'shift-4_-7'], ['shift-3_-4'], ['horFlip', 'shift2_3'], ['horFlip', 'shift2_-6'], ['horFlip', 'shift-9_9'], ['shift6_-2'], ['horFlip', 'shift0_7'], ['shift-2_-3'], ['shift2_-4'], ['horFlip', 'shift-2_-7'], ['horFlip', 'shift5_-7'], ['horFlip', 'shift-2_0'], ['shift-1_-3'], ['horFlip', 'shift9_-7'], ['shift-2_1'], ['horFlip', 'shift-3_-5'], ['shift5_2'], ['shift-6_0'], ['shift3_-6'], ['horFlip', 'shift1_1'], ['horFlip', 'shift3_-5'], ['horFlip', 'shift1_7'], ['horFlip', 'shift-9_7'], ['shift0_-6'], ['shift7_5'], ['horFlip', 'shift2_-1'], ['shift4_-6'], ['horFlip', 'shift10_0'], ['horFlip', 'shift11_11'], ['horFlip', 'shift7_-1'], ['horFlip', 'shift-7_3'], ['shift9_9'], ['shift-7_-1'], ['horFlip', 'shift1_-1'], ['shift13_13'], ['shift0_15'], ['shift11_7'], ['shift0_-5'], ['horFlip', 'gamma7'], ['shift7_1'], ['shift3_-1'], ['shift6_-1'], ['shift5_1'], ['horFlip', 'shift-7_0'], ['shift5_-7'], ['shift0_-1'], ['shift5_-5'], ['shift-1_-5'], ['horFlip', 'shift0_-11'], ['shift1_-1'], ['horFlip', 'shift0_17'], ['horFlip', 'shift-7_-9'], ['shift7_0'], ['shift-7_-11'], ['shift-10_-10'], ['horFlip', 'shift14_14'], ['horFlip', 'shift-11_-7'], ['shift-7_-9'], ['shift7_-5'], ['horFlip', 'shift-9_0'], ['horFlip', 'shift-9_-9'], ['horFlip', 'gamma12'], ['horFlip', 'shift0_18'], ['horFlip', 'shift-7_-11'], ['shift10_0'], ['gamma11'], ['shift-11_-9'], ['horFlip', 'shift11_-11'], ['shift9_0'], ['shift7_-9'], ['shift12_0'], ['horFlip', 'shift20_0'], ['gamma6'], ['shift13_0'], ['horFlip', 'shift0_19'], ['shift14_0'], ['horFlip', 'gamma13'], ['shift0_19'], ['horFlip', 'gamma6'], ['horFlip', 'shift0_20'], ['horFlip', 'shift-11_-11'], ['gamma13'], ['shift0_20'], ['shift16_16'], ['horFlip', 'shift0_22'], ['shift19_0'], ['horFlip', 'shift0_23'], ['shift17_17'], ['shift20_0'], ['shift0_25'], ['horFlip', 'shift24_0'], ['shift18_18'], ['horFlip', 'shift0_25'], ['horFlip', 'shift17_17'], ['horFlip', 'shift0_26'], ['horFlip', 'shift18_18'], ['shift25_0'], ['shift27_0'], ['shift20_20'], ['horFlip', 'rotate-1'], ['horFlip', 'rotate1'], ['horFlip', 'rotate2'], ['horFlip', 'shift21_21'], ['rotate-2'], ['horFlip', 'rotate4'], ['horFlip', 'shift22_22'], ['horFlip', 'shift23_23'], ['horFlip', 'shift24_24'], ['zoomin95', 'shift5_5'], ['horFlip', 'rotate-4'], ['rotate4'], ['shift24_24'], ['shift25_25'], ['horFlip', 'zoomin90'], ['horFlip', 'rotate8'], ['horFlip', 'rotate-10'], ['horFlip', 'rotate12'], ['rotate-14']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['shift0_9'], ['horFlip', 'shift12_12']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift6_6'], ['gamma6']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['horFlip', 'shift4_-3'], ['horFlip', 'shift-3_2'], ['horFlip', 'shift3_-2'], ['shift10_10'], ['shift-3_-2']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift7_-3'], ['shift0_12'], ['horFlip', 'shift-4_-2'], ['horFlip', 'shift7_5'], ['horFlip', 'shift0_10'], ['shift0_14'], ['horFlip', 'shift-4_1'], ['horFlip', 'shift12_0'], ['horFlip', 'gamma8'], ['shift14_14'], ['horFlip', 'shift13_0'], ['horFlip', 'gamma7'], ['gamma7'], ['shift0_18'], ['shift0_22'], ['horFlip', 'gamma6'], ['shift0_26'], ['horFlip', 'rotate-1']]
    # # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift2_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift4_-3'], ['horFlip', 'shift6_-2'], ['horFlip', 'shift6_6'], ['horFlip', 'shift7_-2'], ['shift0_12'], ['shift-4_-2'], ['horFlip', 'shift-4_-3'], ['horFlip', 'shift-3_-3'], ['horFlip', 'gamma8'], ['horFlip', 'shift7_-1'], ['shift11_11'], ['shift-9_0'], ['horFlip', 'shift-1_0'], ['shift14_14'], ['horFlip', 'gamma7'], ['horFlip', 'shift14_0'], ['gamma7'], ['horFlip', 'shift17_0'], ['horFlip', 'shift-7_-11'], ['gamma6'], ['shift0_22'], ['horFlip', 'shift22_0'], ['shift0_26'], ['horFlip', 'shift0_26'], ['horFlip', 'rotate1'], ['rotate-1'], ['horFlip', 'rotate3'], ['zoomin95', 'shift0_5']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift7_-3'], ['horFlip', 'gamma7']] #May be good for Ensemble

    # TRANSFORMATIONS_LIST = [['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]+[['gamma7']]+[['gamma8']]+[['gamma9']]+[['rotate1']]+[['rotate3']]+[['vertFlip']]+[['colorSwap']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['shift2_6'], ['shift-2_6'], ['horFlip', 'shift-4_6'], ['shift-5_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6']]

    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['shift-7_6'], ['horFlip', 'shift7_-3'], ['shift6_6'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift6_6'], ['horFlip', 'shift7_6'], ['horFlip', 'shift-4_5'], ['shift0_12'], ['horFlip', 'shift6_5'], ['shift1_6'], ['horFlip', 'shift7_5'], ['shift-9_9'], ['shift10_10'], ['horFlip', 'shift-2_-3'], ['shift-7_9'], ['shift0_14'], ['horFlip', 'shift-5_-2'], ['horFlip', 'shift6_0'], ['horFlip', 'shift-2_1'], ['horFlip', 'shift9_9'], ['horFlip', 'shift10_10'], ['horFlip', 'shift12_12'], ['shift4_-2'], ['shift11_9'], ['horFlip', 'shift2_0'], ['horFlip', 'shift0_16'], ['horFlip', 'shift13_13'], ['horFlip', 'shift14_0'], ['horFlip', 'shift-7_0'], ['gamma7'], ['shift0_18'], ['horFlip', 'shift15_0']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift7_-4'], ['shift0_12'], ['horFlip', 'shift-4_-2'], ['shift-9_9'], ['horFlip', 'shift4_1'], ['shift0_14'], ['horFlip', 'shift0_12'], ['horFlip', 'shift12_12'], ['horFlip', 'shift7_0'], ['horFlip', 'shift12_0'], ['gamma8'], ['shift14_14'], ['horFlip', 'shift13_0'], ['horFlip', 'shift14_0'], ['shift0_18'], ['gamma6'], ['horFlip', 'shift19_0'], ['shift0_22'], ['horFlip', 'gamma6'], ['horFlip', 'rotate-1'], ['horFlip', 'rotate1'], ['rotate1'], ['shift22_22'], ['horFlip', 'rotate-7']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift4_-2'], ['shift0_12'], ['horFlip', 'shift-2_-2'], ['shift10_10'], ['horFlip', 'shift6_1'], ['horFlip', 'shift-3_1'], ['shift0_26'], ['rotate1']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift7_-3'], ['horFlip', 'gamma8'], ['horFlip', 'gamma7']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['horFlip', 'shift-4_6']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift4_-2'], ['shift0_12'], ['horFlip', 'shift-2_-2'], ['shift10_10'], ['horFlip', 'shift6_1'], ['horFlip', 'shift-3_1'], ['shift0_26'], ['rotate1']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift7_-3'], ['horFlip', 'shift6_6'], ['horFlip', 'shift4_5'], ['horFlip', 'gamma7'], ['gamma7']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift4_-3'], ['shift-4_7'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift7_-2'], ['horFlip', 'shift7_-4'], ['shift0_12'], ['horFlip', 'shift-4_-3'], ['horFlip', 'shift7_0'], ['shift12_12'], ['horFlip', 'shift12_0'], ['horFlip', 'shift10_0'], ['horFlip', 'gamma8'], ['gamma8'], ['shift14_14'], ['gamma7'], ['gamma6'], ['horFlip', 'gamma6'], ['horFlip', 'shift-9_-11'], ['horFlip', 'shift0_22'], ['rotate-1'], ['horFlip', 'rotate2'], ['horFlip', 'rotate-7']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift7_-4'], ['shift0_12'], ['horFlip', 'shift-4_-2'], ['horFlip', 'shift6_1'], ['shift0_14'], ['horFlip', 'shift6_0'], ['horFlip', 'shift7_0'], ['shift6_-2'], ['horFlip', 'gamma7'], ['gamma7'], ['shift0_18'], ['horFlip', 'shift7_-11'], ['horFlip', 'rotate-1'], ['horFlip', 'rotate1'], ['shift21_21']]
    # TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift6_6'], ['shift0_12'], ['horFlip', 'shift-4_-2']]
elif used_dataset=='ImageNet':
    # 1 fold, as long as AORC keeps increasing, Ts from best to worse
    TRANSFORMATIONS_LIST = [['gamma9'], ['gamma8'], ['shift0_1'], ['shift0_2'], ['rotate1'], ['rotate4'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['gamma6'], ['horFlip', 'shift1_0'], ['horFlip', 'shift0_2'], ['zoomin92'], ['zoomin95', 'gamma8'], ['horFlip', 'shift-1_2'], ['horFlip', 'shift0_5'], ['horFlip', 'shift-3_0'], ['shift0_10'], ['shift0_-5'], ['horFlip', 'shift-5_1'], ['horFlip', 'shift0_10'], ['horFlip', 'shift-1_-3'], ['shift1_-5'], ['horFlip', 'shift0_-3'], ['rotate7'], ['horFlip', 'rotate-10'], ['rotate9'], ['horFlip', 'shift15_10']]
    # 10 BS folds, as long as average AORC increases:
    TRANSFORMATIONS_LIST = [['gamma9'], ['gamma8'], ['shift0_1'], ['shift0_2'], ['rotate1'], ['rotate4'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['gamma6'], ['horFlip', 'shift1_0'], ['zoomin92'], ['zoomin95', 'gamma8'], ['horFlip', 'shift-1_2'], ['horFlip', 'shift0_5'], ['horFlip', 'shift2_0'], ['rotate5'], ['horFlip', 'shift-5_1'], ['horFlip', 'shift0_10'], ['horFlip', 'shift0_-10'], ['shift0_-10'], ['horFlip', 'zoomin90'], ['rotate7'], ['horFlip', 'rotate-10']]
# shifts_list = [1,3,5,7]
# shifts_list = [-s for s in shifts_list]+[0]+shifts_list
# shift_transforms = [['shift%d_%d'%(i,j)] for i in shifts_list for j in shifts_list if not (i==0 and j==0)]#Including diagonal shifts
# TRANSFORMATIONS_LIST = [['vertFlip']]+[['colorSwap']]+[['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]+[['gamma8']]+[['gamma9']]

# TRANSFORMATIONS_LIST = [['horFlip']]
#  shifts_list = [i for i in range(15)]
# shift_transforms = [['shift%d_%d'%(0,j)] for j in shifts_list if j!=0]+[['shift%d_%d'%(j,0)] for j in shifts_list if j!=0]+[['shift%d_%d'%(j,j)] for j in shifts_list if j!=0]#Excluding diagonal shifts
# TRANSFORMATIONS_LIST = [['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]
# TRANSFORMATIONS_LIST = shift_transforms
# shift_transforms = [['shift0_%d'%(j)] for j in shifts_list if not (j==0)]
# gama_transforms = [['gamma%d'%(g)] for g in [7,8,9,11,12,13]]
# TRANSFORMATIONS_LIST = [['horFlip'],['zoomin95'],['zoomin97'],['horFlip','zoomin95'],['gamma8'],['gamma12'],['rotate5']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]+[t0+t1 for t0 in [['zoomin95']] for t1 in shift_transforms]
# TRANSFORMATIONS_LIST = [['horFlip'],['zoomin95']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]#+[t0+t1 for t0 in [['zoomin95']] for t1 in shift_transforms]
# TRANSFORMATIONS_LIST = [['horFlip']]+[t0+t1 for t0 in [[],['horFlip']] for t1 in shift_transforms]#+[t0+t1 for t0 in [['zoomin95']] for t1 in shift_transforms]
# TRANSFORMATIONS_LIST = shift_transforms#+[t0+t1 for t0 in [['zoomin95']] for t1 in shift_transforms]
# TRANSFORMATIONS_LIST = [['zoomin90'],['horFlip'],['zoomin90','horFlip']]
# TRANSFORMATIONS_LIST = [['gamma%d'%(g+3)] for g in range(7)]+[['gamma%d'%(g+11)] for g in range(4)]
# TRANSFORMATIONS_LIST = [['vertFlip'],['colorSwap'],['horFlip']]
# rotation_angles = [i+1 for i in range(15)]
# rotation_angles = [-r for r in rotation_angles]+rotation_angles
# TRANSFORMATIONS_LIST = [t0+t1 for t0 in [[],['horFlip']] for t1 in [['rotate%d'%(angle)] for angle in rotation_angles]]
# TRANSFORMATIONS_LIST = [['rotate%d'%(angle)] for angle in rotation_angles]
# TRANSFORMATIONS_LIST = [['rotate%d'%(angle)] for angle in rotation_angles]+[['gamma7']]+[['gamma8']]+[['gamma9']]#+[['zoomin97']]+[['zoomin95']]+[['zoomin92']]
# TRANSFORMATIONS_LIST = [t0+t1 for t0 in [[],['horFlip']] for t1 in [['zoomin%d'%(100-angle)] for angle in rotation_angles]]
# TRANSFORMATIONS_LIST = [t0+t1 for t0 in [[],['horFlip']] for t1 in [['gamma%d'%(10+int(max(rotation_angles)/2)-angle)] for angle in rotation_angles if int(max(rotation_angles)/2)-angle!=0]]
# TRANSFORMATIONS_LIST = [['horFlip'],['zoomin97'],['gamma12'],['rotate2'],['shift-5_-5']]#File saved_logits_Cutout_7.npy, Botstrap +0.0012 over higher MSR (400 Resamples)
# # TRANSFORMATIONS_LIST = [['horFlip'],['zoomin97'],['gamma12'],['rotate2'],['shift-5_-5'],['gamma8'],['horFlip','shift2_3'],
# #                         ['horFlip','gamma12'],['zoomin95','gamma8']]#File saved_logits_Cutout_8, Botstrap +0.0028 over higher MSR (400 Resamples)
# TRANSFORMATIONS_LIST = [['horFlip'],['zoomin97'],['gamma12'],['rotate2'],['shift-5_-5'],['gamma8'],['horFlip','shift2_3'],
#                         ['horFlip','gamma12'],['zoomin95','gamma8'],['horFlip','zoomin97'],['horFlip','rotate-3']]
# TRANSFORMATIONS_LIST = [['horFlip'],['zoomin92'],['gamma12'],['rotate5'],['shift-10_-10'],['gamma8'],['horFlip','shift12_13'],
#                         ['horFlip','gamma12'],['zoomin95','gamma8'],['horFlip','zoomin90'],['horFlip','rotate-10']]#File saved_logits_Cutout_8, Botstrap +0.0028 over higher MSR (400 Resamples)
# TRANSFORMATIONS_LIST = [['shift5_5'],['shift-5_-5'],['shift0_5'],['horFlip','shift5_5'],['shift10_10'],['shift2_3'],['shift1_1'],['shift-1_1'],['horFlip','shift1_1'],['rotate1'],['rotate2'],['rotate4'],
#                         ['rotate7'],['rotate9'],['gamma9'],['gamma6']]
# TRANSFORMATIONS_LIST = [['horFlip'],['zoomin95'],['shift5_5'],['horFlip','shift-5_-5']]
AMIT_FEATURES_FOLDER = '/home/ybahat/data/ErrorDetection/Amit_related/Amit_features/STL10_Amit'#'/home/ybahat/data/Databases/ErrorAndNovelty/STL10_Amit'
FIGURES_FOLDER = '/home/ybahat/PycharmProjects/CFI_private/Error_detection_figures'
PARAMS_SEARCH_FOLDER = '/home/ybahat/PycharmProjects/CFI_private/params_search_figures'
RESULTS_CSV_FOLDER = '/home/ybahat/data/ErrorDetection/saved_results'
SAVED_LOGITS_FOLDER = '/home/ybahat/data/ErrorDetection/Saved_logits'
COMPARISON_CURVE_TYPES = ['diff','comparison','accum_diff']
assert CLASSIFIER_2_USE in ['Cutout','Amit','ResNet18','ResNext','AlexNet','SVHN','CIFAR10','CIFAR100']
assert not TOP5 or CLASSIFIER_2_USE in ['ResNet18','ResNext','AlexNet'],'Why usie top-5 accuracy for datasets other than ImageNet?'
num_classes = NUM_CLASSES[used_dataset]
# if CLASSIFIER_2_USE=='Amit':
# else:
if CURVE_TYPE in ['param_search','num_transformations_performance']:
    # assert USE_ALL_EXISTING_Ts is not None
    HYPER_PARTITION='train'
    # assert HYPER_PARTITION=='train','Not supposed to choose transformations based on the test set...'
    METHODS_2_COMPARE = ['original_MSR','perT_MSR']


assert all([example_utils.RegExp_In_List(method,example_utils.RELIABILITY_SCORE_METHODS+example_utils.UNCERTAINY_SCORE_METHODS,exact_match=True) for method in METHODS_2_COMPARE]),'Unknown confidence estimation method'
KLD_methods = [method for method in METHODS_2_COMPARE if 'KLD_' in method]
if len(KLD_methods)>0 and USE_ALL_EXISTING_Ts is None:
    assert all([any([method[len('KLD_'):]==T[0] and len(T)==1 for T in TRANSFORMATIONS_LIST]) for method in KLD_methods]), \
        'Calling comparison methods %s without proper transformations'%(KLD_methods)
KST_methods = [method for method in METHODS_2_COMPARE if 'KST_' in method]
if len(KST_methods)>0 and USE_ALL_EXISTING_Ts is None:
    assert all([any([method[len('KST_'):]==T[0] and len(T)==1 for T in TRANSFORMATIONS_LIST]) for method in KST_methods]), \
        'Calling comparison methods %s without proper transformations'%(KST_methods)
np.random.seed(0)
full_model_name = CLASSIFIER_2_USE+('_%s'%(SPECIFIC_MODEL) if (CLASSIFIER_2_USE=='Amit' or (CLASSIFIER_2_USE=='Cutout' and SPECIFIC_MODEL in CUTOUT_MODELS)) else '')
if CLASSIFIER_2_USE != 'Cutout':
    SPECIFIC_MODEL = SPECIFIC_MODEL.replace('ensemble','')# Should not have specific_model=ensemble unless using an ensemble. Might cause truble when looking for saved logits content'
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if m != 'model_ensemble_MSR']
elif SPECIFIC_MODEL=='ensemble':
    assert USE_ALL_EXISTING_Ts is None,'Unsupported'
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if (m != 'model_ensemble_MSR' and 'KLD' not in m and 'KST' not in m)]
#     TRANSFORMATIONS_LIST = []
elif 'model_ensemble_MSR' in METHODS_2_COMPARE:
    assert 'Seed' not in SPECIFIC_MODEL,'When computing model_ensemble_MSR score, seed should be set to the default 0'
    TRANSFORMATIONS_LIST += [['model_seed%d'%(seed_num+1)] for seed_num in range(4)]
if USE_ALL_EXISTING_Ts is not None:
    desired_Ts = []
    found_saved_Ts = np.zeros([0]).astype(int)
else:
    desired_Ts = ['+'.join(t) for t in TRANSFORMATIONS_LIST]
    if SPECIFIC_MODEL=='ensemble':
        desired_Ts = [m+t for m in (['']+['model_seed%d_'%(i+1) for i in range(4)]) for t in desired_Ts]+['model_seed%d'%(i+1) for i in range(4)]
    found_saved_Ts = np.zeros([len(desired_Ts)]).astype(int)
saved_logits_name = 'saved_logits_%s%s'%(full_model_name,'_train' if USE_TRAIN_SET else '')
saved_logits_name = saved_logits_name.replace('_ensemble','')
use_saved_logits = False
logits_files_2_load = []
with open(os.path.join(SAVED_LOGITS_FOLDER,'saved_logits_files.csv'),'r') as csvfile:
    for row in csv.reader(csvfile):
        if any([re.search(name+'_(\d)+\.npy',row[0]) for name in ([(saved_logits_name+identifier) for identifier in (['']+['_Seed%d'%(i+1) for i in range(4)])] if SPECIFIC_MODEL=='ensemble' else [saved_logits_name])]):
            saved_Ts = row[1].split(',')
            if re.search('_Seed\d_',row[0]) is not None and SPECIFIC_MODEL=='ensemble':
                seed_num = int(re.search('(?<=_Seed)\d',row[0]).group(0))
                saved_Ts = ['model_seed%d_'%(seed_num)+t for t in saved_Ts]
            if USE_ALL_EXISTING_Ts is not None:
                # Ts_2_add = [T for T in saved_Ts if T not in desired_Ts and
                #             ((any([re.match(p,T) is not None for p in USE_ALL_EXISTING_Ts]) and 'model_seed' not in T) or ('model_ensemble_MSR' in METHODS_2_COMPARE and 'model_seed' in T))]
                Ts_2_add = [T for T in saved_Ts if T not in desired_Ts and
                            ((any([re.match(p,T) is not None for p in USE_ALL_EXISTING_Ts]) and 'model_seed' not in T))]
                found_saved_Ts = np.concatenate([found_saved_Ts,np.zeros([len(Ts_2_add)]).astype(int)])
                desired_Ts += Ts_2_add
            # else:
            if re.search('_Seed\d_',row[0]) is not None and SPECIFIC_MODEL=='ensemble':
                existing_desired_Ts = [('model_seed%d_'%(seed_num)+T) in saved_Ts for T in desired_Ts]
            else:
                existing_desired_Ts = [T in saved_Ts for T in desired_Ts]
            if any(existing_desired_Ts):
                use_saved_logits = True
                # saved_logits_name = row[0]
                logits_files_2_load.append(row[0])
                found_saved_Ts[np.argwhere(existing_desired_Ts)] = len(logits_files_2_load)
                print('Using saved logits from file %s'%(logits_files_2_load[-1]))
                if all(found_saved_Ts>0) and USE_ALL_EXISTING_Ts is None:
                    break
# if USE_ALL_EXISTING_Ts is not None:
print('Using %d different transformations'%(len(desired_Ts)))
TRANSFORMATIONS_LIST = [T.split('+') for T in desired_Ts]
if 'perT_MSR' in METHODS_2_COMPARE:
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if m!='perT_MSR']+['%s_MSR'%('+'.join(t)) for t in TRANSFORMATIONS_LIST]
# if use_saved_logits:
if CLASSIFIER_2_USE != 'Amit':
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if m != 'mandelbaum_scores']
outputs_dict = None
if len(logits_files_2_load)>0:
    loaded_desired_Ts = np.zeros([len(desired_Ts)]).astype(np.bool)
    unique_files_2_load = sorted([i for i in list(set(list(found_saved_Ts))) if i>0],key=lambda x:logits_files_2_load[x-1]) #Removing the invalid 0 index and sorting as a hack to prevent loading model_seed files first
    for file_num in unique_files_2_load:
        # if file_num==0:
        #     continue
        filename_2_load = logits_files_2_load[file_num-1]
        loaded_dict = np.load(os.path.join(SAVED_LOGITS_FOLDER,filename_2_load),allow_pickle=True).item()
        saved_Ts = ['+'.join(T) for T in loaded_dict['Transformations']]
        assert len(saved_Ts)==((loaded_dict['logits'].shape[1]//num_classes)-1),'Saved logits vector length does not match num of saved transformations'
        if re.search('_Seed\d_',filename_2_load) is not None and SPECIFIC_MODEL=='ensemble':
            seed_num = int(re.search('(?<=_Seed)\d',filename_2_load).group(0))
            saved_Ts = ['model_seed%d_'%(seed_num)+t for t in saved_Ts]
        def transformations_match(saved_T,desired_T):
            # if 'Cutout_Seed' in full_model_name:
            #     return saved_T==('model_seed%d_'%(seed_num)+desired_T)
            # else:
            return saved_T==desired_T
        corresponding_T_indexes = [(desired_T_index,saved_T_index) for saved_T_index,T in enumerate(saved_Ts) for desired_T_index,desired_T in enumerate(desired_Ts) if transformations_match(T,desired_T)]
        if outputs_dict is None:
            outputs_dict = copy.deepcopy(loaded_dict)
            outputs_dict['logits'] = np.zeros(shape=[loaded_dict['logits'].shape[0],(1+len(desired_Ts))*num_classes])
            if CURVE_TYPE!='param_search':#I allow loading the orignal image logits as 0 for the Seed!=0 case only when used for param_search
                assert not any(['model_seed' in t for t in saved_Ts]),'Should not use the file containing logits computed by differently seeded models as source for original logits, as those are set to zero there. Solve this if it happens'
            outputs_dict['logits'][:,:num_classes] = loaded_dict['logits'][:,:num_classes]
        # TRANSFORMATIONS_LIST = [outputs_dict['Transformations'][i] for i in corresponding_T_indexes]#I"m not sure why I need this line - It seems like I'm recreating the same list.
        # outputs_dict['logits'] = np.concatenate([outputs_dict['logits'][:,num_classes*(i+1):num_classes*(i+2)] for i in [-1]+corresponding_T_indexes],1)
        for desired_T_index,saved_T_index in corresponding_T_indexes:
            if loaded_desired_Ts[desired_T_index]:
                continue
            loaded_desired_Ts[desired_T_index] = True
            outputs_dict['logits'][:,(desired_T_index+1)*num_classes:(desired_T_index+2)*num_classes] = loaded_dict['logits'][:,(saved_T_index+1)*num_classes:(saved_T_index+2)*num_classes]
        if np.all(loaded_desired_Ts):
            break
if any(found_saved_Ts==0):
    remaining_transformations = [T for i,T in enumerate([T.split('+') for T in desired_Ts]) if not found_saved_Ts[i]]
    assert not os.path.isfile(os.path.join(SAVED_LOGITS_FOLDER, saved_logits_name)),'These logits were already extracted...'
    if '/ybahat/PycharmProjects/' in os.getcwd():
        GPU_2_use = Assign_GPU()
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (GPU_2_use[0])  # Limit to 1 GPU when using an interactive session
    if CLASSIFIER_2_USE == 'Amit':
        import Mandelbaum.elu_network_stl10 as Amit_code
        import reproduce_amit_scores
        model = Amit_code.STL10_Model(amit_model=SPECIFIC_MODEL)
        resulting_dict = model.Evaluate_Model(transformations_list=remaining_transformations,set='val',amitFeatures='mandelbaum_scores' in METHODS_2_COMPARE)
        mandelbaum_scores, mandelbaum_correct_predictions = reproduce_amit_scores.Calc_Amit_scores(model,input_folder=AMIT_FEATURES_FOLDER)
        assert np.all(np.logical_not(mandelbaum_correct_predictions) == resulting_dict['error_indicator']), \
            'There seems to be a mismatch between Mandelbaum''s loaded data and the data used for other methods'
        resulting_dict['mandelbaum_scores'] = mandelbaum_scores
        model_seed_transformations = [True]
    else:#Cutout
        import Cutout.train as Cutout_code
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
            # model.module.load_state_dict(torch.load(os.path.join('/net/mraid11/export/data/ybahat/Databases/SVHN/pretrained_model','svhn-f564f3d8.pth')))
        else:
            # model,_ = Cutout_code.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_Seed0')
            model,_ = Cutout_code.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_'+(SPECIFIC_MODEL if SPECIFIC_MODEL in CUTOUT_MODELS else 'Seed0'))
        data_loader = Cutout_code.Return_Dataloaders(used_dataset,batch_size=int(np.maximum(1,np.floor(32/(1+len(remaining_transformations))))),
                                                     normalize_data=False,download=False)
        data_loader = data_loader[0] if USE_TRAIN_SET else data_loader[1]
        model_seed_transformations = ['model_seed' in '+'.join(t) for t in remaining_transformations]
        assert all(model_seed_transformations) or np.all(np.logical_not(model_seed_transformations)),'Not supporting mixed logits computation'
        predicted_logits_memory_size = len(data_loader.dataset)*(len(remaining_transformations)+1)*num_classes*4
        assert predicted_logits_memory_size<4e9,'Computed logits memory size (%.2f GB, %d transformations) is about to exceed the 4GB data limit'%(predicted_logits_memory_size/1e9,len(remaining_transformations))
        if model_seed_transformations[0]:
            assert SPECIFIC_MODEL in CUTOUT_MODELS
            resulting_dict,actual_computed_transformations = [],[]
            for seed_num in list(set([int(re.search('(?<=model_seed)\d','+'.join(t)).group(0)) for t in remaining_transformations])):
                model,_ = Cutout_code.Load_Model(model='wideresnet', dataset='stl10', test_id='stl10_wideresnet_B32_Seed%d'%(seed_num))
                this_seed_remaining_Ts = [t for t in remaining_transformations if ('model_seed%d'%(seed_num) in t[0])]
                if len(this_seed_remaining_Ts)==1 and this_seed_remaining_Ts[0][0]=='model_seed%d'%(seed_num):#Only computing logits corresponding to original image:
                    resulting_dict.append(Cutout_code.test(loader=data_loader,model=model,return_outputs_dict=True,dataset=used_dataset))
                else:
                    transformer = Transformations.Transformer(transformations=[[t.replace('model_seed%d_'%(seed_num),'') for t in t_chain] for t_chain in this_seed_remaining_Ts],min_pixel_value=0,max_pixel_value=1)
                    resulting_dict.append(Cutout_code.test(loader=data_loader,model=model,transformer=transformer,return_outputs_dict=True,dataset=used_dataset))
                    if not this_seed_remaining_Ts[0] == [['model_seed%d'%(seed_num)]]:#Due to the way I later concatenate all logits from all resulting_dicts, I'm going to include in the logits those that correspond to original images too. So I'm changing the transfomration list accordingly.
                        this_seed_remaining_Ts = [['model_seed%d'%(seed_num)]]+this_seed_remaining_Ts
                actual_computed_transformations.append(this_seed_remaining_Ts)
            # Converting the list of dicts to the expected resulting_dict structure. Adding zeros to substitute for the missing "non-transformed" logits, that were not computed.
            resulting_dict = {'GT_labels':resulting_dict[0]['GT_labels'],'logits':np.concatenate([np.zeros([len(data_loader.dataset),num_classes])]+[d['logits'] for d in resulting_dict],-1)}
            remaining_transformations = [t for t_list in actual_computed_transformations for t in t_list]
        else:
            transformer = Transformations.Transformer(transformations=remaining_transformations,min_pixel_value=0,max_pixel_value=1)
            resulting_dict = Cutout_code.test(loader=data_loader,model=model,transformer=transformer,return_outputs_dict=True,dataset=used_dataset)
    assert len(remaining_transformations)==((resulting_dict['logits'].shape[1]//num_classes)-1),'Saved logits vector length does not match num of saved transformations'
    file_counter = 0
    saved_logits_name += '_%d.npy'
    while os.path.isfile(os.path.join(SAVED_LOGITS_FOLDER,saved_logits_name%(file_counter))):
        file_counter += 1
    saved_logits_name = saved_logits_name%(file_counter)
    resulting_dict['Transformations'] = remaining_transformations
    np.save(os.path.join(SAVED_LOGITS_FOLDER, saved_logits_name), resulting_dict)
    trans_list_4_saving = ','.join(['+'.join(t) for t in sorted(remaining_transformations,key=lambda t:''.join(t))])
    copyfile(os.path.join(SAVED_LOGITS_FOLDER, 'saved_logits_files.csv'),os.path.join(SAVED_LOGITS_FOLDER, 'saved_logits_files.bkp'))
    with open(os.path.join(SAVED_LOGITS_FOLDER, 'saved_logits_files.csv'), 'a') as csvfile:
        csv.writer(csvfile).writerow([saved_logits_name,trans_list_4_saving])
    if outputs_dict is not None:
        assert model_seed_transformations[0] or np.max(np.abs(resulting_dict['logits'][:,:num_classes]-outputs_dict['logits'][:,:num_classes]))<np.percentile(np.abs(resulting_dict['logits']),0.01),'Difference in logits of original images between loaded and computed logits.'
        corresponding_T_indexes = [(desired_T_index,saved_T_index) for saved_T_index,T in enumerate(['+'.join(t) for t in remaining_transformations]) for desired_T_index,desired_T in enumerate(desired_Ts) if T==desired_T]
        for desired_T_index,saved_T_index in corresponding_T_indexes:
            outputs_dict['logits'][:,(desired_T_index+1)*num_classes:(desired_T_index+2)*num_classes] = resulting_dict['logits'][:,(saved_T_index+1)*num_classes:(saved_T_index+2)*num_classes]
    else:
        outputs_dict = resulting_dict
outputs_dict['Transformations'] = TRANSFORMATIONS_LIST
outputs_dict = example_utils.hyper_partition(outputs_dict,used_dataset,HYPER_PARTITION)
GT_labels = outputs_dict['GT_labels']
if TOP5:
    per_version_accuracy = [np.mean(np.any(np.argpartition(outputs_dict['logits'][:,(i)*num_classes:(i+1)*num_classes],kth=-5,axis=-1)[:,-5:]==GT_labels.reshape([-1,1]),-1)) for i in range(len(TRANSFORMATIONS_LIST)+1)]
else:
    per_version_accuracy = [np.mean(np.argmax(outputs_dict['logits'][:,(i)*num_classes:(i+1)*num_classes],-1)==GT_labels) for i in range(len(TRANSFORMATIONS_LIST)+1)]
DISPLAY_SORTED_ACCURACY = True
if DISPLAY_SORTED_ACCURACY:
    print('%sAccuracy on original images: %.3f\nOn transformed images (low to high):'%('Top-5 ' if TOP5 else '',per_version_accuracy[0])+
          ''.join(['\n%s: %.3f'%(t,acc) for acc,t in sorted(zip(per_version_accuracy,TRANSFORMATIONS_LIST))]))
else:
    print('%sAccuracy on original images: %.3f\nOn transformed images:'%('Top-5 ' if TOP5 else '',per_version_accuracy[0])+
          ''.join(['\n%s: %.3f'%(t,per_version_accuracy[i+1]) for i,t in enumerate(TRANSFORMATIONS_LIST)]))

if CURVE_TYPE=='transformations_corr':
    all_posteriors = example_utils.Softmax(outputs_dict['logits'],num_classes=num_classes).reshape([outputs_dict['logits'].shape[0],len(TRANSFORMATIONS_LIST)+1,num_classes])
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
    plt.savefig(os.path.join(FIGURES_FOLDER, 'Transformations_corr_%s.png'%(saved_logits_name[:-4])))
    sys.exit(0)
metrics_dict = example_utils.Logits_2_Metrics(outputs_dict,num_classes=num_classes,effective_posteriors=EFFECTIVE_POSTERIORS,
                                              desired_metrics=METHODS_2_COMPARE,temperatures=TEMPERATURE,transformations=TRANSFORMATIONS_LIST,top5=TOP5,models_ensemble=SPECIFIC_MODEL=='ensemble')
if TOP5:
    detection_labels = np.all(np.argpartition(metrics_dict[EFFECTIVE_POSTERIORS],axis=-1,kth=-5)[:,-5:]!=GT_labels.reshape([-1,1]),-1)
else:
    detection_labels = np.argmax(metrics_dict[EFFECTIVE_POSTERIORS],-1)!=GT_labels
print('Final classification accuracy %.3f'%(1-np.mean(detection_labels)))
# all_T_detection_labels = np.argmax(outputs_dict['logits'].reshape([outputs_dict['logits'].shape[0],-1,num_classes]),-1)!=np.expand_dims(GT_labels,-1)
# T_detection_configurations = [int(''.join(c.astype(int).astype(str)),2) for c in all_T_detection_labels]
# different_T_detection_configurations = set(T_detection_configurations)

if CURVE_TYPE=='posteriors':
    assert len(METHODS_2_COMPARE)<=2
    if CLASSIFIER_2_USE=='Cutout' or True:
        from torchvision import datasets
        STL10_PATH = '/net/mraid11/export/data/ybahat/Databases/stl10'
        test_dataset = datasets.STL10(root=STL10_PATH,split='test')
        with open(os.path.join(STL10_PATH,'stl10_binary','class_names.txt'),'r') as f:
            labels = [l.replace('\n','') for l in f.readlines()]
    NUM_EXAMPLES_2_WRITE = 2
    EXAMPLE_NUM_OFFSET = 40
    scores_negators = [-1 if example_utils.RegExp_In_List(method,example_utils.RELIABILITY_SCORE_METHODS,other_list=example_utils.UNCERTAINY_SCORE_METHODS) else 1 for method in METHODS_2_COMPARE]
    scores_orders = [np.argsort(scores_negators[i]*metrics_dict[method]) for i,method in enumerate(METHODS_2_COMPARE)]
    if len(METHODS_2_COMPARE)==1:
        false_neg_examples = [scores_orders[0][np.argwhere(detection_labels[scores_orders[0]])[EXAMPLE_NUM_OFFSET:EXAMPLE_NUM_OFFSET+NUM_EXAMPLES_2_WRITE]].reshape([-1])]
        false_pos_examples = [scores_orders[0][np.argwhere(np.logical_not(detection_labels)[scores_orders[0]])[-NUM_EXAMPLES_2_WRITE-EXAMPLE_NUM_OFFSET:-EXAMPLE_NUM_OFFSET]].reshape([-1])]
    else:
        ranking_diffs = np.argsort(scores_orders[1])-np.argsort(scores_orders[0])
        false_neg_examples = np.arange(len(detection_labels))[detection_labels]
        false_neg_examples = [false_neg_examples[np.argsort(ranking_diffs[detection_labels])],
                              false_neg_examples[np.argsort(ranking_diffs[detection_labels])]]
        false_neg_examples = [[num for num in nums if np.argsort(scores_orders[0])[num]<6000] for nums in false_neg_examples]
        false_neg_examples = [false_neg_examples[0][-NUM_EXAMPLES_2_WRITE:],false_neg_examples[1][:NUM_EXAMPLES_2_WRITE]]
        false_pos_examples = np.arange(len(detection_labels))[np.logical_not(detection_labels)]
        false_pos_examples = [false_pos_examples[np.argsort(ranking_diffs[np.logical_not(detection_labels)])],
                              false_pos_examples[np.argsort(ranking_diffs[np.logical_not(detection_labels)])]]
        false_pos_examples = [[num for num in nums if np.argsort(scores_orders[0])[num]>7000] for nums in false_pos_examples]
        false_pos_examples = [false_pos_examples[0][:NUM_EXAMPLES_2_WRITE],false_pos_examples[1][-NUM_EXAMPLES_2_WRITE:]]
    CLASS_WIDTH = 3
    width = (CLASS_WIDTH-1)/(len(TRANSFORMATIONS_LIST)+3)
    # labels = [i for i in range(num_classes)]
    x_coords = CLASS_WIDTH*np.arange(len(labels))
    for example_type in ['negative','positive']:
        cur_indicator = false_pos_examples if example_type=='positive' else false_neg_examples
        for advantage_of_method_num in range(len(false_pos_examples)):
            for ind in cur_indicator[advantage_of_method_num]:
                plt.clf()
                plt.subplot(1,2,1)
                plt.imshow(test_dataset.data[ind].transpose((1,2,0)))
                plt.subplot(1,2,2)
                for T_num in range((len(TRANSFORMATIONS_LIST)+2)):
                    if T_num==(len(TRANSFORMATIONS_LIST)+1):
                        plt.bar(x_coords - (1+T_num) * width + CLASS_WIDTH / 2,metrics_dict['T_ensemble_posteriors'][ind],width=2*width)
                    else:
                        plt.bar(x_coords-T_num*width+CLASS_WIDTH/2,example_utils.Softmax(np.expand_dims(outputs_dict['logits'][ind],0),num_classes=10)[0,10*T_num:10*(T_num+1)],
                                width=width)
                plt.gca().set_xticks(x_coords)
                plt.gca().set_xticklabels([str(l) for l in labels])
                if len(METHODS_2_COMPARE) == 1:
                    plt.suptitle('False %s example, MSR %.3f, GT label %s'%(example_type,
                                                                            metrics_dict[METHODS_2_COMPARE[0]][ind],labels[outputs_dict['GT_labels'][ind]]))
                else:
                    better_method = METHODS_2_COMPARE[-1*(advantage_of_method_num-1)]
                    worse_method = METHODS_2_COMPARE[advantage_of_method_num]
                    plt.suptitle('False %s example where %s (%.2e,%d) is better than %s (%.2e,%d). GT: %s'%(example_type,better_method,
                                                                                                            metrics_dict[better_method][ind],np.argsort(scores_orders[-1*(advantage_of_method_num-1)])[ind],
                                                                                                            worse_method,metrics_dict[worse_method][ind],np.argsort(scores_orders[advantage_of_method_num])[ind],labels[outputs_dict['GT_labels'][ind]]))
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()

if CURVE_TYPE=='2Dhists':
    example_utils.Hist_Uncertainty(metrics_dict,np.argmax(metrics_dict[EFFECTIVE_POSTERIORS],-1)!=GT_labels,score_2_show=HIST_SCORES_X,determining_score=HIST_SCORES_Y,
                                   type='ratio_pos_neg_2Dhist',PCA=PCA_method)
    plt.savefig(os.path.join(FIGURES_FOLDER,'Model%s_%s%s_by_%s_hist.png'%(full_model_name,'PCA_' if PCA_method else '',HIST_SCORES_X,HIST_SCORES_Y)))
    plt.clf()

if CURVE_TYPE not in COMPARISON_CURVE_TYPES and False:
    metrics_dict['Merged'] = example_utils.PartialAgreementMerge(leading_scores=-1*metrics_dict['T_ensemble_MSR'],secondary_scores=metrics_dict['weighted_independent_prob_model'],
                                                                 leading_scores_blend_range=[-.9,-0.53],take_secondary=True,range_corresponds_2_secondary=False)
    METHODS_2_COMPARE += ['Merged']

if CURVE_TYPE not in ['comparison','diff','incremental','excess','accum_diff','param_search','excess_savelog','num_transformations_performance','AORC_savelog','AORC']:
    sys.exit(0)

if 'mandelbaum_scores' in METHODS_2_COMPARE:
    metrics_dict['mandelbaum_scores'] = outputs_dict['mandelbaum_scores']
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
        reordered_dict = {'logits':outputs_dict['logits'][:,:num_classes]}
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
        reordered_dict = {'logits':np.concatenate([outputs_dict['logits'][:,i*num_classes:(i+1)*num_classes] for i in methods_order],1)}
        valid_fold_samples = np.ones([reordered_dict['logits'].shape[0],1]).astype(np.bool)
    per_num_transformations_AORC = []
    for num_Ts in tqdm(range(len(methods_order))):
        cur_AORC = []
        if AORC_ORDER=='learned':
            if num_Ts>0:
                temp_dict['logits'] = np.concatenate([reordered_dict['logits'],outputs_dict['logits'][:,methods_order[num_Ts]*num_classes:(methods_order[num_Ts]+1)*num_classes]],1)
        else:
            temp_dict['logits'] = reordered_dict['logits'][:,:num_classes*(1+num_Ts)]
        temp_Ts = [TRANSFORMATIONS_LIST[i-1] for i in (chosen_Ts+[methods_order[num_Ts]]) if i>0]
        for fold_num in range(num_folds):
            # valid_fold_samples = np.random.uniform(low=0,high=1,size=reordered_dict['logits'].shape[0])<0.5#0.9#0.8,2/3
            per_fold_dict = {'logits':temp_dict['logits'][valid_fold_samples[:,fold_num],:]}
            cur_metrics = example_utils.Logits_2_Metrics(per_fold_dict,num_classes=num_classes,effective_posteriors=EFFECTIVE_POSTERIORS,desired_metrics=['T_ensemble_MSR'],
                                                         temperatures=TEMPERATURE,transformations=temp_Ts,top5=TOP5,models_ensemble=SPECIFIC_MODEL=='ensemble',verbosity=False)
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
    plt.title('%s by %s (acc.=%.3f) on %s'%('Amit model: %s,'%(SPECIFIC_MODEL) if CLASSIFIER_2_USE=='Amit' else '%s%s,'%('Top-5 ' if TOP5 else '',CLASSIFIER_2_USE),EFFECTIVE_POSTERIORS,
                                            1-detection_labels.mean(),HYPER_PARTITION))
    plt.savefig(os.path.join(FIGURES_FOLDER,'%sModel_%s%s%s_%s_on_%s_Methods_%s.png'%('Num_Ts_',full_model_name,'top5' if TOP5 else '','modelEns' if SPECIFIC_MODEL=='ensemble' else '',
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

example_utils.Create_RC_figure(risk=risk, coverage=coverage, score=AORC,methods_names=METHODS_2_COMPARE,temperature_dict=TEMPERATURE,comparison_fig=CURVE_TYPE in COMPARISON_CURVE_TYPES,
                               curve_type=CURVE_TYPE,title='%s by %s (acc.=%.3f) on %s'%('Amit model: %s,'%(SPECIFIC_MODEL) if CLASSIFIER_2_USE=='Amit' else '%s%s,'%('Top-5 ' if TOP5 else '',CLASSIFIER_2_USE),EFFECTIVE_POSTERIORS,
                                                                                         1-detection_labels.mean(),HYPER_PARTITION),paper_mode=PAPER_FIG_MODE)
plt.savefig(os.path.join(FIGURES_FOLDER,'%sModel_%s%s%s_%s_on_%s_Methods_%s.png'%((CURVE_TYPE+'_') if CURVE_TYPE is not None else '',full_model_name,'top5' if TOP5 else '','modelEns' if SPECIFIC_MODEL=='ensemble' else '',
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
                                          SPECIFIC_MODEL=='ensemble',HYPER_PARTITION,'AORC' not in CURVE_TYPE])))

