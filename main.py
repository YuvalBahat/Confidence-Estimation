from utils import *
from confidence_estimates import Process_logits,RELIABILITY_SCORE_METHODS,UNCERTAINY_SCORE_METHODS
from evaluation import *

# Experiment parameters:
CURVE_TYPE = 'AORC'#'eAURC','AORC','AUPR','AUROC'
TOP5 = False
METHODS_2_COMPARE = ['Transformations_MSR','original_MSR','Transformations_MSR_BS']
TEMPERATURE = {'default':1.}
CLASSIFIER_2_USE = 'CIFAR10'#'WideResNet','ResNet18','ResNext','AlexNet','SVHN','CIFAR10','CIFAR100'
HYPER_PARTITION = 'test'#'train','test'

MODEL_VERSION = 'None'#'None','AT','dist','Seed0_no_augm','flipOnly','flipOnlyHalfData','Seed1','Seed2','ensemble','Seed0_untrained','Seed0_no_flip'
EFFECTIVE_POSTERIORS = 'original_posteriors'#'T_ensemble_posteriors','original_posteriors','T_bootstraping_posteriors'
used_dataset = 'ImageNet' if CLASSIFIER_2_USE in ['ResNet18','ResNext','AlexNet'] else 'svhn' if CLASSIFIER_2_USE=='SVHN' else 'cifar10' if CLASSIFIER_2_USE=='CIFAR10' else 'cifar100' if CLASSIFIER_2_USE=='CIFAR100' else 'stl10'
NUM_CLASSES = {'stl10':10,'ImageNet':1000,'svhn':10,'cifar10':10,'cifar100':100}
IM_SIZE = {'stl10':96,'ImageNet':224,'svhn':32,'cifar10':32,'cifar100':32}

if used_dataset=='cifar100':
    TRANSFORMATIONS_LIST = [['shift-1_0'], ['shift-1_-1'], ['shift0_1'], ['shift0_-1'], ['shift0_-2'], ['horFlip', 'shift2_0'], ['horFlip', 'shift1_0'], ['horFlip', 'shift2_1'], ['horFlip'], ['horFlip', 'shift2_2'], ['horFlip', 'shift0_2'], ['shift-1_-2'], ['shift0_-3'], ['horFlip', 'shift0_-2'], ['shift1_2'], ['horFlip', 'shift1_-3'], ['shift2_1'], ['horFlip', 'shift4_0'], ['horFlip', 'shift0_-3'], ['horFlip', 'shift2_-3'], ['shift1_3'], ['horFlip', 'shift-3_0'], ['horFlip', 'shift6_0'], ['horFlip', 'shift0_-5'], ['horFlip', 'shift4_4'], ['horFlip', 'shift0_5'], ['horFlip', 'shift-5_-3'], ['shift0_5'], ['horFlip', 'shift7_0'], ['horFlip', 'shift5_-5'], ['rotate3'], ['shift10_10'], ['horFlip', 'shift5_-11'], ['shift9_11'], ['shift-11_-9'], ['horFlip', 'shift11_-11'], ['horFlip', 'shift12_12'], ['shift14_14'], ['horFlip', 'shift14_14']]
elif used_dataset=='cifar10':
    TRANSFORMATIONS_LIST = [['shift1_-1'], ['shift0_-1'], ['shift1_0'], ['horFlip', 'shift0_-1'], ['shift-1_0'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['shift1_1'], ['horFlip', 'shift-2_-1'], ['horFlip', 'shift1_-2'], ['shift-3_1'], ['horFlip', 'shift2_0']]
elif used_dataset=='svhn':
    TRANSFORMATIONS_LIST = [['shift5_0'], ['shift4_0'], ['shift5_4'], ['shift5_2'], ['shift-3_-5'], ['shift-5_-3'], ['shift8_0'], ['shift8_8'], ['shift9_0'], ['shift9_9'], ['shift13_13'], ['shift14_14'], ['shift12_12'], ['shift10_10'], ['shift10_0'], ['shift11_0']]
elif used_dataset=='stl10':
    if CLASSIFIER_2_USE=='ELU':
        TRANSFORMATIONS_LIST = [['gamma12'], ['zoomin95'], ['shift0_-1'], ['horFlip', 'zoomin95'], ['shift0_-5'], ['shift1_0'], ['horFlip', 'shift1_1'], ['shift0_-7'], ['zoomin95', 'shift0_5'], ['horFlip', 'shift0_-1'], ['rotate5'], ['horFlip', 'shift0_-5']]
    else:
        TRANSFORMATIONS_LIST = [['shift-6_6'], ['shift0_10'], ['shift-4_6'], ['horFlip', 'shift-4_6'], ['horFlip', 'shift-3_6'], ['horFlip', 'shift4_6'], ['horFlip', 'shift5_-2'], ['horFlip', 'shift5_6'], ['horFlip', 'shift6_-3'], ['horFlip', 'shift7_-4'], ['shift0_12'], ['horFlip', 'shift-4_-2'], ['shift-9_9'], ['horFlip', 'shift4_1'], ['shift0_14'], ['horFlip', 'shift0_12'], ['horFlip', 'shift12_12'], ['horFlip', 'shift7_0'], ['horFlip', 'shift12_0'], ['gamma8'], ['shift14_14'], ['horFlip', 'shift13_0'], ['horFlip', 'shift14_0'], ['shift0_18'], ['gamma6'], ['horFlip', 'shift19_0'], ['shift0_22'], ['horFlip', 'gamma6'], ['horFlip', 'rotate-1'], ['horFlip', 'rotate1'], ['rotate1'], ['shift22_22'], ['horFlip', 'rotate-7']]
elif used_dataset=='ImageNet':
    TRANSFORMATIONS_LIST = [['gamma9'], ['gamma8'], ['shift0_1'], ['shift0_2'], ['rotate1'], ['rotate4'], ['horFlip', 'shift0_1'], ['shift-3_0'], ['gamma6'], ['horFlip', 'shift1_0'], ['zoomin92'], ['zoomin95', 'gamma8'], ['horFlip', 'shift-1_2'], ['horFlip', 'shift0_5'], ['horFlip', 'shift2_0'], ['rotate5'], ['horFlip', 'shift-5_1'], ['horFlip', 'shift0_10'], ['horFlip', 'shift0_-10'], ['shift0_-10'], ['horFlip', 'zoomin90'], ['rotate7'], ['horFlip', 'rotate-10']]

assert CLASSIFIER_2_USE in ['WideResNet','ResNet18','ResNext','AlexNet','SVHN','CIFAR10','CIFAR100']
assert not TOP5 or CLASSIFIER_2_USE in ['ResNet18','ResNext','AlexNet'],'Why usie top-5 accuracy for datasets other than ImageNet?'
num_classes = NUM_CLASSES[used_dataset]
assert all([RegExp_In_List(method,RELIABILITY_SCORE_METHODS+UNCERTAINY_SCORE_METHODS,exact_match=True) for method in METHODS_2_COMPARE]),'Unknown confidence estimation method'

np.random.seed(0)
full_model_name = CLASSIFIER_2_USE+('_%s'%(MODEL_VERSION) if (CLASSIFIER_2_USE=='WideResNet' and MODEL_VERSION in STL10_MODEL_VERSIONS) else '')
if CLASSIFIER_2_USE != 'WideResNet':
    MODEL_VERSION = MODEL_VERSION.replace('ensemble','')# Should not have specific_model=ensemble unless using an ensemble. Might cause truble when looking for saved logits content'

desired_Ts = ['+'.join(t) for t in TRANSFORMATIONS_LIST]
if MODEL_VERSION=='ensemble':
    desired_Ts = [m+t for m in (['']+['model_seed%d_'%(i+1) for i in range(4)]) for t in desired_Ts]+['model_seed%d'%(i+1) for i in range(4)]
print('Using %d different transformations'%(len(desired_Ts)))
TRANSFORMATIONS_LIST = [T.split('+') for T in desired_Ts]

saved_logits_file_name = 'saved_logits_%s'%(full_model_name)
saved_logits_file_name = saved_logits_file_name.replace('_ensemble','')
logits_files_2_load,found_saved_Ts = Locate_Saved_logits(os.path.join(SAVED_LOGITS_FOLDER,SAVED_LOGITS_CSV_FILENAME+'.csv'),saved_logits_file_name,MODEL_VERSION,desired_Ts)
if 'perT_MSR' in METHODS_2_COMPARE:
    METHODS_2_COMPARE = [m for m in METHODS_2_COMPARE if m!='perT_MSR']+['%s_MSR'%('+'.join(t)) for t in TRANSFORMATIONS_LIST]

classifier_output_dict = Load_Saved_Logits(logits_files_2_load,found_saved_Ts,desired_Ts,num_classes,models_ensemble_mode=MODEL_VERSION=='ensemble')

if any(found_saved_Ts==0):#Need to run classifier as some required logits (corresponding to certain transformed image versions) are missing:
    classifier_output_dict = Infer_Logits(classifier_output_dict,desired_Ts,found_saved_Ts,used_classifier=CLASSIFIER_2_USE,model_version=MODEL_VERSION,
                                          used_dataset=used_dataset,num_classes=num_classes,saved_logits_file_name=saved_logits_file_name)

classifier_output_dict['Transformations'] = TRANSFORMATIONS_LIST
classifier_output_dict = Subset_Partition(classifier_output_dict,used_dataset,HYPER_PARTITION)
GT_labels = classifier_output_dict['GT_labels']
if TOP5:
    per_version_accuracy = [np.mean(np.any(np.argpartition(classifier_output_dict['logits'][:,(i)*num_classes:(i+1)*num_classes],kth=-5,axis=-1)[:,-5:]==GT_labels.reshape([-1,1]),-1)) for i in range(len(TRANSFORMATIONS_LIST)+1)]
else:
    per_version_accuracy = [np.mean(np.argmax(classifier_output_dict['logits'][:,(i)*num_classes:(i+1)*num_classes],-1)==GT_labels) for i in range(len(TRANSFORMATIONS_LIST)+1)]
print('%sAccuracy on original images: %.3f\nOn transformed images (low to high):'%('Top-5 ' if TOP5 else '',per_version_accuracy[0])+
      ''.join(['\n%s: %.3f'%(t,acc) for acc,t in sorted(zip(per_version_accuracy,TRANSFORMATIONS_LIST))]))

metrics_dict = Process_logits(classifier_output_dict,num_classes=num_classes,effective_posteriors=EFFECTIVE_POSTERIORS,
                                              desired_metrics=METHODS_2_COMPARE,temperatures=TEMPERATURE,transformations=TRANSFORMATIONS_LIST,top5=TOP5,models_ensemble=MODEL_VERSION=='ensemble')
if TOP5:
    detection_labels = np.all(np.argpartition(metrics_dict[EFFECTIVE_POSTERIORS],axis=-1,kth=-5)[:,-5:]!=GT_labels.reshape([-1,1]),-1)
else:
    detection_labels = np.argmax(metrics_dict[EFFECTIVE_POSTERIORS],-1)!=GT_labels
print('Final classification accuracy %.3f'%(1-np.mean(detection_labels)))

risk,coverage,AORC = Calculate_Scores(
    detection_scores=[(-1 if RegExp_In_List(method,RELIABILITY_SCORE_METHODS,UNCERTAINY_SCORE_METHODS) else 1)*metrics_dict[method] for method in METHODS_2_COMPARE],
    detection_labels=detection_labels,curve_type=CURVE_TYPE)

fig_title = '%s by %s (acc.=%.3f) on %s'%('%s%s,'%('Top-5 ' if TOP5 else '',CLASSIFIER_2_USE),EFFECTIVE_POSTERIORS,1-detection_labels.mean(),HYPER_PARTITION)

Create_Eval_Figure(risk=risk, coverage=coverage, score=AORC,methods_names=METHODS_2_COMPARE,temperature_dict=TEMPERATURE,curve_type=CURVE_TYPE,title=fig_title)
