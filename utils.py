import os
import csv
import re
import numpy as np

SAVED_LOGITS_FOLDER = '/home/ybahat/data/ErrorDetection/Saved_logits'



def Locate_Saved_logits(csv_file_path,saved_logits_file_name,model_version,desired_transformations,use_all_saved_transformations):
    logits_files_2_load = []
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