import numpy as np
import re
import os
from math import factorial
import copy

SAVED_DRAWN_RANDOMS_FOLDER = '/home/ybahat/data/ErrorDetection/Saved_Drawn_Randoms'
RELIABILITY_SCORE_METHODS = ['Transformations_MSR','original_MSR','Transformations_MSR_BS','perT_MSR','.*_MSR']
UNCERTAINY_SCORE_METHODS = ['MLP_detection']


def Process_logits(outputs_dict,num_classes,effective_posteriors,desired_metrics,temperatures=1.,transformations=None,top5=False,models_ensemble=False,verbosity=True):
    BOOTSTRAPING_RESAMPLES_PORTION = 0.001
    ALTERNATIVE_CLASSES_TOP_SUPPORTING_TRANSFORMATIONS_FRACTION = 0.1
    SLIDING_WIN_SIZE_REL_2_NUM_BOOTSTRAPING_RESAMPLES = 1#10
    NUM_BOOTSTRAPPING_RESAMPLES_RANGE = [100,1000]
    assert outputs_dict['logits'].shape[1]==num_classes*(1+len(transformations)),'Logits shape does not match # of transformations'
    non_tranformed_Ts_indexes = [0]
    if models_ensemble:
        logits = outputs_dict['logits']
        corresponding_transfomrations = copy.deepcopy(transformations)
        non_tranformed_Ts_indexes += [i+1 for i,t in enumerate(['+'.join(t) for t in corresponding_transfomrations]) if re.search('model_seed\d$',t) is not None]
        NUM_BOOTSTRAPPING_RESAMPLES_RANGE = [5*v for v in NUM_BOOTSTRAPPING_RESAMPLES_RANGE]
    else:
        logits = np.concatenate([outputs_dict['logits'][:,:num_classes]]+[outputs_dict['logits'][:,(i+1)*num_classes:(i+2)*num_classes] for i,t in enumerate(transformations) if 'model_seed' not in '+'.join(t)],1)
        corresponding_transfomrations = [t for t in transformations if 'model_seed' not in '+'.join(t)]
    logits_of_all_models = np.concatenate([outputs_dict['logits'][:,:num_classes]]+[outputs_dict['logits'][:,(i+1)*num_classes:(i+2)*num_classes] for i,t in enumerate(transformations) if re.search('^model_seed\d$','+'.join(t)) is not None],1)
    desired_metrics = [effective_posteriors]+desired_metrics
    if isinstance(temperatures,float) or isinstance(temperatures,int):
        temperatures = {'default':float(temperatures)}
        default_temperature_metrics = desired_metrics
    else:#temperatures is a dict:
        default_temperature_metrics = [metric for metric in desired_metrics if metric not in temperatures.keys()]
    dict_2_return = {}
    for key in temperatures.keys():
        temperature = temperatures[key]
        if key=='default':
            cur_metrics =  default_temperature_metrics
        else:
            cur_metrics = [key] if key in desired_metrics else []
        if len(cur_metrics)==0:
            continue
        posteriors = np.reshape(Softmax(logits,temperature=temperature,num_classes=num_classes),[logits.shape[0],-1,num_classes])
        if 'model_ensemble_MSR' in cur_metrics:
            posteriors_of_all_models = np.reshape(Softmax(logits_of_all_models,temperature=temperature,num_classes=num_classes),[logits_of_all_models.shape[0],-1,num_classes])
        num_transformations = posteriors.shape[1]
        if num_transformations>30:
            num_bootstraping_resamples = NUM_BOOTSTRAPPING_RESAMPLES_RANGE[1]
        else:
            num_bootstraping_options = factorial(2*num_transformations-1)/factorial(num_transformations)/factorial(num_transformations-1)
            num_bootstraping_resamples = int(np.minimum(NUM_BOOTSTRAPPING_RESAMPLES_RANGE[1],np.maximum(NUM_BOOTSTRAPPING_RESAMPLES_RANGE[0],num_bootstraping_options*BOOTSTRAPING_RESAMPLES_PORTION)))
        def Bootstraping_Ranking(bootstraped_reliability,base_posteriors=None):
            def Sliding_Window_Plurality(votes,num_candidates=posteriors.shape[0],win_size=int(SLIDING_WIN_SIZE_REL_2_NUM_BOOTSTRAPING_RESAMPLES*num_bootstraping_resamples)):
                already_picked = np.zeros([num_candidates]).astype(np.bool)
                cur_maximum = ('',0);   myMap = {}; resulting_order = [];   remaining_win_votes = 1*win_size;   votes_ind = -1
                num_votes = len(votes)
                while True:
                    while remaining_win_votes>0 and (votes_ind+1)<num_votes: # The second condition is for the case of using a window larger than num_bootstraping_resamples, in which the lowest ranking votes are picked from the same window (which cannot be slide any further)
                        votes_ind += 1
                        cur_vote = votes[votes_ind]
                        if already_picked[cur_vote]:    continue
                        remaining_win_votes -= 1
                        if cur_vote in myMap.keys():    myMap[cur_vote] += 1
                        else:   myMap[cur_vote] = 1
                        if myMap[cur_vote]>cur_maximum[1]:  cur_maximum = (cur_vote,myMap[cur_vote])
                    resulting_order.append(cur_maximum[0])
                    already_picked[cur_maximum[0]] = True
                    if np.all(already_picked):  break
                    remaining_win_votes = cur_maximum[1]
                    myMap.pop(cur_maximum[0])
                    cur_maximum = max(myMap.items(),key=lambda x:x[1]) if len(myMap.items())>0 else ('',0)
                return resulting_order

            candidates = (np.arange(posteriors.shape[0]).reshape([-1,1])*np.ones([1,num_bootstraping_resamples])).astype(int).reshape([-1])
            reliability_order = Sliding_Window_Plurality(candidates[np.argsort(bootstraped_reliability.reshape([-1]))[::-1]])
            return np.linspace(1,0,posteriors.shape[0])[np.argsort(reliability_order)]

        for metric in cur_metrics:
            if verbosity:
                print('Computing metric: %s'%(metric))
            if metric=='T_ensemble_posteriors':
                # Calculating average posteriors (fbar):
                if models_ensemble:
                    dict_2_return[metric] = np.mean(posteriors, 1)
                else:
                    dict_2_return[metric] = np.mean(np.stack([posteriors[:,0,:]]+[posteriors[:,i+1,:] for i,t in enumerate(transformations) if 'model_seed' not in '+'.join(t)],1), 1)
            elif re.match('.*_MSR$',metric) is not None and re.match('.*(?=_MSR$)',metric).group(0) in ['original']+['+'.join(t) for t in corresponding_transfomrations]:
                # MSR according to any of the image versions (including the original, non-transformed image:
                full_t_list = ['original']+['+'.join(t) for t in corresponding_transfomrations]
                if models_ensemble:
                    full_t_list = ['original' if re.search('model_seed\d$',t) is not None else t for t in full_t_list]
                assert len(full_t_list)==posteriors.shape[1],'Posteriors size not as expected. Check why'
                T_num = [i for i,t in enumerate(full_t_list) if t==re.match('.*(?=_MSR$)',metric).group(0)]
                posteriros_of_desired_T = np.mean(posteriors[:,T_num,:],1)
                if top5:
                    dict_2_return[metric] = np.sum(posteriros_of_desired_T[np.arange(posteriors.shape[0]).reshape([-1, 1]),
                                                                           np.argpartition(posteriros_of_desired_T[:,:], kth=-5,axis=-1)[:, -5:].reshape([-1, 5])], -1)
                else:
                    dict_2_return[metric] = np.max(posteriros_of_desired_T,-1)
            elif metric == 'Transformations_MSR':
                # MSR of averaged posteriors:
                posteriors_mean = np.mean(posteriors, 1)
                if top5:
                    dict_2_return[metric] = np.sum(posteriors_mean[np.arange(posteriors.shape[0]).reshape([-1, 1]),
                                                                   np.argpartition(effective_posteriors, kth=-5, axis=-1)[:, -5:].reshape([-1, 5])], -1)
                else:
                    # dict_2_return[metric] = np.max(posteriors_mean, -1)
                    dict_2_return[metric] = posteriors_mean[np.arange(posteriors.shape[0]),np.argmax(effective_posteriors,-1)]
            elif metric=='model_ensemble_MSR':
                assert not models_ensemble,'When in mocel ensemble more, original_MSR is actually model_ensemble_MSR, so no reason to calculate it again'
                # MSR of averaged posteriors from differently seeded models (MEthod by Lakshminarayanan):
                posteriors_mean = np.mean(posteriors_of_all_models, 1)
                if top5:
                    dict_2_return[metric] = np.sum(posteriors_mean[np.arange(posteriors.shape[0]).reshape([-1, 1]),
                                                                   np.argpartition(effective_posteriors, kth=-5, axis=-1)[:, -5:].reshape([-1, 5])], -1)
                else:
                    # dict_2_return[metric] = np.max(posteriors_mean, -1)
                    dict_2_return[metric] = posteriors_mean[np.arange(posteriors.shape[0]),np.argmax(effective_posteriors,-1)]
            elif metric=='Transformations_MSR_BS':
                resampled_indexes = Draw_Bootstraping_Sampling_Pattern(num_bootstraping_resamples,posteriors.shape[1])
                if top5:
                    bootstraped_T_MSR = posteriors[np.arange(posteriors.shape[0]).reshape([-1, 1, 1,1]),np.expand_dims(np.expand_dims(resampled_indexes,0),-1),
                                                   np.argpartition(effective_posteriors, kth=-5,axis=-1)[:,-5:].reshape([-1, 1, 1,5])]
                    bootstraped_T_MSR = np.mean(np.sum(bootstraped_T_MSR,-1),2)
                else:
                    bootstraped_T_MSR = np.mean(posteriors[np.arange(posteriors.shape[0]).reshape([-1,1,1]),np.expand_dims(resampled_indexes,0),np.argmax(effective_posteriors,-1).reshape([-1,1,1])],2)
                dict_2_return[metric] = Bootstraping_Ranking(bootstraped_T_MSR,base_posteriors=effective_posteriors)
            elif metric == 'original_posteriors':
                # Returning posteriors of original image:
                dict_2_return[metric] = np.mean(posteriors[:, non_tranformed_Ts_indexes, :],1)
            elif metric in ['mandelbaum_scores','MLP_detection']:
                continue # This metric is calculated outside this function
            else:
                raise Exception('Unsupported metric %s'%(metric))
            if isinstance(effective_posteriors,str):# This would happen in the first iteration (metric), so that other metrics can use this info:
                effective_posteriors = dict_2_return[effective_posteriors]
    return dict_2_return

def Softmax(logits, temperature=1.,num_classes=None):
    USE_FLOAT64 = True
    TEMP_WORKAROUND = False
    assert len(logits.shape)==2,'Unrecognized logits shape'
    if logits.shape[0]==0:
        return logits
    if USE_FLOAT64:
        soft_max = np.minimum(np.finfo(np.float64).max, np.exp(logits.astype(np.float64) / np.array(temperature).reshape([-1, 1])))
    else:
        soft_max = np.minimum(np.finfo(np.float32).max,np.exp(logits/np.array(temperature).reshape([-1,1])))
    if num_classes is not None and num_classes!=logits.shape[1]:
        soft_max = soft_max.reshape([soft_max.shape[0],-1,num_classes])
    if USE_FLOAT64:
        if TEMP_WORKAROUND:
            soft_max = soft_max / np.maximum(np.finfo(np.float64).tiny, np.minimum(np.finfo(np.float64).max,np.sum(soft_max, axis=-1,keepdims=True)))
            soft_max = soft_max / np.maximum(np.finfo(np.float64).tiny, np.sum(soft_max, axis=-1,keepdims=True))  # Normalizing again for the case where the actual sum used for normalization overflowed, and the minimum above was used.
        else:
            soft_max = soft_max / np.maximum(np.finfo(np.float64).eps,np.minimum(np.finfo(np.float64).max, np.sum(soft_max, axis=-1, keepdims=True)))
            soft_max = soft_max / np.maximum(np.finfo(np.float64).eps, np.sum(soft_max, axis=-1,keepdims=True))  # Normalizing again for the case where the actual sum used for normalization overflowed, and the minimum above was used.
    else:
        soft_max = soft_max/np.maximum(np.finfo(np.float32).eps,np.minimum(np.finfo(np.float32).max,np.sum(soft_max,axis=-1,keepdims=True)))
        soft_max = soft_max/np.maximum(np.finfo(np.float32).eps,np.sum(soft_max,axis=-1,keepdims=True))#Normalizing again for the case where the actual sum used for normalization overflowed, and the minimum above was used.
    if num_classes is not None and num_classes!=logits.shape[1]:
        soft_max = soft_max.reshape(logits.shape)
    return soft_max

def Draw_Bootstraping_Sampling_Pattern(num_resamples,num_transformations):
    # Using this function instead of random bootstrapping resampling to allow reproducing results
    resampled_indexes = np.empty([0,0])
    if os.path.isfile(os.path.join(SAVED_DRAWN_RANDOMS_FOLDER, 'bootsraping_sampling.npz')):
        resampled_indexes = np.load(os.path.join(SAVED_DRAWN_RANDOMS_FOLDER, 'bootsraping_sampling.npz'))['resampled_indexes']
    if resampled_indexes.shape[0]<num_resamples or resampled_indexes.shape[1]<num_transformations:
        print('!!! Drawing new random bootstraping resampling pattern !!!')
        old_sampling_data = 1*resampled_indexes
        resampled_indexes = np.random.uniform(low=0, high=1, size=[max(resampled_indexes.shape[0],num_resamples),max(resampled_indexes.shape[1],num_transformations)])  # This random instantiation supprots up to 1000 resamples of up to 100 transformations
        resampled_indexes[:old_sampling_data.shape[0],:old_sampling_data.shape[1]] = old_sampling_data
        np.savez(os.path.join(SAVED_DRAWN_RANDOMS_FOLDER, 'bootsraping_sampling.npz'),resampled_indexes=resampled_indexes)
    return np.round(num_transformations * resampled_indexes[:num_resamples, :num_transformations] - 0.5).astype(np.int)
