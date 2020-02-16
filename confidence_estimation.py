import re
import numpy as np


RELIABILITY_SCORE_METHODS = ['T_ensemble_MSR','mandelbaum_scores','original_MSR','T_ensemble_mean_MSR','T_ensemble_consensus','T_MSR_gap_2_2nd','T_MSR_gap_STD_ratio',
                             'T_ensemble_geometric_MSR','T_MSR_bootstraping','Independent_prob_model_bootstraping','weighted_independent_prob_model_bootstraping','perT_MSR','.*_MSR']
UNCERTAINY_SCORE_METHODS = ['T_ensemble_variance','KLD_.*','T_ensemble_entropies','choose_\d\.(\d)+_variance','T_ensemble_logits_variance','choose_\d\.(\d)+_logits_variance', \
                            'T_ensemble_variance_MSR','T_ensemble_bias_proxy','T_ensemble_bias_proxy_and_var','T_ensemble_dist_var','T_ensemble_var_2_oneHot', \
                            'T_ensemble_bias_proxy_max','T_ensemble_var_2_oneHot_choose_\d\.(\d)+_STD','T_ensemble_MSR_choose_\d\.(\d)+_STD', \
                            'T_ensemble_MSR_choose_\d\.(\d)+_STD_normed','T_ensemble_var_2_oneHot_choose_\d\.(\d)+_STD_normed','T_ensemble_variance_normed', \
                            'T_ensemble_STD','T_ensemble_STD_normed','PCA','Grad','REAL_T_ensemble_bias','Independent_prob_model','Gaussian_Model_Probs',
                            'Independent_per_T_prob_model','weighted_independent_prob_model','MLP_detection','KST_.*','original_manipulated_entropy','original_entropy']


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
