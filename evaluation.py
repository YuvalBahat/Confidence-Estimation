import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

def Calculate_Scores(detection_scores,detection_labels,curve_type):
    assert curve_type in ['eAURC','AORC','AUPR','AUROC']
    if not isinstance(detection_scores,list):
        detection_scores = [detection_scores]
    if not isinstance(detection_labels, list):
        detection_labels = len(detection_scores)*[detection_labels]
    scores = []
    if curve_type in ['eAURC','AORC']:
        risk,coverage = [],[]
        for i in range(len(detection_scores)):
            increasing_uncertainty_order = np.argsort(detection_scores[i])
            coverage.append(np.linspace(0,1,num=len(detection_scores[i])))
            risk.append(np.cumsum(detection_labels[i][increasing_uncertainty_order]))
            risk[-1] = risk[-1]/len(coverage[-1])
            if any([exp in curve_type for exp in ['eAURC','AORC']]):
                minimal_risk = np.cumsum(np.sort(detection_labels[i]))/len(coverage[-1])
                maximal_risk = np.cumsum(np.sort(detection_labels[i])[::-1])/len(coverage[-1])
                if 'AORC' in curve_type:
                    scores.append(np.mean(maximal_risk - risk[-1])/np.mean(maximal_risk-minimal_risk))
                else:
                    scores.append(np.mean(risk[-1]-minimal_risk))
            elif curve_type=='incremental':
                scores.append(np.mean(1-risk[-1]))
                risk[-1] = np.diff(risk[-1])
                coverage[-1] = coverage[-1][:-1]
            else:
                scores.append(np.mean(1-risk[-1]))
        return risk,coverage,scores
    elif curve_type=='AUPR':
        precision,recall = [],[]
        for i in range(len(detection_scores)):
            temp = metrics.precision_recall_curve(y_true=detection_labels[i].astype(int),probas_pred=detection_scores[i])
            precision.append(temp[0])
            recall.append(temp[1])
            scores.append(metrics.average_precision_score(y_true=detection_labels[i].astype(int),y_score=detection_scores[i]))
        return precision,recall,scores
    elif curve_type=='AUROC':
        fpr,tpr = [],[]
        for i in range(len(detection_scores)):
            temp = metrics.roc_curve(y_true=detection_labels[i].astype(int),y_score=detection_scores[i])
            fpr.append(temp[0])
            tpr.append(temp[1])
            scores.append(metrics.roc_auc_score(y_true=detection_labels[i].astype(int),y_score=detection_scores[i]))
        return tpr,fpr,scores

def Create_Eval_Figure(risk, coverage, score,methods_names,temperature_dict,curve_type,title):
    LINE_WIDTH = 2
    methods_order = np.argsort(score)
    if curve_type!='eAURC':
        methods_order = methods_order[::-1]
    score_diffs = [score[methods_order[i]] - score[methods_order[i + 1]] for i in range(len(methods_order) - 1) if
                   methods_order[i + 1] < len(score)] + [0]
    plt.clf()
    print('Sorted metric scores:')
    for j,i in enumerate(methods_order):
        plt.plot(coverage[i], risk[i],linewidth=LINE_WIDTH)
        print('%s: %.4f (+%.2e)'%(methods_names[i],score[i],score_diffs[j]))
    plt.legend(['%s (T=%.2f,%+.2e): %.4f' % (
        methods_names[i], temperature_dict[methods_names[i]] if methods_names[i] in temperature_dict.keys() else 1,
        score_diffs[j], score[i]) for j, i in enumerate(methods_order)])
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title(title)
    plt.savefig('confidence_estimation_performance.png')