import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

def unpickle(file):
    import pickle as cPickle
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data


INPUT_FOLDER = '/home/ybahat/data/ErrorDetection/Amit_related/Amit_features/STL10_Amit'

def main():
    AMIT_MODEL = 'None'
    SVHN_exp = False
    CIFAR100 = False
    score_dist, correct_test_predictions = Calc_IAD_Scores(AMIT_MODEL,SVHN_exp=SVHN_exp,CIFAR100=CIFAR100)
    tested_images_dataset = 'SVHN' if SVHN_exp else 'cifar100' if CIFAR100 else 'STL10_Original'
    np.savez(os.path.join(INPUT_FOLDER+'_'+AMIT_MODEL,'val_scores_%s.npz'%(tested_images_dataset)),scores=score_dist,correct_test_predictions=correct_test_predictions)


def Calc_IAD_Scores(model,SVHN_exp=False,CIFAR100=False,input_folder=INPUT_FOLDER):
    # Loading saved features, predictions and labels:
    input_folder = input_folder+('_'+model.amit_model)
    tested_images_dataset = 'SVHN' if SVHN_exp else 'cifar100' if CIFAR100 else 'STL10_Original'
    file_suffix = "%s_%s%s.pickle" % (model.amit_model, 'val', '_%s'%(tested_images_dataset))

    # if not os.path.isfile(os.path.join(input_folder, "test_features_8000_" + file_suffix)):
    #     outputs_dict = model.Evaluate_Model(set='val', amitFeatures=True)
    test_features = unpickle(os.path.join(input_folder, "test_features_8000_" + file_suffix))
    test_predictions = unpickle(os.path.join(input_folder, "test_predictions_8000_" + file_suffix))
    test_labels = unpickle(os.path.join(input_folder, "test_labels_8000_" + file_suffix))
    file_suffix = "%s.pickle" % (model.amit_model)
    if not os.path.isfile(os.path.join(input_folder, "train_predictions_8000_" + file_suffix)):
        outputs_dict = model.Evaluate_Model(set='train', amitFeatures=True)

    train_predictions = unpickle(os.path.join(input_folder, "train_predictions_8000_" + file_suffix))
    train_labels = unpickle(os.path.join(input_folder, "train_labels_8000_" + file_suffix))
    train_features = unpickle(os.path.join(input_folder, "train_features_8000_" + file_suffix))
    if SVHN_exp or CIFAR100:
        num_novel_samples = int(len(test_predictions) / 2)
    else:
        num_novel_samples = 0
        errorIs = 'novel'

    correct_test_predictions=[0]*len(test_predictions)
    samples2discard = []
    for i in range(len(test_predictions)-num_novel_samples):
        if errorIs=='normal' or np.argmax(test_predictions[i])==np.argmax(test_labels[i]):
            correct_test_predictions[i]=1
        elif errorIs=='ignored':
            samples2discard.append(i)
    if len(samples2discard)>0:
        correct_test_predictions = np.delete(correct_test_predictions,samples2discard)
        test_predictions = np.delete(test_predictions,samples2discard,axis=0)
        test_labels = np.delete(test_labels,samples2discard,axis=0)
        # if not args.MCD:
        test_features = np.delete(test_features, samples2discard, axis=0)
    correct_test_predictions = np.array(correct_test_predictions)
    # if not args.MCD:
    print('Train Data shape is', train_predictions.shape, 'and labels is', train_labels.shape)
    print('Train features shape is', train_features.shape)
    print('Test features shape is', test_features.shape)
    print('Test Data shape is', test_predictions.shape, 'and labels is', test_labels.shape)
    #List with largest value index and the ratio
    ratios = []

    print("Performing nearest neighbour")
    nan_indices = np.isnan(train_features)
    train_features[nan_indices] = 0
    nan_indices = np.isnan(test_features)
    test_features[nan_indices] = 0
    nbrs = NearestNeighbors(n_neighbors=500, algorithm='ball_tree').fit(train_features)
    print("Finding closest members test")
    distances_test, indices_test = nbrs.kneighbors(test_features)
    score_dist = []
    for i in trange(len(test_labels)):
        pos_score = 1e-6
        neg_score = 1e-6
        for j in range(len(indices_test[0])):
            if np.argmax(test_predictions[i])==np.argmax(train_labels[indices_test[i][j]]):
                if distances_test[i,j]==0:
                    pos_score += 1
                else:
                    pos_score += np.exp(-distances_test[i,j])
            else:
                if distances_test[i,j]==0:
                    neg_score += 1
                else:
                    neg_score += np.exp(-distances_test[i,j])
        score_dist.append(pos_score/neg_score)
    score_dist = np.array(score_dist)
    return score_dist,correct_test_predictions

if __name__=='__main__':
    main()