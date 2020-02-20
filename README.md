# Confidence-Estimation
Code for the submitted manuscript titled "Classification Confidence Estimation with Test-Time Data-Augmentation"

### Usage insturctions:
1.Update the desired experiment parameters in main.py  
2.Update the relevant dataset path in utils.py  
3.Run main.py  

### Subsets Partition
The partitions of the different validation datasets into evaluation and experimental subsets are defined in the npz files inside the datasets_partition folder. This folder contains the randomly drawn bootstrapping resampling indices, to allow reproducing the results in the paper. 
