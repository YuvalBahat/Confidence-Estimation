# Confidence-Estimation
Code for the paper "[Classification Confidence Estimation with Test-Time Data-Augmentation](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2006.16705&sa=D&sntz=1&usg=AFQjCNHsVS_Un8lNBoUHpe5N49_PnfR9vA)", by [Yuval Bahat](https://sites.google.com/view/yuval-bahat/home) and [Greg Shakhnarovich](https://home.ttic.edu/~gregory/).

### Usage insturctions:
1.Update the desired experiment parameters in main.py  
2.Update the relevant dataset path in utils.py  
3.Run main.py  

### Subsets Partition
The partitions of the different validation datasets into evaluation and experimental subsets are defined in the npz files inside the datasets_partition folder. This folder contains the randomly drawn bootstrapping resampling indices, to allow reproducing the results in the paper. 

If you find our work useful in your research or publication, please cite it:

```
@article{bahat2020classification,
  title={Classification confidence estimation with test-time data-augmentation},
  author={Bahat, Yuval and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:2006.16705},
  year={2020}
}
```
