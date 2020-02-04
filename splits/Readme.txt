`<train/val/test>_imgs.txt` contains ids of webpage images that belong to these splits.
Splits are created such that webpage images from about 748 domains are for training, 187 domains for validation, and 234 domains for testing.
This results in 4267 webpage images in train, 1982 in val, 1673 in test
The webpage domains in train, val, test splits are disjoint sets.
This will help us to understand the power of the Model to generalize over webpages of unseen domains!

trainval_imgs.txt contains train_imgs and val_images combined in a single file
Trainval Images Statistics:
    Mean: [0.8992, 0.8977, 0.8966]
    Std: [0.2207, 0.2166, 0.2217]

domain_wise_imgs contains 1169 .txt files (each corresponding to a domain) and the file contains ids of webpage images that belong to that particular domain.
These files can be used to calculate macro accuracy for train/val/test data.