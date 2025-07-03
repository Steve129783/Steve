Supervised Learning models: 1. CNN, 2. CVAE

What we input:
1. Three datasets, 0: "correct", 1: "high", 2: "low"
These datasets includs 50x50 patches from components with different print energy.
![image](https://github.com/user-attachments/assets/25ca3d79-7ec1-415a-9f78-1aaa46326e8f)

Current objectives:
1. Check CNN and CVAE latent space 
2. Optimise the hyperparameters

Tasks
1. Use high defect sensitivity channels to classify the patches.
____________________________________________________________________________________________________________
## Code name: 0_image_save.py
1. Read .tif files and separate them to different folders
2. output padded full images (200x200) and patches (50x50).

Output: 
1. ![352_centered_200x200](https://github.com/user-attachments/assets/37ae058c-41c8-4413-9809-4c5417b8c795)

2. ![000_x50_y50](https://github.com/user-attachments/assets/faa78a56-f36e-4a4d-8ba9-bdbf1f29ee84)

____________________________________________________________________________________________________________
## Code name: 1_Pack.py
1. Read patches from folders and label them by their folders’ name
2. Pack them together to a .pt file 

Output:
1. E:\Project_CNN\2_Pack\data_cache.pt
![image](https://github.com/user-attachments/assets/f2712475-3c81-4388-91e9-9292b0b3590e)


____________________________________________________________________________________________________________
## Code name: 2_CNN_model.py
1. The training script reads group_ids from the data_cache.pt file, counts the unique values, and automatically
   sets n_classes accordingly (in your case, 3 classes).
3. We split the dataset with ratios 70% train, 15% validation, and 15% test to help prevent overfitting and
   ensure a reliable estimate of generalization.
4. CNN use two layers, Conv Block 1:(1, 50, 50) - (16, 25, 25)，
                       Conv Block 2:(16, 25, 25) - (32, 12, 12),
   then flatten the 32x12x12 = 4608 to 128 dimensions (hidden fully-connected layer similar to latent space of
   VAE but does not have the ability to generate and interpolate) then classify the group of patches.
5. After training with early stopping on the validation set, we evaluate final performance on the held-out test
   set and report test accuracy.

Output:
1. Test Accuracy: 0.9654
2. best_model.pth
____________________________________________________________________________________________________________
## Code name: 3_c_vis.py
1. visualize the channels in 2 layers

Output:
1. The encoder layers
![image](https://github.com/user-attachments/assets/2973975c-fb34-4935-8877-cb593daca143)

Figure layer 0

From this figure, some channels are high sensitivity to the defect pixels.

![image](https://github.com/user-attachments/assets/ee1b1f76-281d-4de0-b4c4-bda6ca2ef20c)

Figure layer 1
____________________________________________________________________________________________________________
## Code name: c4_CVAE_model.py
1. The CAVE training script reads information from the data_cache.pt file, and train with the labeled datasets.
2. model automatically separate the dataset to 'train' (0.7), 'val' (0.15) and 'test' (0.15).
3. Three layers Conv2d 0: (1, 50, 50) - (64, 25, 25)
                Conv2d 1: (64, 25, 25) - (128, 12, 12)
                Conv2d 2: (128, 12, 12) - (256, 6, 6)
   Then flatten 256x6x6 = 9216 to 128 latent dimensions.
4. The classification head (self.classifier) takes μ as input and outputs three logits; these logits are passed
   through a softmax to produce the probabilities for groups 0, 1, and 2, and the index with the highest
   probability is chosen as the final predicted class.

Output: 
1. Epoch 21/50 | Train Acc=0.9954 | Val Acc=0.9802 | Val MSE=0.009520 | Val KL=2.2877
    Top 5 KL dims (dim, kl): [(55, 0.9330708384513855), (5, 0.7913203835487366), (6, 0.009043470025062561), (118, 0.008738156408071518), 
(93, 0.008143114857375622)]
Early stopping triggered at epoch 21.
Test Acc=0.9821 | Test MSE=0.009161

model has better classification performance than CNN.

2. save best weight best_model.pth
____________________________________________________________________________________________________________
## Code name: 5_latent_cluster.py
1. input data_cache.py and import the best_model.pth to extract only μ (ignore log variance logσ^2).
2. Use PCA to reduce the dimension of latent space, then visualize the map.

Output:
1. ![image](https://github.com/user-attachments/assets/99d13631-205c-42cc-bd79-62061bcf6e0e)
Figure 1

From this image, the model clustering function is influenced by the 'edge' and 'corner' features which are caused 
by the padding, if we only cluster patches depending on the featuers from defects, this problem can be avoided.

3. ![image](https://github.com/user-attachments/assets/2c76df61-0400-4ac9-8a5a-4079e032d6d8)
Figure 2
____________________________________________________________________________________________________________
## Code name: Additional functions (not been implemented)
1. Add more datasets
2. Limit the model extracts the feature map only depends on the defect features.

















