### Supervised Learning models: 
1. CNN, 2. VAE\
This training pipeline ensures strict reproducibility of the model training process on identical hardware and
code versions by globally fixing random seeds (Python, NumPy, PyTorch, CUDA), disabling nondeterministic
algorithms (cuDNN, TF32), and locking environment variables (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG).

### What we input:
1. Three datasets, 0: "correct", 1: "high", 2: "low"
   
group 0 corresponding to label：{'correct'}\
group 1 corresponding to labe：{'high'}\
group 2 corresponding to labe：{'low'}

These datasets includs 50x50 patches from components with different print energy.
![image](https://github.com/user-attachments/assets/25ca3d79-7ec1-415a-9f78-1aaa46326e8f)

### Current objectives:
1. Check CNN and VAE latent space 
2. Optimise the hyperparameters (1. increase the reconstruction performance，2. latent space separability)

### Tasks
1. Use high defect sensitivity channels to classify the patches.

### Mistake
1. In the meeting 3rd Jul, I said my model is 'CVAE', that is wrong. That model is a ‘joint VAE–classifier model’.
   The VAE component is trained in an unsupervised manner on the unlabeled data, whereas the classification head
   is supervised with labeled examples to enforce class-specific structure in the latent space.
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
## Code name: c2_CNN_model.py
1. The training script reads group_ids from the data_cache.pt file, counts the unique values, and automatically
   sets n_classes accordingly (in this case, 3 classes).
3. We split the dataset with ratios 70% train, 15% validation, and 15% test to help prevent overfitting and
   ensure a reliable estimate of generalization.
4. CNN use two layers\
   Conv Block 1:(1, 50, 50) - (16, 25, 25)\
   Conv Block 2:(16, 25, 25) - (32, 12, 12)\
   \
   then flatten the 32x12x12 = 4608 to 128 dimensions (hidden fully-connected layer similar to latent space of
   VAE but does not have the ability to generate and interpolate) then classify the group of patches.
6. After training with early stopping on the validation set, we evaluate final performance on the held-out test
   set and report test accuracy.

Output:
1. Test Accuracy: 0.9638
2. best_model.pth
____________________________________________________________________________________________________________
## Code name: 3_c_vis.py
1. visualize the channels in 2 layers

Output:
1. The encoder layers
![image](https://github.com/user-attachments/assets/799d2bbb-46b1-47cb-b209-70aad1430531)

Figure layer 0

From this figure, some channels are high sensitivity to the defect pixels.

![image](https://github.com/user-attachments/assets/e7a4e565-918d-4ff0-9dc9-46f16eec00b9)

Figure layer 1
____________________________________________________________________________________________________________
## Code name: 4_c_hidden_vis，py
1. We visualize samples in the CNN hidden feature space by extracting the 128-dimensional vector from the penultimate
   layer and projecting it to 2D with PCA.

Output：
1. ![image](https://github.com/user-attachments/assets/6257ddf4-4c45-4673-b882-1621f88bb998)

This figure shows that the expressive power of the latent space of CNN is weaker than the VAE, and the classification
accuracy is also weaker than VAE but CNN is more lightweight.

____________________________________________________________________________________________________________
## Code name: v5_VAE_model.py
1. The VAE training script reads information from the data_cache.pt file, and train with the labeled datasets.
2. model automatically separate the dataset to 'train' (0.7), 'val' (0.15) and 'test' (0.15).
3. Three layers\
   Conv2d 0: (1, 50, 50) - (64, 25, 25)\
   Conv2d 1: (64, 25, 25) - (128, 12, 12)\
   Conv2d 2: (128, 12, 12) - (256, 6, 6)\
   Then flatten 256x6x6 = 9216 to 128 latent dimensions.
5. The classification head (self.classifier) takes μ as input and outputs three logits; these logits are passed
   through a softmax to produce the probabilities for groups 0, 1, and 2, and the index with the highest
   probability is chosen as the final predicted class.

Output: 
1. Epoch 21/50 | Train Acc=0.9954 | Val Acc=0.9802 | Val MSE=0.009520 | Val KL=2.2877
    Top 5 KL dims (dim, kl): [(55, 0.9330708384513855), (5, 0.7913203835487366), (6, 0.009043470025062561), (118, 0.008738156408071518), 
(93, 0.008143114857375622)]
Early stopping triggered at epoch 21.
Test Acc=0.9819 | Test MSE=0.000848

model has better classification performance than CNN.

2. save best weight best_model.pth
____________________________________________________________________________________________________________
## Code name: 6_v_latent_cluster.py
1. input data_cache.py and import the best_model.pth to extract only μ (ignore log variance logσ^2).
2. Use PCA to reduce the dimension of latent space, then visualize the map.

Output:
1. ![image](https://github.com/user-attachments/assets/aa32f54d-b02e-4e4b-aad5-feecb44cc5bf)


![image](https://github.com/user-attachments/assets/365e91e0-b771-4f55-b433-3697cffd0b3a)
____________________________________________________________________________________________________________
## Code name: 7_v_layer_vis.py
1. use hook to visualize a defined layer's channels to a sample patch.
   
Output:
1. Channel image, patch image and thermal-diagram
   ![image](https://github.com/user-attachments/assets/634930ae-662d-48f2-8eba-39d55513644f)
   Figure layer(64, 25, 25)
____________________________________________________________________________________________________________
## Code name: 8_recon_val.py
1. Test the reconstruction performance of VAE model

Output:
1. ![image](https://github.com/user-attachments/assets/b3955bc9-df45-42df-8bc8-48c9f1b2a414)
  Figure This is reconstruction demonstration
Average MSE: 0.001099\
Average PSNR: 31.71 dB\
Average SSIM: 0.7979

____________________________________________________________________________________________________________





