### Optimized points：
1. VAE training program used wrong 'kl' formula.
2. Pack program remove the 'padding' method to reduce the unrelated information
3. Add CNN channel weight to research the relationship between layers

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
![000_x0_y50](https://github.com/user-attachments/assets/1f4af70b-a253-415a-961e-c6e0c3681fcf)

### Current objectives:
1. Research the relationship between layers
2. check CNN hidden space and VAE latent space

### Tasks
1. Use weight difference to regulate the information that the CNN model input to hidden space.

### Mistake
1. In the meeting 3rd Jul, I said my model is 'CVAE', that is wrong. That model is a ‘joint VAE–classifier model’.
   The VAE component is trained in an unsupervised manner on the unlabeled data, whereas the classification head
   is supervised with labeled examples to enforce class-specific structure in the latent space.
2. 'kl' formula in VAE training program is wrong.
____________________________________________________________________________________________________________
## Code name: 0_image_save.py
1. Read .tif files and separate them to different folders
2. output padded full images (original size) and patches (50x50).

Output: 
1. ![000_cropped_150x150](https://github.com/user-attachments/assets/049e3790-50f6-4abd-85e3-396bfdd675c6)

2. ![000_x0_y50](https://github.com/user-attachments/assets/1f4af70b-a253-415a-961e-c6e0c3681fcf)
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
1. Epoch 14  Val Loss=0.4581  Val Acc=0.8423
Early stopping
Test Accuracy: 0.9485
2. best_model.pth
____________________________________________________________________________________________________________
## Code name: 3_c_vis.py
1. visualize the channels in 2 layers

Output:
1. The encoder layers
![image](https://github.com/user-attachments/assets/3e7a18b2-e1d0-40cc-9c9d-c5a11015dee7)
Figure layer 0

From Figure 1, channels [11, 13] are high sensitivity to the defect pixels.

![image](https://github.com/user-attachments/assets/c3c5ad07-c95c-41f9-926b-0aecdfa8f1cf)
Figure layer 1

From Figrue 2, channels [11, 13] are high sensitivity to the defect pixels.
____________________________________________________________________________________________________________
## Code name: 4_c_hidden_vis，py
1. We visualize samples in the CNN hidden feature space by extracting the 128-dimensional vector from the penultimate
   layer and projecting it to 2D with PCA.

Output：
1. ![image](https://github.com/user-attachments/assets/c1fb4590-4c2a-4e9e-b31f-daf0f2c3972c)
2. ![image](https://github.com/user-attachments/assets/4f12a95f-c531-4535-83d9-52b622a1bebf)

This figure shows the CNN model can classify the patches to different clusters in a good accuracy and PCA dimensional-reduction
make the hidden space changing linearly, y-axis shows the size of defects and x-axis shows the texture and number of components.
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
1. Epoch 18/100 | Train Acc=0.9180 | Val Acc=0.8119 | Val MSE=0.000068 | Val KL=0.0303
No improvement for 2 epoch(s)
Early stopping at epoch 18
Test Acc=0.9639 | Test MSE=0.000021

model has better classification performance than CNN.
____________________________________________________________________________________________________________
## Code name: 6_v_latent_cluster.py
1. input data_cache.py and import the best_model.pth to extract only μ (ignore log variance logσ^2).
2. Use PCA to reduce the dimension of latent space, then visualize the map.

Output:
1. ![image](https://github.com/user-attachments/assets/6bbee6d8-a835-4267-9c87-7e87f4c19349)
Analysation: The latent space map shows that these three clusters, the 'correct' patch is separated by 'high'
and 'low', the patch which is closer to the centre of the 'low' cluster, likely to has more defects or more
roughness texture, and closer to 'high' centre will be more smooth.
____________________________________________________________________________________________________________
## Code name: 7_v_layer_vis.py
1. use hook to visualize a defined layer's channels to a sample patch.
   
Output:
1. Channel image, patch image and thermal-diagram
   ![image](https://github.com/user-attachments/assets/19c7547d-e224-47a7-8c93-74a5024655e8)
   Figure layer(64, 25, 25)
____________________________________________________________________________________________________________
## Code name: 8_recon_val.py
1. Test the reconstruction performance of VAE model

Output:
1. ![image](https://github.com/user-attachments/assets/530117e9-ed4f-42c8-a70f-7828be51f020)
  Figure This is reconstruction demonstration
Average MSE: 0.000025\
Average PSNR: 46.37 dB\
Average SSIM: 0.9881

____________________________________________________________________________________________________________





