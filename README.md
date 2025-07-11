## Whole process
1. Use Semantic Segmentation to align the main body in every dataset from Guan, then cut them to 50x50 patches
2. Use CNN to cluster them (4 component per row or per column)
3. Use VAE + classify head to cluster them (4 component per row or per column)
4. Contrast the result and combine them together to prove the effectiveness of this method to assistant people
   tune the printer's parameters.


### Optimized points：
1. VAE training program used wrong 'kl' formula.
2. Pack program remove the 'padding' method to reduce the unrelated information
3. Add CNN channel weight to research the relationship between layers
4. The hidden space visualization program is repaired

### Prediction to the final program
The patches from different components can be clustered in the hidden space, operator can check the location of
the target cluster to know whether the current parameter (print temp, print speed, angle et al.) is appropriate 
compare with his predict.
\
## Please check 'c5_channel_weight.py' part.

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
## Code name: c5_channel_weight.py
1. This code gives higher defect sensitivity channels larger weight and gives others a short weight to train
   the mdoel.

Output predict: the channels in layer 2 will store more defect informations and the classifier would like to 
use defect features to classify the patches.

Output 1：
1. I optimized the code that we discussed in meeting, it shows a better defect sensitivity.
   Weight：0\
   Epoch 13  Val Loss=0.4907  Val Acc=0.7996\
Early stopping\
Test Acc: 0.8378\
This shows because of the information decreasing, the classifier cannot contain the high classification accuracy.

3. Compare with the c2 (full channels), the new hidden space c11 shows that the defects linearly growth along the
   x-axis with the greater independence.
   ![image](https://github.com/user-attachments/assets/823eab5d-4cd0-46eb-8119-d4f8deb4f863)
This figure shows the boundary of clusters are not as clear as c2 but better in interpretability.

4. This image shows the change in layer 0 and 1.
   ![image](https://github.com/user-attachments/assets/e31c9c40-c16f-4374-8cb0-9d879857c99f)
Layer 0 only remain 2 channels

   ![image](https://github.com/user-attachments/assets/de25ead3-77c4-48d1-9d80-987b425f873c)

Output 2:
1. Weight：0.1\
   Epoch 14  Val Loss=0.1407  Val Acc=0.9433\
Early stopping\
Test Acc: 0.9513\
Better accuracy
  ![image](https://github.com/user-attachments/assets/8d3c51cb-9de5-4419-acb7-dd2948d011aa)
  ![image](https://github.com/user-attachments/assets/f5ff5f32-7bc1-40d5-9896-91cd80ec29e0)
  ![image](https://github.com/user-attachments/assets/124cb62b-5e00-411d-821d-3b4c17a778c1)

## Conclusion:
Through contrast the hidden space from two weight (0 and 0.1), this 'post-process' method specifically designed 
to block useless information.
____________________________________________________________________________________________________________
____________________________________________________________________________________________________________
## Code name: v1_VAE_model.py
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
## Code name: 2_v_latent_cluster.py
1. input data_cache.py and import the best_model.pth to extract only μ (ignore log variance logσ^2).
2. Use PCA to reduce the dimension of latent space, then visualize the map.

Output:
1. ![image](https://github.com/user-attachments/assets/6bbee6d8-a835-4267-9c87-7e87f4c19349)
Analysation: The latent space map shows that these three clusters, the 'correct' patch is separated by 'high'
and 'low', the patch which is closer to the centre of the 'low' cluster, likely to has more defects or more
roughness texture, and closer to 'high' centre will be more smooth.
____________________________________________________________________________________________________________
## Code name: 3_v_layer_vis.py
1. use hook to visualize a defined layer's channels to a sample patch.
   
Output:
1. Channel image, patch image and thermal-diagram
   ![image](https://github.com/user-attachments/assets/19c7547d-e224-47a7-8c93-74a5024655e8)
   Figure layer 1 （64, 25, 25)\
   <img width="1978" height="1240" alt="image" src="https://github.com/user-attachments/assets/4714e7e2-82f1-4e6f-9601-b66e1816c06f" />
   Figure layer 2 （64, 25, 25)
____________________________________________________________________________________________________________
## Code name: 4_recon_val.py
1. Test the reconstruction performance of VAE model

Output:
1. ![image](https://github.com/user-attachments/assets/530117e9-ed4f-42c8-a70f-7828be51f020)
  Figure This is reconstruction demonstration
Average MSE: 0.000025\
Average PSNR: 46.37 dB\
Average SSIM: 0.9881
____________________________________________________________________________________________________________
## Code name: 5_VAE_channel.py
1. Use the same method to limit the model extracting the defect features in layer 1 while reset the layer 2 and 3
to retrain them.

Output:
1. Epoch 26/100  Train Acc=0.9948 | Val Acc=0.9329  Val MSE=0.000184  Val KL=3.4598\
Early stopping\
Test Acc=0.9460  Test MSE=0.000184

2. <img width="2439" height="1233" alt="image" src="https://github.com/user-attachments/assets/7a70c4c1-8502-480d-96ba-c7d0ab805357" />
   <img width="2430" height="1211" alt="image" src="https://github.com/user-attachments/assets/e9019ed6-29b2-44c5-98c9-cbe2d3301cd2" />
   


3. <img width="1553" height="1205" alt="image" src="https://github.com/user-attachments/assets/47df365a-5f59-47d0-b7f9-ef13355e78c3" />
   <img width="1828" height="1206" alt="image" src="https://github.com/user-attachments/assets/2e70fcc2-a6c5-4197-8662-ca3c876c305a" />

____________________________________________________________________________________________________________
## Next step:
1. Pack the patches from other components to see the cluster ability of the model.

## My question:







