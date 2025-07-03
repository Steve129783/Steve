Supervised Learning models: 1. CNN, 2. CVAE

What we input:
1. Three datasets, 0: "correct", 1: "high", 2: "low"
These datasets includs 50x50 patches from components with different print energy.
![image](https://github.com/user-attachments/assets/25ca3d79-7ec1-415a-9f78-1aaa46326e8f)

Current objectives:
1. Check CNN and CVAE latent space 
2. Optimise the hyperparameters
____________________________________________________________________________________________________________
Code name: 0_image_save.py
1. Read .tif files and separate them to different folders
2. output padded full images (200x200) and patches (50x50).

Output: 
1. ![352_centered_200x200](https://github.com/user-attachments/assets/37ae058c-41c8-4413-9809-4c5417b8c795)

2. ![000_x50_y50](https://github.com/user-attachments/assets/faa78a56-f36e-4a4d-8ba9-bdbf1f29ee84)

____________________________________________________________________________________________________________
Code name: 1_Pack.py
1. Read patches from folders and label them by their folders’ name
2. Pack them together to a .pt file 

Output:
1. E:\Project_CNN\2_Pack\data_cache.pt
![image](https://github.com/user-attachments/assets/f2712475-3c81-4388-91e9-9292b0b3590e)


____________________________________________________________________________________________________________
Code name: 2_CNN_model.py
1. The training script reads group_ids from the cached .pt file, counts the unique values, and automatically
   sets n_classes accordingly (in your case, 3 classes).
3. We split the dataset with ratios 70% train, 15% validation, and 15% test to help prevent overfitting and
   ensure a reliable estimate of generalization.
4. CNN use two layers, Conv Block 1:(1,50,50)-(16,25,25)，
                       Conv Block 2:(16,25,25)-(32,12,12),
   then flatten the 32x12x12 = 4608 to 128 dimensions (hidden fully-connected layer similar to latent space of
   VAE but does not have the ability to generate and interpolate) then classify the group of patches.
5. After training with early stopping on the validation set, we evaluate final performance on the held-out test
   set and report test accuracy.

Output:
1. Test Accuracy: 0.9654
2. best_model.pth
____________________________________________________________________________________________________________
Code name: 3_c_vis.py
1. visualize the channels in 2 layers

Output:
1. The encoder layers
![image](https://github.com/user-attachments/assets/36eeeae2-7d23-4f3a-b2e3-5ad774514432)
Figure layer 0
![image](https://github.com/user-attachments/assets/725a93a9-aa8f-499e-9d27-2bbf3d0a1b78)
Figure layer 1















