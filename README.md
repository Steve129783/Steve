Supervised Learning models: 1. CNN, 2. CVAE

What we input:
1. Three datasets, 0: "correct", 1: "high", 2: "low"
These datasets includs 50x50 patches from components with different print energy.
![image](https://github.com/user-attachments/assets/25ca3d79-7ec1-415a-9f78-1aaa46326e8f)


Code name: 0_image_save.py
1. Padding the side length to multiples of 50
2. split them to 50x50 patches and separate them to 3 files.

Output:
![image](https://github.com/user-attachments/assets/1e226c26-c500-4e14-b5e3-b42d42c8a1a1)

Code name: 1_CNN_model.py
1. (optional) separate the images in files to 
2. split them to 50x50 patches and separate them to 3 files.

Output:
![image](https://github.com/user-attachments/assets/1e226c26-c500-4e14-b5e3-b42d42c8a1a1)

Code name: 2_c_vis.py
1. Show the channels in CNN encoder

Output:
![image](https://github.com/user-attachments/assets/77d39d14-a808-4d31-aa12-b115ae0360c9)
![image](https://github.com/user-attachments/assets/7a55f4d4-cc47-4269-8d17-a34f9ee4194e)
![image](https://github.com/user-attachments/assets/0cfbdcce-4ae5-4e99-88ea-d11889b1e53f)

Code name: 3_Pack.py
1. Read images and label by the group their belong to.
2. Pack them together for the training

















