# Spine segmentation (T2 MRI)

### Verifying segmentation masks and images were correctly aligned in the data loader
The functions in the ``common\spineDataLoader.py`` script ensure that the images/labels match each other. Only masks containing all the labels (background 0, disc 1, vertebrae 2) were used, and the remaining masks (and corresponding images) were not used. Only the useful images were taken in the data loader, and the unsued images were pruned in the data loader. The ``common\utilities\iterators\batchiterators_spine.py`` script also takes care of matching the images and the labels based on the filename. You can also explore the data in the script ``exploration\runScript_spine_readData.py`` for more information. I had also visually inspected the images to see if they matched the underlying image. 

### Data assumptions
Only masks containing all the labels (background 0, disc 1, vertebrae 2) were used, and the remaining masks (and corresponding images) were not used. You can explore the data in the script ``exploration\runScript_spine_readData.py`` for more information. You can edit the ``experiments/u_spine_UNet_unet_reg_v1_shuffled_noDropout.json`` file to make any changes to the data loading scheme by changing the location to the data/masks folder. The code is modular enough to withstand slight modifications. The data seems to be acquired from different scanners, and perhaps a different normalization/harmonization scheme may be necessary. As a first step, I just used a simple division by 255. 

### Model assumptions 
U-Net model in tensorflow was used, and the network was trained for 30 epochs with the Adam optimizer, learning rate of 1e-5, batch size of 2, and the dice loss (shown to be good for class imbalance). Dropout can also be used (check the JSON file in ``experiments/u_spine_UNet_unet_reg_v1_shuffled_noDropout.json`` for changes and modifications. I would simply experiment with this basic model as it has been shown to be useful for many segmentation tasks. More complicated models, or even pre-trained models can be further explored after. 

### Instructions to run code

#### Copy only images and labels from local to remote desktop with this command
> scp -r "C:/Users/ex/Spine/Data/spine/*" tejasm@ex:/home/tejasm/Data/spine/

#### Copy code from local to remote desktop with this command
> scp -r "C:/Users/ex/Spine/Code/spine_linux/*" tejasm@ex:/home/tejasm/Code/spine/spine_linux/

#### Run training with this command (on GPU ID 0)
> CUDA_VISIBLE_DEVICES=0 python runScript_Spine_Seg_Train.py /home/tejasm/Code/spine/spine_linux/experiments/u_spine_UNet_unet_reg_v1_shuffled_noDropout.json