# Spine segmentation (T2 MRI)

#### Copy only images and labels from local to remote desktop with this command
> scp -r "C:/Users/ex/Spine/Data/spine/*" tejasm@ex:/home/tejasm/Data/spine/

#### Copy code from local to remote desktop with this command
> scp -r "C:/Users/ex/Spine/Code/spine_linux/*" tejasm@ex:/home/tejasm/Code/spine/spine_linux/

#### Run training with this command (on GPU ID 0)
> CUDA_VISIBLE_DEVICES=0 python runScript_Spine_Seg_Train.py /home/tejasm/Code/spine/spine_linux/experiments/u_spine_UNet_unet_reg_v1_shuffled_noDropout.json