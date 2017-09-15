python ../train/run.py --train_data_path=../data/cifar10/data_batch* \
            --dataset='cifar10' \
            --num_gpus=0    \
            --mode='train'  \
            --lr_setHook=True
