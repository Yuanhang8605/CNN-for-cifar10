CONFIG = {
    # Network Parameters: 
    'USE_BOTTLENECK':False,
    'NUM_RESIDUAL_UNITS':5,
    'RELU_LEAKINESS':0.1,
    'WEIGHT_DECAY_RATE':0.0002,

    # Solver Parameters:
    'TRAIN_BATCH_SIZE':128,
    'EVAL_BATCH_SIZE':100,
    'NUM_CLASSES':10,
    #'MIN_LRN_RATE':0.0001,
    'LRN_RATE':0.1,
    'OPTIMIZER':'mom',
    'MAX_ITER': 100000,

    # results saving path
    'CHECKPOINT_DIR':'/home/yuan/CNN_V4/logs/resnet/train',
    'LOG_DIR':'/home/yuan/CNN_V4/logs/resnet',  
    'EVAL_DIR':'/home/yuan/CNN_V4/logs/resnet/eval'
}
