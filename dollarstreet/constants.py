# train/val params
BATCH_SIZE = 256
NUM_WORKERS = 12
SHUFFLE = True
PIN_MEMORY = True
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
VALID_MODELS = ["resnet", "squeezenet", "densenet"]

# dataset params
PATH_COL = 'imageRelPath'
TARGET_COL = 'imagenet_sysnet_id'
