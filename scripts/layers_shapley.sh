DATASET=cifar10
ARCH=resnet18_in
SAVEDIR=./snl_output/layer_shapley
MODELDIR=./pretrained_models/cifar10/resnet18_in/best_checkpoint.pth.tar
RELU_BUDGET=45000
FINETUNE_EPOCH=100
EPOCHS=2000
MODELDIR=./pretrained_models/cifar10/resnet18_in/best_checkpoint.pth.tar
LOGNAME=resnet18_in_shapley.txt
SAVEDIR=./snl_output/layer_shapley
LR=1e-3
THRESHOLD=1e-5
ALPHA=1e-5
BATCH=128

CUDA_VISIBLE_DEVICES=2 python3  layer_shapley.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --batch $BATCH --logname "$LOGNAME" 

