DATASET=cifar10
ARCH=resnet18_in
SAVEDIR=./snl_output/cifar10/$ARCH/
MODELDIR=./pretrained_models/cifar10/resnet18_in/best_checkpoint.pth.tar
RELU_BUDGET=150000
FINETUNE_EPOCH=100
EPOCHS=300
MODELDIR=./pretrained_models/cifar10/resnet18_in/best_checkpoint.pth.tar
LOGNAME=resnet18_in_unstructured_$RELU_BUDGET.txt
SAVEDIR=./snl_output/cifar10/$RELU_BUDGET/$ARCH/
LR=1e-3
THRESHOLD=1e-5
ALPHA=1e-5
BATCH=128

CUDA_VISIBLE_DEVICES=1 python3  snl_finetune_unstructured_projected.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH 
