DATASET=cifar100
ARCH=resnet18_in
SAVEDIR=./snl_output/cifar100/$ARCH/
MODELDIR=./pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
RELU_BUDGET=45000
FINETUNE_EPOCH=100
EPOCHS=2000
MODELDIR=./pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
LOGNAME=resnet18_in_unstructured_$RELU_BUDGET.txt
SAVEDIR=./snl_output/cifar100/$RELU_BUDGET/$ARCH/
LR=1e-3
THRESHOLD=1e-5
ALPHA=1e-5
BATCH=128

CUDA_VISIBLE_DEVICES=0,1 python3  snl_finetune_unstructured_HyperParam.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH 

