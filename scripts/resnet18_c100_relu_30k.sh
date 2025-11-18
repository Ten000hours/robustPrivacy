DATASET=cifar100
ARCH=resnet18_in
RELU_BUDGET=400000
FINETUNE_EPOCH=100
EPOCHS=600
MODELDIR=./pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
LOGNAME=resnet18_in_unstructured_$RELU_BUDGET.txt
SAVEDIR=./output/cifar100/$RELU_BUDGET/$ARCH/
LR=1e-3
THRESHOLD=1e-5
ALPHA=1e-5
BATCH=128
TYPE=privshap

CUDA_VISIBLE_DEVICES=2 python3 privshap_unstructured.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH --type $TYPE
