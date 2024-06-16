DATASET=cifar100
ARCH=resnet9_in
SAVEDIR=./snl_output/layer_shapley
MODELDIR=./pretrained_models/cifar100/resnet9_in/best_checkpoint.pth.tar
RELU_BUDGET=45000
FINETUNE_EPOCH=100
EPOCHS=2000
MODELDIR=./pretrained_models/cifar100/resnet9_in/best_checkpoint.pth.tar
LOGNAME=resnet9_in_shapley.txt
SAVEDIR=./snl_output/layer_shapley
LR=1e-3
THRESHOLD=1e-5
ALPHA=1e-5
BATCH=128

python3  layer_shapley.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --batch $BATCH --logname "$LOGNAME" &

