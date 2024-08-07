DATASET=tiny_imagenet
arch=resnet18_in
Path=./pretrained_models/$DATASET/$arch

python3 train_unstructured.py "$DATASET" "$arch" "$Path" --gpu 0 