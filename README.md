# nncls

A simple toolbox for image classification task on top of [torchvision](https://github.com/pytorch/vision)

# ⭐ Models

## ResNet
train:
`
torchrun --nproc_per_node=8 train.py --model resnet50 --batch-size 128 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232
`

eval:
`
torchrun --nproc_per_node=8 train.py --model resnet50 --test-only --weights ImageNet1K_V2
`
