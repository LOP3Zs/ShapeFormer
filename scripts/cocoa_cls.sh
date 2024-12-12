export CUDA_VISIBLE_DEVICES=2
export AISTRON_DATASETS=../data/datasets/

python -W ignore train_net.py \
    --config-file configs/COCOA-cls/shapeformerv2_R50_FPN_cocoa_cls_8ep_bs2.yaml \
    --num-gpus 1 
