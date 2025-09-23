export CUDA_VISIBLE_DEVICES=1
export AISTRON_DATASETS=../train/cocoa_format_annotations_aistron.json

python -W ignore train_net.py \
    --config-file configs/COCOA-cls/shapeformerv2_R50_FPN_cocoa_cls_8ep_bs2.yaml \
    --num-gpus 1 
