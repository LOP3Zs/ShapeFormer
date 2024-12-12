export CUDA_VISIBLE_DEVICES=1
export AISTRON_DATASETS=../data/datasets/

python -W ignore train_net.py \
    --config-file configs/KINS2020/shapeformerv2_R50_FPN_kins2020_6ep_bs1.yaml \
    --num-gpus 1 
