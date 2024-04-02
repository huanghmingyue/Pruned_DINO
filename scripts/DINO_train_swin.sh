coco_path=$1
backbone_dir=$2
export CUDA_VISIBLE_DEVICES=$3 && python main.py \
	--output_dir ~/autodl-tmp/logs/DOTAv2/swin-4scale-2 -c config/DINO/DINO_4scale_swin.py --resume ~/autodl-tmp/logs/DOTAv2/swin-4scale-1/checkpoint0015.pth --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir

