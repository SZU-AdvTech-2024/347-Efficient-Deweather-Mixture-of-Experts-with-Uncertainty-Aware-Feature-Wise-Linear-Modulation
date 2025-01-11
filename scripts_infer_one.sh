#################################### Allweather #######################################################
model_path=output/train/allweather_moe-film-linear-basenet-star-gelu-n8-k4_bs16_ep1_ps8_embed384_mlpx4_mlpupsample-outchx4_cnn-embed_wo-pe_normalize_vgg0.04_lr0.0002/best_metric.pth
output_dir=output/infer_one
img_path=data/RVSD/test/snow/00003/0015.jpg
dataset=RVSD
task=desnow
cuda='1'

mkdir -p ${output_dir}

export CUDA_VISIBLE_DEVICES=${cuda}

python infer_one.py --task ${task} --dataset ${dataset} \
--img-path $img_path \
--model-path $model_path --model-name mowe \
--gpu-list 1 \
--exp $output_dir \
>> ${output_dir}/exp.txt 2>&1