ps=8
bs=16
ep=1 # warmup=3
lr=0.0002
scheduler='cosine+warmup'
task='desnow'   # optional: derain, desnow, deraindrop(allweather), dehaze(maw_sim, maw_real)
dataset='RVSD'
model='mowe'       # optional: transweather
dim=384
interval=1

# ############################## baseline #######################################
# export CUDA_VISIBLE_DEVICES='4,5,6,7'
# OUTPUT_DIR=output/train
# EXPERIMENT=allweather_bs${bs}_ep100_ps${ps}_embed${dim}_mlpx4_mlpupsample-outchx4_cnn-embed_wo-pe_normalize_vgg0.04_lr${lr}
# mkdir -p "$OUTPUT_DIR"/"$EXPERIMENT"

# torchrun --master_port 29501 --nproc_per_node=4 train.py \
# --task $task --dataset $dataset --augment-enable \
# --model $model --disable-residual \
# --img-patch $ps --embed-dim $dim --layer 2 \
# --loss-list content l1 1 perception vgg16 0.04 \
# --epoch $ep --global-bs $bs --optimizer adamw --lr $lr --scheduler $scheduler \
# --gpu-list 4 5 6 7 \
# --test-interval ${interval} \
# --exp ${EXPERIMENT} \
# 2>&1 |tee -a ${OUTPUT_DIR}/${EXPERIMENT}/exp.txt

# ############################## mofme #######################################

num_experts=(8)
topk=(4)

for i in "${num_experts[@]}"  
do
    for j in "${topk[@]}"
    do
    export CUDA_VISIBLE_DEVICES='0,1'
    OUTPUT_DIR=output/train
    EXPERIMENT=allweather_moe-film-linear-basenet-star-gelu-n${i}-k${j}_bs${bs}_ep${ep}_ps${ps}_embed${dim}_mlpx4_mlpupsample-outchx4_cnn-embed_wo-pe_normalize_vgg0.04_lr${lr}
    mkdir -p "$OUTPUT_DIR"/"$EXPERIMENT"

    torchrun --master_port 29510 --nproc_per_node=2 train.py \
    --task $task --dataset $dataset --augment-enable \
    --model $model --disable-residual \
    --moe-enable --gate film --type-expert ffn --num-expert ${i} --top-k ${j} \
    --img-patch $ps --embed-dim $dim --layer 2 \
    --overlap-crop \
    --loss-list content l1 1 perception vgg16 0.04 \
    --epoch $ep --global-bs $bs --optimizer adamw --lr $lr --scheduler $scheduler \
    --gpu-list 0 1 \
    --test-interval ${interval} \
    --exp ${EXPERIMENT} \
    2>&1 |tee -a ${OUTPUT_DIR}/${EXPERIMENT}/exp.txt
    done
done
