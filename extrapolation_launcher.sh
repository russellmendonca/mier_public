export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=64

parallel -j $N_JOBS \
    'CUDA_VISIBLE_DEVICES=$(({%} % $N_GPUS))' python launch_mier.py ./configs/envs/ood/cheetah-negated-joints.json ./configs/exps/mier-extrapolate.json \
     --log_annotation "train_steps1000/itr_400" \
     --load_model_itr 400 \
     --task_id_for_extrapolation={1} \
     --seed={2} \
    ::: 10 11 12 13 14 15 16 17 18 19  \
    ::: 0
