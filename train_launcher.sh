
export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=48

parallel -j $N_JOBS \
    'CUDA_VISIBLE_DEVICES=$(({%} % $N_GPUS))' python launch_mier.py ./configs/envs/cheetah-negated-joints.json ./configs/exps/mier-train.json \
     --log_annotation default \
     --seed={1} \
     --num_training_steps_per_epoch={2} \
    ::: 0 1 2 \
    ::: 1000 2000
