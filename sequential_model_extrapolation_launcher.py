import os

for seed in [0, 1, 2] :
    for itr in [200, 100, 0]:
        if seed == 0 and itr == 0 :
            pass
        else:
            cmd_str = "parallel -j 64" + \
            " \'CUDA_VISIBLE_DEVICES=4\' python launch_mier.py ./configs/envs/ood/walker-rand-params.json ./configs/exps/mier-extrapolate.json" + \
            " --log_annotation ood/train_steps1000_seed"+str(seed)+"/" + \
            " --load_model_itr "+str(itr)+\
            " --task_id_for_extrapolation={1}" + \
            " --seed={2}"+ \
            " ::: 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59"+  \
            " ::: "+str(seed)
            os.system(cmd_str)

