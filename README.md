# MIER
Model Identification and Experience Relabelling

## Installation
1. Run ./setup.sh
2. Add the following to `~/.bash_aliases`, and then `source ~/.bash_aliases`

   `alias mier='source activate mier ; export PYTHONPATH=<path to mier>; export MIER_DATA_PATH=<path to desired output directory>'`
    After this, typing mier in any new terminal will set up the environment.
3. [Mujoco](https://www.roboti.us/index.html) installation
    - install mujoco in `~/.mujoco` (we use mujoco 1.5)
    - add mujoco location and nvidia driver location to the `LD_LIBRARY_PATH`
    

## Launchers
1. Meta-training : 

      `python launch_mier.py ./configs/envs/<env_name> ./configs/exps/train/<exp_name> --log_annotation <experiment name> --seed <seed>`
   
2. Extrapolation : 
      
       `python launch_mier.py ./configs/envs/<env_name> ./configs/exps/test/<exp_name> --log_annotation <experiment name> --load_model_itr <model iteration> --task_id_for_extrapolation <id> --seed <seed>`

exp_names for environments with only variable reward functions are `mier-meta-train-only-rew.json` (training) and `mier-extrapolate-sep-models.json` (extrapolation). For environments with variable dynamics, use `mier-train.json` (training )and `mier-extrapolate.json` (extrapolation). See `.train_launcher.sh` and `extrapolation_launcher.sh` for examples of how to launch experiments with gnu-parallel. The environment configuration file overrides the experiment configuration file. When running extrapolation, add value to load_path_prefix in the environment config file (see example in cheetah-negated-joints config file).

       
       
       
   


