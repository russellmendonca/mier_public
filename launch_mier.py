import os
import click
import json
import os.path as osp
import argparse
from misc_utils import deep_update_dict
from configs.default import *
from mier import MIER

#def main(env_config, exp_config, log_annotation, seed, fast_adapt_steps, task_id_for_extrapolation):
def main(args):
    variant = default_mier_config
    for config in [args.exp_config, args.env_config]:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    variant_name = ''
    for name in [var_arg[0] for var_arg in get_variant_args()]:
        var_value = getattr(args, name)       
        if var_value != None:
            if name not in ['seed', 'task_id_for_extrapolation', 'log_annotation']:
                variant_name+= '_'+name+'_'+str(var_value)
            if name in variant:
                variant[name] = getattr(args, name)
            elif name in variant['model_hyperparams']:
                variant['model_hyperparams'][name] = var_value
    run_mode = variant['run_mode']
    if run_mode == "extrapolate":
        if variant_name != '':
            variant_name += '/'
        variant_name += 'task_'+str(variant['task_id_for_extrapolation'])

    variant['log_dir'] = osp.join(os.environ.get('MIER_DATA_PATH'), variant['env_name'], run_mode,
                                  variant['log_annotation'] ,  variant_name, 'seed-' + str(variant['seed']))

    assert run_mode in [ "train", "extrapolate"]
    if run_mode == 'train':
        MIER(variant).train()
    elif run_mode == "extrapolate":
        MIER(variant).extrapolate()

def get_variant_args():
    return [ ('num_training_steps_per_epoch', int), ('log_annotation',), ('fast_adapt_steps', int), ('task_id_for_extrapolation', int), 
             ('cross_task_relabelling_for_testing', bool) , ('seed', int), ('load_model_itr', int) ]

parser = argparse.ArgumentParser()
parser.add_argument("env_config", default=None)
parser.add_argument("exp_config", default=None)

for variant_arg in get_variant_args():
    if len(variant_arg) == 1:
        parser.add_argument("--"+variant_arg[0], default=None)
    else:
        parser.add_argument("--"+variant_arg[0], default=None, type=variant_arg[1])

args = parser.parse_args()
main(args)
