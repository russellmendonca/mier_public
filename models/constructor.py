import numpy as np
import tensorflow as tf

from models.fc import FC
from models.bnn import BNN
from parameterized_model import ParameterizedModel
from misc_utils import AttrDict


def construct_model(sess, obs_dim, act_dim, model_hyperparams):
    # context_dim = 5 ,  rew_dim=1, hidden_dim=200,
    # ada_state_dynamics_pred = True, ada_rew_pred = True,
    # fast_adapt_steps = 2 , fast_adapt_lr = 0.01,
    # reg_weight = 1, pred_dynamics = True, fixed_preupdate_context = True,  num_networks=1, num_elites=1):
    # output_dim = rew_dim + obs_dim
    model = BNN(sess, obs_dim, act_dim, model_hyperparams)
    # ada_state_dynamics_pred, ada_rew_pred,
    # fast_adapt_steps , fast_adapt_lr, reg_weight , fixed_preupdate_context )

    model.add(FC(model.hidden_dim, input_dim=obs_dim + act_dim + model.context_dim, activation="swish",
                 weight_decay=0.000025))
    model.add(FC(model.hidden_dim, activation="swish", weight_decay=0.00005))
    model.add(FC(model.hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(model.hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(model.output_dim, weight_decay=0.0001))

    # model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    model.build_graph()
    return model


#
# def format_samples_for_training(samples):
#    obs = samples['observations']
#    act = samples['actions']
#    next_obs = samples['next_observations']
#    rew = samples['rewards']
#    delta_obs = next_obs - obs
#    inputs = np.concatenate((obs, act), axis=-1)
#    outputs = np.concatenate((rew, delta_obs), axis=-1)
#    return inputs, outputs
#
#
# def reset_model(model):
#    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
#    model.sess.run(tf.initialize_vars(model_vars))
#

if __name__ == '__main__':
    model = construct_model()
