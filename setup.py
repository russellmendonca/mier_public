import tensorflow as tf
from metalearning_envs import ENVS
from metalearning_envs.wrappers import NormalizedBoxEnv
from softlearning.value_functions import vanilla
from softlearning.value_functions.utils import create_double_value_function
from softlearning.policies.gaussian_policy import FeedforwardGaussianPolicy
from softlearning.policies.uniform_policy import UniformPolicy
from softlearning.misc.utils import set_seed, initialize_tf_variables
from softlearning_sac import SAC

from models.bnn_contextual import BNN as BNN_contextual
from models.fake_env_contextual import FakeEnv as FakeEnvContextual

from softlearning_sampler import ContextConditionedSimpleSampler as SoftlearningSAC_ContextConditionedSimpleSampler
from data import ReplayBuffer,  MultiTaskReplayBuffer
from misc_utils import TensorBoardLogger, parse_network_arch, load_data, get_sep_model_hyperparams, set_random_seed

def setup_sess(self):

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)
    self.sess = tf.keras.backend.get_session()

def setup(self):

    set_random_seed(self.seed)
    self.env_params['n_tasks'] = self.n_train_tasks + self.n_val_tasks
    self.env = NormalizedBoxEnv(ENVS[self.env_name](**self.env_params))

    #set_gpu_mode(self.device == 'cuda')
    self.obs_dim = obs_dim = int(self.env.observation_space.shape[0])
    self.act_dim = act_dim = int(self.env.action_space.shape[0])
    context_dim = self.context_dim

    setup_sess(self)
    self.logger = TensorBoardLogger(self.log_dir)

    ############## end to end load path #########################
    if self.load_path_prefix !=None and \
        ((self.run_mode == 'train' and self.continue_training_from_loaded_model) or self.run_mode == 'extrapolate'):
        load_path = self.load_path_prefix + "seed-"+str(self.seed) + "/Itr_"+str(self.load_model_itr)+"/"

        if self.joint_state_reward_model:
            self.model_load_path = load_path +'model.pkl'
        else:
            self.state_model_load_path = load_path + 'state_model.pkl'
            self.reward_model_load_path = load_path + 'reward_model.pkl'

        self.sac_load_path = load_path+'sac.pkl'

        self.replay_buffer_load_path = load_path + 'replay_buffer.pkl'
        #self.pre_adapt_replay_buffer_load_path = load_path + 'pre_adapt_replay_buffer.pkl'

        if self.run_mode == "extrapolate":
            self.cross_task_data_load_path = load_path + 'replay_buffer.pkl'

    ############################## Setup Model #############################

    if self.joint_state_reward_model:

        self.model = BNN_contextual(self.sess, obs_dim, act_dim, self.context_dim,
                                        model_hyperparams=self.model_hyperparams)
        if self.model_load_path:
            self.model.load_model(self.model_load_path)
        self.fake_env = FakeEnvContextual({'model': self.model}, self.env.termination_fn)

    else:
        assert self.meta_learn_state_dynamics == False
        state_model_hyperparams, reward_model_hyperparams = get_sep_model_hyperparams(
                                    self.model_hyperparams, self.meta_learn_state_dynamics, self.meta_learn_reward)


        self.state_model = BNN_contextual(self.sess, obs_dim, act_dim, self.context_dim,
                                   model_hyperparams= state_model_hyperparams)

        self.model = self.reward_model = BNN_contextual(self.sess, obs_dim, act_dim, self.context_dim,
                                    model_hyperparams= reward_model_hyperparams)

        if self.state_model_load_path:
            self.state_model.load_model(self.state_model_load_path)

        if self.reward_model_load_path:
            self.reward_model.load_model(self.reward_model_load_path)

        self.fake_env = FakeEnvContextual({'state_model': self.state_model, 'reward_model': self.reward_model},
                               self.env.termination_fn, joint_state_reward_model=False)

     ############### Setup SAC #############################

    with tf.variable_scope("softlearning"):
        Qs = create_double_value_function(
            vanilla.create_feedforward_Q_function,
            observation_shape=(obs_dim + context_dim,),
            action_shape=(act_dim,),
            hidden_layer_sizes=parse_network_arch(self.critic_nn_arch)
        )

        policy = FeedforwardGaussianPolicy(
            input_shapes=((obs_dim + context_dim,),),
            output_shape=(act_dim,),
            hidden_layer_sizes=parse_network_arch(self.actor_nn_arch),
            squash=True
        )

        initial_exploration_policy = UniformPolicy(
            input_shapes=((obs_dim + context_dim,),),
            output_shape=(act_dim,))

        self.sac_trainer = SAC(
            observation_shape=(obs_dim + context_dim,),
            action_shape=(act_dim,),
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            session=self.sess,
            discount=self.sac_hyperparams["discount_factor"],
            tau=self.sac_hyperparams["target_update_rate"],
            reward_scale=self.sac_hyperparams["sac_reward_scale"],
            reparameterize=True,
            lr=self.sac_hyperparams["actor_learning_rate"],
            target_update_interval=self.sac_hyperparams["target_update_interval"],
            target_entropy=self.sac_hyperparams["target_entropy"]
        )

        initialize_tf_variables(self.sess, only_uninitialized=True)
        if self.sac_load_path:

            self.sac_trainer.load_model(self.sac_load_path)

        self.sampler = SoftlearningSAC_ContextConditionedSimpleSampler(
            env=self.env,
            max_path_length=self.max_path_length,
            policy=self.sac_trainer._policy,
            exploration_policy=initial_exploration_policy
        )

    ######################## DataSet, Buffer Setup #####################################################
    if self.run_mode == "train":

        self.replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_max_sample_size, self.n_train_tasks)
        self.pre_adapt_replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_max_sample_size, self.n_train_tasks)
        self.model_buffer = MultiTaskReplayBuffer(self.replay_buffer_max_sample_size, self.n_train_tasks)

        if self.replay_buffer_load_path != None:
            self.replay_buffer.load_data(self.replay_buffer_load_path)

        if self.pre_adapt_replay_buffer_load_path != None:
            self.pre_adapt_replay_buffer.load_data(self.pre_adapt_replay_buffer_load_path)

    elif self.run_mode == "extrapolate":

        if self.cross_task_data_load_path != None:
            self.cross_task_data, self.cross_task_data_size = load_data(self.cross_task_data_load_path, True)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_max_sample_size)
        self.model_buffer = ReplayBuffer(self.replay_buffer_max_sample_size)
