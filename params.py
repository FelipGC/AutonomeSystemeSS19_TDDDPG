# default hyperparameters
MAX_EPISODES = int(1e6)
MAX_STEP = int(1e6)
EPISODE_RANGE_DATA = 100  # length of epsisode interval for data (e.g. avg score,...)
BUFFER_SIZE = int(1e5)  # replay memory size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # soft update
LR_ACTOR = 1e-3  # learning rate actor
LR_CRITIC = 1e-3  # learning rate critic
WEIGHT_DECAY = 0  # weight decay
ACTOR_UNITS = [128, 128, 64]
CRITIC_UNITS = [128, 128, 64]
ADD_OU_NOISE = True  # Ornstein-Uhlenbeck noise
MU = 0.  # noise parameter
THETA = 0.15  # noise parameter
SIGMA = 0.3  # noise parameter
SIGMA_MIN = 0.025  # min noise parameter
SIGMA_DECAY = 0.975  # sigma decay value
POLICY_DELAY = 1  # update global networks  and the policy every n steps
DROP_OUT_PROB = 0.1
TRAINING_MODE = True  # False only for testing
INITIAL_RANDOM_ROLLOUTS = True  # True if we want to start with completely random, actions
PATH_LOGS = 'logs'
PATH_DASH = 'logs_dash'
PATH_FIGURE = 'figures'
PATH_GRAPHS = 'graphs'
PATH_WEIGHTS = 'weights'
TWIN_DELAY = True
USE_SMOOTHED_MIN_MAX_GRAPH = True
RANGE_VALUE_FOR_SMOOTH_GRAPHS = int(MAX_EPISODES / EPISODE_RANGE_DATA)

# Hyperparameters for Visualization, Add Parameters as Strings to the PARAMS_EVAL list
PARAMS_EVAL = ['BATCH_SIZE', 'UNITS', 'BUFFER_SIZE', 'DELAY', 'DROP_OUT_PROB', 'TAU', 'INITIAL_RANDOM_ROLLOUTS']

# display graphs for which statistic methods, e.g. max, min, avg, ect
# AVG, MIN, MAX, MED, CUR
STATISTICS = {'avg': False, 'min': False, 'max': False, 'med': False, 'cur': False}


def get_parameters():
    return globals()


def print_parameters():
    [print(k, ':\t', v) for k, v in globals().copy().items() if
     not k.startswith('_') and k != 'tmp' and k != 'In' and k != 'Out' and not hasattr(v, '__call__')]

# Colors
# Blue #1F76B4
# Orange #FFA500
# Green #2CA02C
# Red #FF0100
