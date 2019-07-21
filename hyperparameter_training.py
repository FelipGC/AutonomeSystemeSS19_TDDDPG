import sys
import params
import run
from utils import Logger

NUM_OF_TRAINING_PHASES = 3


PARAMETERS_TO_CHANGE = {
    'POLICY_DELAY': [1, 2, 4, 8, 16],
    'TAU': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    'BUFFER_SIZE': [8192, 262144, 1048576],
    'UNITS': [64, 128],
    'BATCH_SIZE': [32, 64],
    'INITIAL_RANDOM_ROLLOUTS': [True, False],
    'DROP_OUT_PROB': [0, 0.1, 0.2],
}

index = 0
max_count = NUM_OF_TRAINING_PHASES * sum([len(x) for x in PARAMETERS_TO_CHANGE.values()])

for _ in range(2):
    twin = "Twin" if params.get_parameters()['TWIN_DELAY'] else "Single"
    for param, values in PARAMETERS_TO_CHANGE.items():
        if str(param) == 'UNITS':
            init_value = params.get_parameters()['ACTOR_UNITS']
            init_value_2 = params.get_parameters()['CRITIC_UNITS']
            for value in values:
                for x in range(NUM_OF_TRAINING_PHASES):
                    unit_array = [value, value]
                    params.get_parameters()['ACTOR_UNITS'] = unit_array
                    params.get_parameters()['CRITIC_UNITS'] = unit_array
                    index += 1
                    print('<< Starting training with: {} = {}\t({}/{})>>'.format(param, value, index, max_count))
                    run.main(label="{}-{}#{}-{}".format(x, param, value, twin))
                    sys.stdout = Logger.terminal
            params.get_parameters()['ACTOR_UNITS'] = init_value
            params.get_parameters()['CRITIC_UNITS'] = init_value_2
        else:
            init_value = params.get_parameters()[param]
            # train with distinct value
            for value in values:
                for x in range(NUM_OF_TRAINING_PHASES):
                    params.get_parameters()[param] = value
                    index += 1
                    print('<< Starting training with: {} = {}\t({}/{})>>'.format(param, value, index, max_count))
                    run.main(label="{}-{}#{}-{}".format(x, param, value, twin))
                    sys.stdout = Logger.terminal
            # restore initial value
            params.get_parameters()[param] = init_value
    params.get_parameters()['TWIN_DELAY'] = not params.get_parameters()['TWIN_DELAY']
