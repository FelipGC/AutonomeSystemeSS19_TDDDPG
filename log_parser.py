from collections import deque, namedtuple, defaultdict
from itertools import zip_longest

import os
import plotly.graph_objs as go
import plotly.offline as py
import params

sum_rewards = 0
sum_rewards_count = 0


def get_plot_traces(p, p_avg, p_min, p_max, p_med, p_current, param_name=None, param_val=None):
    labels = None
    if param_name:
        labels = param_name + ": " + param_val

    trace_avg = go.Scatter(x=p, y=p_avg, mode='lines', name=labels if param_name else 'Avg. score (p. 100 eps)')
    trace_min = go.Scatter(x=p, y=p_min, mode='lines', name=labels if param_name else 'Min. score (p. 100 eps)')
    trace_max = go.Scatter(x=p, y=p_max, mode='lines', name=labels if param_name else 'Max. score (p. 100 eps)')
    trace_med = go.Scatter(x=p, y=p_med, mode='lines', name=labels if param_name else 'Med. score (p. 100 eps)')
    trace_current = go.Scatter(x=p, y=p_current, mode='lines', name=labels if param_name else 'Score')

    data = [trace_avg, trace_min, trace_max, trace_med, trace_current]
    return data




def get_hyperparams(lines):
    batch_size, gamma, tau, lr_actor, lr_critic, weight_decay, actor_units, critic_units, add_ou_noise, mu, theta, \
    sigma, sigma_min, sigma_decay, policy_decay, drop_out_prob, random_rollouts = ["" for _ in range(17)]

    for line in lines:

        parameter_to_display = [param in line for param in params.PARAMS_EVAL]
        if True not in parameter_to_display:
            continue

        line = [x.strip("\n") for x in line.split(" ")]
        if "BATCH_SIZE" in line:  batch_size = line[2]
        if "GAMMA" in line:  gamma = line[2]
        if "TAU" in line:  tau = line[2]
        if "LR_ACTOR" in line:  lr_actor = line[2]
        if "LR_CRITIC" in line:  lr_critic = line[2]
        if "WEIGHT_DECAY" in line:  weight_decay = line[2]
        if "ACTOR_UNITS" in line:  actor_units = ''.join(line[2:])
        if "CRITIC_UNITS" in line:  critic_units = ''.join(line[2:])
        if "ADD_OU_NOISE" in line:  add_ou_noise = line[2]
        if "MU" in line:  mu = line[2]
        if "THETA" in line:  theta = line[2]
        if "SIGMA" in line:  sigma = line[2]
        if "SIGMA_MIN" in line:  sigma_min = line[2]
        if "SIGMA_DECAY" in line:  sigma_decay = line[2]
        if "POLICY_DELAY" in line:  policy_decay = line[2]
        if "DROP_OUT_PROB" in line:  drop_out_prob = line[2]
        if "INITIAL_RANDOM_ROLLOUTS" in line:  random_rollouts = line[2]

    params_dict = {'batch_size': batch_size, 'gamma': gamma, 'tau': tau,
                   'lr_actor': lr_actor, 'lr_critic': lr_critic, 'weight_decay': weight_decay,
                   'actor_units': actor_units, 'critic_units': critic_units, 'add_ou_noise': add_ou_noise,
                   'mu': mu, 'theta': theta, 'sigma': sigma, 'sigma_min': sigma_min,
                   'sigma_decay': sigma_decay, 'policy_decay': policy_decay, 'drop_out_prob': drop_out_prob,
                   'random_rollouts': random_rollouts}

    params_dict = {k: v for k, v in params_dict.items() if v is not ''}
    return params_dict

def get_avg_data(selected_hyperparam, selected_value,twin_delay):
    global sum_rewards
    global sum_rewards_count

    hyper_parameters_values = []
    sum_rewards_count = 0
    sum_rewards = 0

    data_avg = defaultdict(lambda: defaultdict(list))
    data_min = defaultdict(lambda: defaultdict(list))
    data_max = defaultdict(lambda: defaultdict(list))
    data_cur = defaultdict(lambda: defaultdict(list))
    data_med = defaultdict(lambda: defaultdict(list))
    count = 0
    for filename in os.listdir(params.PATH_DASH):
        if 'Logs' not in filename:
            continue
        if selected_hyperparam+"#"+selected_value+"-" not in filename:
            continue
        if twin_delay == "True":
            if "Twin" not in filename:
                continue
        if twin_delay == "False":
            if "Single" not in filename:
                continue
        if os.stat(params.PATH_DASH + "/" + filename).st_size == 0:
            continue
        with open(params.PATH_DASH + "/" + filename, 'r') as f:
            count = count+1
            head = [next(f) for _ in range(40)]
            hyper_parameters = get_hyperparams(head)

            hyper_parameters_values.append(hyper_parameters)

            hyperparameter_name, hyper_parameters_value =  filename.split("-")[2].split("#")

            p_avg, p_min, p_max, p_med, p_cur = [], [], [], [], []

            for line in f:
                if 'Episode' not in line or 'Agents' not in line:
                    continue
                data = line.split(" ")
                p_min.append(float(data[7]))
                p_max.append(float(data[10]))
                p_avg.append(float(data[14]))
                p_med.append(float(data[17]))
                p_cur.append(float(data[20][:-2]))


                data_avg[hyperparameter_name][hyper_parameters_value].append(p_avg)
                if not params.USE_SMOOTHED_MIN_MAX_GRAPH:
                    data_min[hyperparameter_name][hyper_parameters_value].append(p_min)
                    data_max[hyperparameter_name][hyper_parameters_value].append(p_max)
                data_cur[hyperparameter_name][hyper_parameters_value].append(p_cur)
                data_med[hyperparameter_name][hyper_parameters_value].append(p_med)
            # Smooth min and max graphs over 100 Episodes if True
            if params.USE_SMOOTHED_MIN_MAX_GRAPH:
                for i in range(params.RANGE_VALUE_FOR_SMOOTH_GRAPHS):
                    max_val_init = max(p_max[:params.EPISODE_RANGE_DATA])
                    for j in range(params.EPISODE_RANGE_DATA):
                        p_max.append(max_val_init)
                    p_max = p_max[params.EPISODE_RANGE_DATA:]
                data_max[hyperparameter_name][hyper_parameters_value].append(p_max)

                for k in range(params.RANGE_VALUE_FOR_SMOOTH_GRAPHS):
                    min_val_init = min(p_min[:params.EPISODE_RANGE_DATA])
                    for l in range(params.EPISODE_RANGE_DATA):
                        p_min.append(min_val_init)
                    p_min = p_min[params.EPISODE_RANGE_DATA:]
                data_min[hyperparameter_name][hyper_parameters_value].append(p_min)
        sum_rewards_count = sum_rewards_count + sum(p_cur)
    sum_rewards = sum_rewards_count/count
    return data_avg, data_min, data_max, data_med, data_cur

def get_avg_graphs(selected_hyperparam, selected_value, twin_delay):
    traces_array = []
    data_avg, data_min, data_max, data_med, data_cur = get_avg_data(selected_hyperparam, selected_value, twin_delay)
    if params.STATISTICS['avg']:
        traces_array.append((plot_mean_graph(data_avg, "Average", twin_delay)))

    if params.STATISTICS['min']:
        traces_array.append(plot_mean_graph(data_min, "Min", twin_delay))

    if params.STATISTICS['max']:
        traces_array.append(plot_mean_graph(data_max, "Max", twin_delay))

    if params.STATISTICS['med']:
        traces_array.append(plot_mean_graph(data_med, "Med", twin_delay))

    if params.STATISTICS['cur']:
        traces_array.append(plot_mean_graph(data_cur, "Current", twin_delay))

    overview_traces = plot_overview_graph({'avg': data_avg, 'max': data_max, 'min': data_min, 'cur': data_cur, 'med': data_med},
                             "Overview", twin_delay)

    for overview in overview_traces:
       traces_array.append(overview)
    return traces_array

def get_multi_value_graphs(selected_hyperparam, selected_evaluation):
    hyper_parameters_values = []

    data_avg = defaultdict(lambda:(defaultdict(lambda: defaultdict(list))))
    data_min = defaultdict(lambda:(defaultdict(lambda: defaultdict(list))))
    data_max = defaultdict(lambda:(defaultdict(lambda: defaultdict(list))))
    data_cur = defaultdict(lambda:(defaultdict(lambda: defaultdict(list))))
    data_med = defaultdict(lambda:(defaultdict(lambda: defaultdict(list))))

    for filename in os.listdir(params.PATH_DASH):
        if 'Logs' not in filename:
            continue
        if selected_hyperparam+"#" not in filename:
            continue
        if os.stat(params.PATH_DASH + "/" + filename).st_size == 0:
            continue
        with open(params.PATH_DASH + "/" + filename, 'r') as f:
            head = [next(f) for _ in range(40)]
            hyper_parameters = get_hyperparams(head)

            hyper_parameters_values.append(hyper_parameters)

            hyperparameter_name, hyper_parameters_value =  filename.split("-")[2].split("#")

            twin_delay = False
            if 'Twin' in filename:
                 twin_delay = True
                 twin_key = 'Twin'
            else:
                twin_delay = False
                twin_key = 'Single'

            p_avg, p_min, p_max, p_med, p_cur = [], [], [], [], []

            for line in f:
                if 'Episode' not in line or 'Agents' not in line:
                    continue
                data = line.split(" ")

                p_avg.append(float(data[14]))
                p_min.append(float(data[7]))
                p_max.append(float(data[10]))
                p_med.append(float(data[17]))
                p_cur.append(float(data[20][:-2]))


            if selected_evaluation == "Average":
                data_avg[twin_key][hyperparameter_name][hyper_parameters_value].append(p_avg)
            elif selected_evaluation == "Min":
                if not params.USE_SMOOTHED_MIN_MAX_GRAPH:
                    data_min[twin_key][hyperparameter_name][hyper_parameters_value].append(p_min)
            elif selected_evaluation == "Max":
                if not params.USE_SMOOTHED_MIN_MAX_GRAPH:
                    data_max[twin_key][hyperparameter_name][hyper_parameters_value].append(p_max)
            elif  selected_evaluation == "Current":
                data_cur[twin_key][hyperparameter_name][hyper_parameters_value].append(p_cur)
            elif  selected_evaluation == "Median":
                data_med[twin_key][hyperparameter_name][hyper_parameters_value].append(p_med)

            # Smooth min and max graphs over 100 Episodes if True
            if params.USE_SMOOTHED_MIN_MAX_GRAPH:
                if selected_evaluation == "Max":
                    for i in range(params.RANGE_VALUE_FOR_SMOOTH_GRAPHS):
                        max_val_init = max(p_max[:100])
                        for j in range(100):
                            p_max.append(max_val_init)
                        p_max = p_max[100:]
                    data_max[twin_key][hyperparameter_name][hyper_parameters_value].append(p_max)
                if selected_evaluation == "Min":
                    for k in range(params.RANGE_VALUE_FOR_SMOOTH_GRAPHS):
                        min_val_init = min(p_min[:100])
                        for l in range(100):
                            p_min.append(min_val_init)
                        p_min = p_min[100:]
                    data_min[twin_key][hyperparameter_name][hyper_parameters_value].append(p_min)

    traces_array = []

    overview_traces = plot_overview_multi_val_graph({'avg': data_avg, 'max': data_max, 'min': data_min, 'cur': data_cur, 'med': data_med},
                             "Overview")

    for overview in overview_traces:
       traces_array.append(overview)

    return traces_array

def plot_overview_multi_val_graph(data_list, statistic_type):
    traces = []

    parameter = ""

    for statistics, data in data_list.items():
        for (key, val) in data.items():

            avg_mean = {}

            for (param_key, param_vals) in val.items():
                for param_val, param_data in param_vals.items():
                    mean_values = list(map(avg, zip_longest(*param_data)))
                    avg_mean[param_val] = mean_values


            for p, v in avg_mean.items():
                parameter = statistics

                traces.append(get_plot_traces(list(range(len(v))), v, [], [], [], [], param_name=statistics + "(" + key + ")", param_val=p)[0])

                layout = go.Layout(title=statistic_type + " Scores for " + param_key + ' (' + parameter + ')' , legend=dict(orientation="h"),yaxis={'title': 'average score (p. 100 eps.)'})

    overview = []

    figure = go.Figure(data=traces, layout=layout)
    overview.append(figure)

    return overview


def plot_mean_graph(data, statistic_type, twin_delay):

    if twin_delay == 'True':
        algorithm_name = 'Twin Delay'
    else:
        algorithm_name = 'Single Delay'

    for (key, val) in data.items():
        avg_mean = {}
        for (param_key, param_val) in val.items():

            # mean_values = list(map(mean, zip(*param_val)))
            mean_values = list(map(avg, zip_longest(*param_val)))

            avg_mean[param_key] = mean_values

        traces = []
        layout = go.Layout(title=statistic_type + " Scores for " + key + ' (' + algorithm_name + ')',
                           legend=dict(orientation="h"),
                           yaxis={'title': 'average score (p. 100 eps.)'})
        for p, v in avg_mean.items():
            traces.append(get_plot_traces(list(range(len(v))), v, [], [], [], [], param_name=key, param_val=p)[0])

        figure = go.Figure(data=traces, layout=layout)
        #py.plot(figure, filename='{}/{}-scores_{}.html'.format(params.PATH_GRAPHS, key, statistic_type))

        return figure

def plot_overview_graph(data_list, statistic_type, twin_delay):
    traces = defaultdict(list)
    parameter = ""

    if twin_delay == 'True':
        algorithm_name = 'Twin Delay'
    else:
        algorithm_name = 'Single Delay'

    for statistics, data in data_list.items():
        for (key, val) in data.items():
            avg_mean = {}
            for (param_key, param_val) in val.items():
                parameter = key

                # mean_values = list(map(mean, zip(*param_val)))
                mean_values = list(map(avg, zip_longest(*param_val)))
                avg_mean[param_key] = mean_values


            for p, v in avg_mean.items():
                    traces[key].append(get_plot_traces(list(range(len(v))), v, [], [], [], [],
                                                       param_name=statistics + "(" + key + ")", param_val=p)[0])
    overview = []


    for parameter, trace in traces.items():
        layout = go.Layout(title=statistic_type + " Scores for " + parameter + ' (' + algorithm_name + ')',
                           legend=dict(orientation="h"),
                           yaxis={'title': 'average score (p. 100 eps.)'})
        figure = go.Figure(data=trace, layout=layout)

        overview.append(figure)

    return overview


def avg(x):
    x = [i for i in x if i is not None]
    return sum(x) / len(x)


def display_graphs( figures_array, file_name1, file_name2):
    for figure in figures_array:
        py.plot(figure, filename='{}/{}-scores_{}.html'.format(params.PATH_GRAPHS, file_name1, file_name2))




# TESTS
# display_graphs(get_multi_value_graphs(selected_hyperparam="BATCH_SIZE", selected_evaluation="Average"))
