import copy
import random
import sys
from collections import deque, namedtuple, defaultdict
from itertools import zip_longest

import os
import plotly.graph_objs as go
import torch
import numpy as np
import plotly.offline as py
import params

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_directories():
    if not os.path.exists(params.PATH_FIGURE):
        os.mkdir(params.PATH_FIGURE)
    if not os.path.exists(params.PATH_WEIGHTS):
        os.mkdir(params.PATH_WEIGHTS)
    if not os.path.exists(params.PATH_GRAPHS):
        os.mkdir(params.PATH_GRAPHS)
    if not os.path.exists(params.PATH_LOGS):
        os.mkdir(params.PATH_LOGS)


def copy_weights(copy_from, copy_to):
    for _from, _to in zip(copy_from.parameters(), copy_to.parameters()):
        _to.data.copy_(_from.data)


def soft_update(local_model, global_model):
    for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
        global_param.data.copy_(params.TAU * local_param.data + (1.0 - params.TAU) * global_param.data)


class Noise:
    """ See Ornstein-Uhlenbeck"""

    def __init__(self, size):
        self.mu = np.full((1, size), params.MU)
        self.t = params.THETA
        self.s = params.SIGMA
        self.s_min = params.SIGMA_MIN
        self.s_d = params.SIGMA_DECAY
        self.size = size
        self.noise = None
        self.reset()

    def reset(self):
        self.noise = copy.copy(self.mu)
        self.s = max(self.s * self.s_d, self.s_min)

    def sample(self):
        dx = self.t * (self.mu - self.noise) + np.random.standard_normal(self.size) * self.s
        self.noise += dx
        return self.noise


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=params.BUFFER_SIZE)
        self.batch_size = params.BATCH_SIZE
        self.exp = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        exp = self.exp(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self):
        sample = random.sample(self.buffer, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in sample])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in sample])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sample])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in sample])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in sample])).float().to(DEVICE)
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()


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


def get_histograms(step_dict_count):
    traces = [
        go.Histogram(x=step_dict_count[x * 100:(x + 1) * 100], opacity=0.35,
                     name="Eps. {}-{}".format(x * 100, (x + 1) * 100))
        for x in
        range(int(len(step_dict_count) / 100))]
    layout = go.Layout(barmode='overlay')
    fig = go.Figure(data=traces, layout=layout)
    return fig


def pie_chart(step_dict_count, time):
    labels = ['25% Steps', '50% Steps', '75% Steps', '75%<x100% Steps', '100%']
    for y in range(int(len(step_dict_count) / 100)):
        values = []
        count_25, count_50, count_75, count_75_100, count_100 = 0, 0, 0, 0, 0
        for x in range(100):
            if step_dict_count[x] <= 250:
                count_25 += 1
            elif step_dict_count[x] <= 500:
                count_50 += 1
            elif step_dict_count[x] <= 750:
                count_75 += 1
            elif step_dict_count[x] <= 1000:
                count_75_100 += 1
            else:
                count_100 += 1
            values = [count_25, count_50, count_75, count_75_100, count_100]
        trace_pie_chart = go.Pie(labels=labels, values=values, hoverinfo='label+value')
        py.plot([trace_pie_chart], filename='{}/pie_chart-{}-{}.html'.format(params.PATH_GRAPHS, (y + 1) * 100, time))
        step_dict_count = step_dict_count[100:]


class Logger(object):
    terminal = sys.stdout

    def __init__(self, filename):
        self.log = open(filename, "w")

    def write(self, message):
        Logger.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def __del__(self):
        print('\nDeleted logger: {}\n'.format(id(self)))

    def flush(self):
        pass


class LogParser:

    def get_hyperparams(self, lines):
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

    def compute_avg_graphs(self):
        hyper_parameters_values = []

        data_avg = defaultdict(lambda: defaultdict(list))
        data_min = defaultdict(lambda: defaultdict(list))
        data_max = defaultdict(lambda: defaultdict(list))
        data_cur = defaultdict(lambda: defaultdict(list))
        data_med = defaultdict(lambda: defaultdict(list))

        for filename in os.listdir(params.PATH_DASH):
            if 'Logs' not in filename:
                continue
            if os.stat(params.PATH_DASH + "/" + filename).st_size == 0:
                continue

            with open(params.PATH_DASH + "/" + filename, 'r') as f:
                head = [next(f) for _ in range(40)]
                hyper_parameters = self.get_hyperparams(head)

                hyper_parameters_values.append(hyper_parameters)

                hyperparameter_name, hyper_parameters_value =  filename.split("-")[2].split("#")

                print(hyperparameter_name, hyper_parameters_value)
                # print(self.get_hyperparams(head))

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
                data_min[hyperparameter_name][hyper_parameters_value].append(p_min)
                data_max[hyperparameter_name][hyper_parameters_value].append(p_max)
                data_cur[hyperparameter_name][hyper_parameters_value].append(p_cur)
                data_med[hyperparameter_name][hyper_parameters_value].append(p_med)

                # for (param_name, param_val) in hyper_parameters.items():
                #     for parameter in hyper_parameters_values:
                #         if param_name in parameter and parameter[param_name] is param_val :
                #             data_avg[param_name][param_val].append(p_avg)
                #             data_min[param_name][param_val].append(p_min)
                #             data_max[param_name][param_val].append(p_max)
                #             data_cur[param_name][param_val].append(p_cur)
                #             data_med[param_name][param_val].append(p_med)
                #             break

        if params.STATISTICS['avg']:
            self.plot_mean_graph(data_avg, "Average")

        if params.STATISTICS['min']:
            self.plot_mean_graph(data_min, "Min")

        if params.STATISTICS['max']:
            self.plot_mean_graph(data_max, "Min")

        if params.STATISTICS['med']:
            self.plot_mean_graph(data_med, "Med")

        if params.STATISTICS['cur']:
            self.plot_mean_graph(data_min, "Current")

        self.plot_overview_graph({'avg': data_avg, 'max': data_max, 'min': data_min, 'cur': data_cur, 'med': data_med},
                                 "Overview")

    def plot_mean_graph(self, data, statistic_type):
        for (key, val) in data.items():
            avg_mean = {}
            for (param_key, param_val) in val.items():

                # mean_values = list(map(mean, zip(*param_val)))
                mean_values = list(map(self.avg, zip_longest(*param_val)))

                avg_mean[param_key] = mean_values

            traces = []
            layout = go.Layout(title=statistic_type + " Scores for " + key, xaxis={'title': 'episodes'},
                               yaxis={'title': 'average score (p. 100 eps.)'})
            for p, v in avg_mean.items():
                traces.append(get_plot_traces(list(range(len(v))), v, [], [], [], [], param_name=key, param_val=p)[0])

            figure = go.Figure(data=traces, layout=layout)
            #return figure
            py.plot(figure, filename='{}/{}-scores_{}.html'.format(params.PATH_GRAPHS, key, statistic_type))

    def plot_overview_graph(self, data_list, statistic_type):
        traces = defaultdict(list)
        parameter = ""
        for statistics, data in data_list.items():
            for (key, val) in data.items():
                avg_mean = {}
                for (param_key, param_val) in val.items():
                    parameter = key

                    # mean_values = list(map(mean, zip(*param_val)))
                    mean_values = list(map(self.avg, zip_longest(*param_val)))

                    avg_mean[param_key] = mean_values

                for p, v in avg_mean.items():
                    if parameter is key:
                        traces[key].append(get_plot_traces(list(range(len(v))), v, [], [], [], [],
                                                           param_name=statistics + "(" + key + ")", param_val=p)[0])

        for parameter, trace in traces.items():
            layout = go.Layout(title=statistic_type + " Scores for " + parameter, xaxis={'title': 'episodes'},
                               yaxis={'title': 'average score (p. 100 eps.)'})
            figure = go.Figure(data=trace, layout=layout)
            py.plot(figure, filename='{}/{}-scores_{}.html'.format(params.PATH_GRAPHS, parameter, statistic_type))


    def avg(self, x):
        x = [i for i in x if i is not None]
        return sum(x) / len(x)


# READ LOGS AND DISPLAY GRAPH DATA
log_parser = LogParser()

# Set PARAMS_EVAL and STATISTICS in params.py
# log_parser.compute_avg_graphs()

########### TESTING ###########

def get_dummyFigure():
    trace1 = go.Bar(
        x=['giraffes', 'orangutans', 'monkeys'],
        y=[20, 14, 23],
        name='SF Zoo'
    )
    trace2 = go.Bar(
        x=['giraffes', 'orangutans', 'monkeys'],
        y=[12, 18, 29],
        name='LA Zoo'
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

########## TESTING END ##########