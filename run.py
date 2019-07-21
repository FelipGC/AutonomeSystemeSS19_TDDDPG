from collections import defaultdict, deque
from mlagents.envs import UnityEnvironment
from td_ddpg_agent import TDAgent
from ddpg_agent import Agent
import sys
import atexit
import numpy as np
import params
import torch
import datetime
import pickle
import plotly.graph_objs as go


from utils import DEVICE, create_directories, get_plot_traces, get_histograms, Logger

AVG_REWARD_POINTS_QUEUE = {}
AVG_REWARD_POINTS_DEQUEUE = {}
AVG_REWARD_POINTS = defaultdict(list)
MIN_REWARD_POINTS = defaultdict(list)
MAX_REWARD_POINTS = defaultdict(list)
CURRENT_SCORES = defaultdict(list)
MEDIAN_REWARD_POINTS = defaultdict(list)
BRAIN_NAME_TO_AGENT = {}
NOW = datetime.datetime.now()
STEP_DICT_COUNT = defaultdict(list)

# Array for figure-saving to load in dashboard
figures = []


def create_agents(env, twin_delay):
    env_info_all = env.reset(train_mode=True)
    for brain_name in env.brain_names:
        brain = env.brains[brain_name]
        env_info = env_info_all[brain_name]
        output_dim = brain.vector_action_space_size[0]
        input_count = env_info.vector_observations.shape[1]

        print("Creating agents for: {}".format(brain_name))
        print("   -> Input: {}".format(input_count))
        print("   -> Output: {}".format(output_dim))
        agent_count = len(env_info.agents)
        print('Number of agents:', agent_count)
        state = env_info.vector_observations[0]
        print('States have the form of:', state)
        if twin_delay:
            agent = TDAgent(brain_name, input_count, output_dim, agent_count)
        else:
            agent = Agent(brain_name, input_count, output_dim, agent_count)
        BRAIN_NAME_TO_AGENT[brain_name] = agent
        AVG_REWARD_POINTS_QUEUE[brain_name] = deque(maxlen=params.EPISODE_RANGE_DATA)
        AVG_REWARD_POINTS_DEQUEUE[brain_name] = deque(maxlen=params.EPISODE_RANGE_DATA)

        # Load model if not in trainings mode
        if not params.TRAINING_MODE:
            agent.load_model('{}/best_checkpoint_actor_{}.pth'.format(params.PATH_WEIGHTS, brain_name))


def perform_step(env, env_info_all, episode_brain_reward, episode_nr):
    action_dict = {}
    info_state_action = {}
    for brain_name in BRAIN_NAME_TO_AGENT.keys():
        env_info = env_info_all[brain_name]
        agent = BRAIN_NAME_TO_AGENT[brain_name]
        states = env_info.vector_observations
        # First EPISODE_RANGE_DATA episodes are completely random
        if episode_nr <= params.EPISODE_RANGE_DATA and params.TRAINING_MODE and params.INITIAL_RANDOM_ROLLOUTS:
            actions = np.random.uniform(-1, 1, (agent.agent_count, agent.action_size))
        else:
            actions = agent.get_actions(states)
        # Save action state pair
        info_state_action[brain_name] = (states, actions)
        action_dict[brain_name] = actions
    # Do step in env relative to the actions of all agents (and brain)
    env_info_all = env.step(action_dict)
    # Handle step
    any_done = False

    for brain_name in BRAIN_NAME_TO_AGENT.keys():
        env_info = env_info_all[brain_name]
        agent = BRAIN_NAME_TO_AGENT[brain_name]
        dones = env_info.local_done
        any_done = any_done or np.any(dones)
        rewards = env_info.rewards
        states_old, actions = info_state_action[brain_name]
        next_states = env_info.vector_observations
        # Store information
        agent.store_transitions(states_old, actions, rewards, next_states, dones)
        # Accumulate reward
        episode_brain_reward[brain_name] += np.sum(rewards)
        # Do not train if not in trainings mode
        if not params.TRAINING_MODE:
            continue
        # Once data has been stored, learn (== step())
        agent.step()
    return env_info_all, any_done


def run(env):
    create_agents(env, params.TWIN_DELAY)
    # Start episodes
    print('=' * 60)
    print('Starting training process:')
    print('=' * 60)
    for episode in range(1, params.MAX_EPISODES + 1):
        episode_brain_reward = defaultdict(int)
        env_info_all = env.reset(train_mode=True)
        done = False
        current_step = 0
        while not done and current_step < params.MAX_STEP:
            current_step += 1
            env_info_all, done = perform_step(env, env_info_all, episode_brain_reward, episode)
        # Calculate average score
        for brain_name in BRAIN_NAME_TO_AGENT.keys():
            env_info = env_info_all[brain_name]
            agent_count = len(env_info.agents)
            agent = BRAIN_NAME_TO_AGENT[brain_name]
            agent.reset_noise()
            r = episode_brain_reward[brain_name]
            current_score = r / agent_count
            AVG_REWARD_POINTS_QUEUE[brain_name].append(current_score)
            AVG_REWARD_POINTS_DEQUEUE[brain_name].append(current_score)
            STEP_DICT_COUNT[brain_name].append(current_step)
            # Save information
            avg = np.mean(AVG_REWARD_POINTS_DEQUEUE[brain_name])
            median_s = np.median(AVG_REWARD_POINTS_DEQUEUE[brain_name])
            min_s = np.min(AVG_REWARD_POINTS_QUEUE[brain_name])
            max_s = np.max(AVG_REWARD_POINTS_QUEUE[brain_name])
            AVG_REWARD_POINTS[brain_name].append(avg)
            MEDIAN_REWARD_POINTS[brain_name].append(median_s)
            CURRENT_SCORES[brain_name].append(current_score)

            print(
                "\rEpisode: {} {}; Agents: {}; \tMin. Score: {:.2f} \tMax. Score: {:.2f}  \tAvg. Score: {:.2f} \tMed. Score: {:.2f} \tCurrent Score: {:.2f}".format(
                    episode, brain_name,
                    agent_count, min_s, max_s, avg, median_s, current_score), end="")
            if episode % params.EPISODE_RANGE_DATA == 0:
                MIN_REWARD_POINTS[brain_name] += [min_s] * params.EPISODE_RANGE_DATA
                MAX_REWARD_POINTS[brain_name] += [max_s] * params.EPISODE_RANGE_DATA
                AVG_REWARD_POINTS_QUEUE[brain_name].clear()
                agent = BRAIN_NAME_TO_AGENT[brain_name]
                if params.TRAINING_MODE:
                    torch.save(agent.actor_local.state_dict(),
                               '{}/last_checkpoint_actor_{}.pth'.format(params.PATH_WEIGHTS, brain_name))
                    if agent.best_avg_score < avg:
                        torch.save(agent.actor_local.state_dict(),
                                   '{}/best_checkpoint_actor_{}.pth'.format(params.PATH_WEIGHTS, brain_name))
                print(
                    "\rEpisode: {}-{} {}; \tMin. Score: {:.2f} \tMax. Score: {:.2f}  \tAvg. Score: {:.2f} \tMed. Score: {:.2f}".format(
                        episode - 100, episode, brain_name, min_s, max_s, avg, median_s))

                # store best average score
                agent.best_avg_score = max(agent.best_avg_score, avg)

    env.close()


def plot_graph(env):
    # Show plot
    print('\nPlotting graphs...')
    for brain_name in env.brains:
        p_avg = AVG_REWARD_POINTS[brain_name]
        p_current = CURRENT_SCORES[brain_name]
        p = list(range(len(p_avg)))
        p_min = MIN_REWARD_POINTS[brain_name]
        p_max = MAX_REWARD_POINTS[brain_name]
        p_med = MEDIAN_REWARD_POINTS[brain_name]
        data = get_plot_traces(p, p_avg, p_min, p_max, p_med, p_current)

        # add figure to display in dashboard
        figures.append(convertRawData(data))

        time = NOW.strftime("%Y-%m-%d %H-%M")
        # py.plot(data, filename='{}/{}-scores-{}.html'.format(params.PATH_GRAPHS, brain_name, time))

        # plot histogram
        histograms = get_histograms(STEP_DICT_COUNT[brain_name].copy())
        # add figure to display in dashboard
        figures.append(histograms)

        #py.plot(histograms, filename='{}/{}-histogram-{}.html'.format(params.PATH_GRAPHS, brain_name, time))
        # plot pie graph
        #pie_chart(STEP_DICT_COUNT[brain_name].copy(), time)


# auxiliary function for raw trace data
def convertRawData(data):
    layout = go.Layout(xaxis={'title': 'episodes'}, yaxis={'title': 'score'})
    figure = go.Figure(data=data, layout=layout)
    return figure


# auxiliary function to save the figure objects
def save_figures():
    with open('figures/tempfigures.pickle', 'wb') as figures_file:
        print('\nFigures saved!')
        pickle.dump(figures, figures_file)


def main(label='train'):
    try:
        # Create directories if not existent
        create_directories()
        print("""     
           _     __  __   _   _             ___     ___      _      ___   
          | |   |  \/  | | | | |    o O O  / __|   / __|    / |    / _ \  
          | |__ | |\/| | | |_| |   o       \__ \   \__ \    | |    \_, /  
          |____||_|__|_|  \___/   TS__[O]  |___/   |___/   _|_|_   _/_/_  
        "`-0-0-'"`-0-0-'"`-0-0-'./o--000'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'                                                                               
            """)
        print("""Felip Guimerà, Stefanie Kunze, Filip Hristov, Christopher Masurek""")

        logger = Logger("{}/Logs-{}-{}.txt".format(params.PATH_LOGS, label, NOW.strftime("%Y-%m-%d_%H-%M")))
        sys.stdout = logger
        print(NOW.strftime("%Y-%m-%d %H:%M"))
        print("Device: {}".format(DEVICE))
        print("\nParams:\n", )
        params.print_parameters()
        print("\n")
        print("Logs:\n")

        env = UnityEnvironment(file_name="buildings/Linux/3DBall_train_linux.x86_64", worker_id=0, seed=1)
        atexit.register(save_figures)
        atexit.register(plot_graph, env)

        run(env=env)

        # delete logger
        del logger
    except KeyboardInterrupt:
        print("\nTraining interrupted")


if __name__ == '__main__':
    main()
    exit()
