import gym
import numpy as np
import time
import cProfile
import os

# Define run
def Run(env, agent, max_timestep, learn_interval, save_interval, render=True, delay=0.0, profile=False, enable_eposide_timestep=True, noise_power=0):

    action_space_min = np.array(env.action_space.low)
    action_space_max = np.array(env.action_space.high)

    episode_timestep = 1
    done = True
    learn_count = 0
    save_count = 0

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    for timestep in range(1, max_timestep + 1):

        episode_timestep += 1

        if done:
            episode_timestep = 1
            episode_cumulative_reward = 0
            next_observation_partial = env.reset()
            next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if enable_eposide_timestep else next_observation_partial

            if render: env.render()
            if render: time.sleep(delay)

        observation = next_observation

        action = agent.act(observation)
        limited_action = np.clip(action, -1, 1)
        corrupted_action = np.clip(limited_action + noise_power * np.random.randn(*limited_action.shape), -1, 1)
        scaled_action = (corrupted_action / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
        next_observation_partial, reward, done, info = env.step(scaled_action)
        next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if enable_eposide_timestep else next_observation_partial
        agent.record(observation, action, reward, next_observation, done)

        episode_cumulative_reward += reward
        if done: print('Episode finished\t\tEpisode timestep: ' + str(episode_timestep) + '\t\tTimestep: ' + str(
            timestep) + '\t\tTotal reward: ' + str(episode_cumulative_reward))

        if render: env.render()
        if render: time.sleep(delay)

        if done:
            if (timestep / learn_interval) >= (learn_count + 1):
                learn_count += 1
                agent.learn()
            if (timestep / save_interval) >= (save_count + 1):
                save_count += 1
                agent.save()

    agent.save()
    env.close()

    if profile:
        pr.disable()
        pr.dump_stats('profile.dat')
        os.system("snakeviz profile.dat")