import numpy as np
import time
import cProfile

# Define run
def Run(env, agent, max_timestep, learn_interval, save_interval, render=True, delay=0.0, profile=False):

    episode_timestep = 1
    done = True

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    for timestep in range(1, max_timestep + 1):

        episode_timestep += 1

        if done:
            episode_timestep = 1
            episode_cumulative_reward = 0
            next_observation_partial = env.reset()
            next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1)))
            if render: env.render()
            if render: time.sleep(delay)

        observation = next_observation

        action = agent.act(observation)
        next_observation_partial, reward, done, info = env.step(action)
        next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1)))
        agent.record(observation, action, reward, next_observation, done)

        episode_cumulative_reward += reward

        if done: print('Episode finished\t\tEpisode timestep: ' + str(episode_timestep) + '\t\tTimestep: ' + str(
            timestep) + '\t\tTotal reward: ' + str(episode_cumulative_reward))

        if render: env.render()
        if render: time.sleep(delay)

        if (timestep % learn_interval) == 0: agent.learn()
        if (timestep % save_interval) == 0: agent.save()

    agent.save()
    env.close()

    if profile:
        pr.disable()
        pr.dump_stats('profile.dat')