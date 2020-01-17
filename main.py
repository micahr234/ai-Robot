import gym
import numpy as np

from agent import Agent

# Run simulation
env = gym.make('Pendulum-v0')
agent = Agent(env, "test", action_temperature=5, action_temperature_multiplier=1, discount=0.9999, batch_size=200, learn_rate=0.0001, memory_buffer_size=200000, next_learn_factor=0.1, num_of_hypothetical_actions=100)


for i_episode in range(200000):
    score = 0
    t = 0
    observation = env.reset()
    env.render()

    while True:

        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        agent.react(observation, action, reward, next_observation, done)
        score += reward

        env.render()
        #print("Episode: " + str(i_episode) + "\t\tTimestep: " + str(t) + "\t\tReward " + str(reward) + "\t\tAction " + str(action) + "\t\tObservation " + str(observation))

        if done:

            print("Episode " + str(i_episode) + " finished after " + str(t + 1) + " timesteps. Cumulative reward = " + str(score))
            break

        observation = next_observation
        t += 1

env.close()
