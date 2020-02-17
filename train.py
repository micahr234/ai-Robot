import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchDiscrete import AgentTorchDiscrete

# Run simulation
env = gym.make('CartPoleBulletEnv-v1', renders=True, discrete_actions=True)

action_space = [[0, 1]]
state_space_min = [-1, -1, -1, -1]
state_space_max = [1, 1, 1, 1]
reward_space_min = [0]
reward_space_max = [1]
agent = AgentTorchDiscrete("test1", action_space, state_space_min, state_space_max, reward_space_min, reward_space_max,
                           discount=0.999, batch_size=1000, value_learn_rate=0.001, policy_learn_rate=0.0001, policy_copy_rate=0.1,
                           learn_iterations=10, memory_buffer_size=1000000, next_learn_factor=0.5)

learn_episodes = 100
test_episodes = 100
learn_interval = 10
render = True

cumulative_score = 0
for n in range(learn_episodes + test_episodes):

    next_observation = env.reset()
    if render: env.render()

    score = 0
    t = 0

    while True:

        t += 1
        observation = next_observation

        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        agent.record(observation, action, reward, next_observation, done)
        score += reward

        if render: env.render()

        if done:

            if n < learn_episodes:
                print("Learn episode " + str(n) + " finished after " + str(t) + " timesteps - reward = " + str(score))
                if (n % learn_interval) == (learn_interval-1): agent.learn()

            else:
                print("Test episode " + str(n) + " finished after " + str(t) + " timesteps - reward = " + str(score))
                cumulative_score += score

            break

if test_episodes > 0:
    print("Test average score " + str(cumulative_score / test_episodes))

env.close()
