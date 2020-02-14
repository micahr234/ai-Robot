import gym
from agentTorchDiscrete import AgentTorchDiscrete

# Run simulation
env = gym.make('Pendulum-v0')
agent = AgentTorchDiscrete(env, "test30", num_of_hypothetical_actions=9)
cumulative_score = 0
episodes = 200

for i_episode in range(episodes):
    score = 0
    t = 0
    observation = env.reset()
    env.render()

    while True:

        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        score += reward

        env.render()

        if done:

            print("Episode " + str(i_episode) + " finished after " + str(t + 1) + " timesteps. Cumulative reward = " + str(score))
            cumulative_score += score
            break

        observation = next_observation
        t += 1

env.close()
print("Average score " + str(cumulative_score / episodes))