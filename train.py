import gym
from agentTorchDiscrete import AgentTorchDiscrete

# Run simulation
env = gym.make('Pendulum-v0')
#random score = -1221
#1 actions 5, epochs 600 lr 0.0001 nlf = 0.0, score = -153
#2 actions 5, epochs 600 lr 0.0001 nlf = 0.1, score = -143
#3 actions 5, epochs 300, lr = 0.0001, nlf = 0.0, probMultiple = 0.2, score = -163
#4 actions 5, epochs 300, lr = 0.0001, nlf = 0.0, probMultiple = 0.1, score = -188
#5 actions 9, epochs 300, lr = 0.0001, nlf = 0.00, probMultiple = 0.1, score = -145
#6 actions 9, epochs 300, lr = 0.0001, nlf = 0.01, probMultiple = 0.1, score = ~-900
#7 actions 9, epochs 300, lr = 0.0001, nlf = 0.00, probMultiple = 0.1, random actions, score = bad
#8 actions 9, epochs 300, lr = 0.0001, nlf = 0.00, probMultiple = 0.05, random actions, score = bad
#9 actions 9, epochs 300, lr = 0.0001, nlf = 0.00, probMultiple = 0.01, random actions, score = bad
#10 actions 9, epochs 300, lr = 0.0001, nlf = 0.00, probMultiple = 0.1, fixed action + random, score = -149
#11 actions 9, epochs 300, lr = 0.0001, nlf = 0.00, probMultiple = 0.1, fixed action + random, PyTorch, score = -149
#next_learn_factor=0.3
#AgentTorchDiscrete(env, "test29", discount=0.99, batch_size=1000, learn_rate=0.001, policy_learn_rate=0.0001, learn_iterations=10, memory_buffer_size=1000000, next_learn_factor=0.0, num_of_hypothetical_actions=9)
agent = AgentTorchDiscrete(env, "test30", discount=0.99, batch_size=1000, value_learn_rate=0.0001, policy_learn_rate=0.001, max_policy_learn_rate=0.0001, learn_iterations=10, memory_buffer_size=1000000, next_learn_factor=0.0, num_of_hypothetical_actions=9)
probMultiple = 0.01
episodes = 300

for n in range(episodes):

    observation = env.reset()
    env.render()

    score = 0
    t = 0

    while True:

        #action_random = np.random.normal(0, 0.01)
        action = agent.act(observation, prob_factor=probMultiple)# + action_random
        next_observation, reward, done, info = env.step(action)
        agent.record(observation, action, reward, next_observation, done)
        score += reward

        env.render()

        if done:

            print("Episode " + str(n) + " finished after " + str(t + 1) + " timesteps - cumulative reward = " + str(score))
            agent.learn()
            break

        observation = next_observation
        t += 1

env.close()
