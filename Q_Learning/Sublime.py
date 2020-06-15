import gym
env = gym.make('MountainCar-v0')
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

'''done = False

while not done:
    action = 2
    env.render()
    new_state, reward, done, _ = env.step(action) # take a random action
    print(new_state)
env.close()'''