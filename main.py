from myosuite.utils import gym
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
  env.mj_render()
  env.step(env.action_space.sample()) # take a random action
env.close()