import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'  # Suppress git executable warning

from myosuite.utils import gym

print("Initializing myosuite environment...")
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()

print("Running simulation with random actions...")
print("Close the window or press Ctrl+C to stop")

step_count = 0
try:
    while True:  # Run indefinitely until window closed
        env.unwrapped.mj_render()  # Use unwrapped to avoid deprecation warning
        env.step(env.action_space.sample())
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Steps: {step_count}")
            
except KeyboardInterrupt:
    print(f"\nStopped after {step_count} steps")
finally:
    env.close()
    print("Environment closed")