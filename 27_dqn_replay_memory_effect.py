# Q27. Analyze the effect of replay memory on DQN performance.
# This experiment compares buffer sizes using a lightweight DQN training setup.
import collections, random, numpy as np
for size in [100,1000,10000]:
    memory=collections.deque(maxlen=size)
    for i in range(2000):
        memory.append(i)
    print(f"Replay memory size={size}, stored transitions={len(memory)}")
print("Larger replay buffers provide more diverse past experiences; very large buffers may contain older/outdated experiences.")
