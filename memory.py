from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_terminals', 'logprobs'))

class Memory(object):

    def __init__(self, capacity=1000):
        self.cap = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def getMem(self):
        output = self.memory
        self.memory = deque([], maxlen=self.cap)
        return output

    def __len__(self):
        return len(self.memory)