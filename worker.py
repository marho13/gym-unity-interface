import numpy as np
import ray
from PPO import *
from environment import giveEnv
from memory import *
import torch

@ray.remote(max_restarts=-1, max_task_retries=2)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, name):
        self.model = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, ConvNet, "cpu", K_epochs=4, eps_clip=0.2)
        self.env = giveEnv(name)
        self.optimizer = torch.optim.SGD(self.model.policy.parameters(), lr=lr)
        self.memory = Memory()

    def compute_gradients(self, weights):
        self.model.policy.set_weights(weights)#Not used for actor version
        rew = self.performNActions(1000)
        loss = self.model.getLossGrad(self.memory)
        # self.memory = Memory()
        return [self.model.policy.get_gradients(), loss, rew]

    def performNActions(self, N):
        state = self.env.reset()
        img = np.array(state[0], dtype=np.float16)
        imugnss = np.array(state[1], dtype=np.float16)
        totRew = 0.0

        for step in range(N):
            prevState = torch.from_numpy(img.copy())
            action, dist = self.model.policy.act(np.expand_dims(img, axis=0))

            state, rew, done, info = self.env.step(action.detach().numpy())

            prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.unsqueeze(torch.from_numpy(state[0].copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            done = torch.unsqueeze(torch.tensor(done), 0)

            self.memory.push(prevState, action, stateMem, rew, done, dist)

            totRew += rew

            if done:
                break

        return totRew

    def getActionLayer(self):
        state = self.env.reset()
        # s = self.numpyToTensor(state)
        return self.model.policy.act(state)

    def getValueLayer(self):
        state = self.env.reset()
        # s = self.numpyToTensor(state)
        action, dist = self.model.policy.act(state)
        # print(action)
        logprobs, stateval, distentropy, actionprobs, actionlogprobs = self.model.policy.tester(state, action)
        return logprobs, stateval, distentropy, actionprobs, actionlogprobs

    def set_grads_calc_weights(self, grad):
        self.optimizer.zero_grad()
        self.model.policy.set_gradients(grad)
        self.optimizer.step()

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]