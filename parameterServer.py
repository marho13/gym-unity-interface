import numpy as np
import torch
from model import *
from environment import giveEnv
from memory import *

class ParameterServer(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, envName, numAgents):
        self.num_agents = numAgents
        self.model = ConvNet(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.env = giveEnv(envName)
        self.memory = Memory()

    def getActorGrad(self, grad, rew):  # My idea could decrease the number in np.divide
        get_weighted_grads = self.calcAvgGrad(*self.getWeightedGrads(grad, rew))#avg_grad = self.calcAvgGrad(*grad)
        output = []
        for g in grad:
            temp = np.add(g, get_weighted_grads)#avg_grad)
            output.append(temp)

        return output

    def highestRewGrad(self, grad, rew):
        index = self.getHighestRew(rew)
        self.updater(grad[index])
        return self.model.get_weights()

    def getHighestRew(self, rew):
        maxy = rew[0]
        currentLargest = 0
        for a in range(len(rew)):
            if rew[a] > maxy:
                maxy = rew[a]
                currentLargest = a
        return currentLargest

    def rewardweighted(self, grad, rew):
        output = self.getWeightedGrads(grad, rew)
        self.calcAvgGrad(*output)
        return self.model.get_weights()

    def getWeightedGrads(self, grad, rew):
        reward, miny = self.MoveRewards(rew)
        totRew = self.getTotRew(reward)
        output = []
        for g in range(len(grad)):
            if reward[g] == 0.0:
                weight = (1/totRew)
            else:
                weight = reward[g]/totRew
            output.append([])
            for x in range(len(grad[g])):
                output[g].append((grad[g][x] * weight))
        return output

    def MoveRewards(self, rew):
        minimum = min(rew)
        output = []
        for r in rew:
            output.append(r+minimum)
        return output, minimum

    def getTotRew(self, rew):
        sum = 0
        for r in rew:
            sum += r
        return sum

    def calcAvgGrad(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.updater(summed_gradients)
        # summed_gradients = np.divide(summed_gradients, self.num_agent) #The averaged gradient
        return summed_gradients

    def updater(self, gradient):
        self.optimizer.zero_grad()
        self.model.set_gradients(gradient)
        self.optimizer.step()

    def get_weights(self):
        return self.model.get_weights()

    def performNActions(self, N):
        state = self.env.reset()
        totRew = 0.0
        for t in range(N):
            state = np.expand_dims(state, axis=0)
            prevState = torch.from_numpy(state.copy())

            action, dist = self.model.act(state.copy())
            action = action.detach().numpy()
            action[0][0] = (action[0][0] * 2) - 1
            # print(action, action.shape)
            state, rew, done, info = self.env.step(
                action[0].copy())  # .item())#actionTranslated)#Need to make it [1, actionsize]
            totRew += rew
            if done:
                break
        return totRew

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]