from model import ConvNet
from PPO import PPO

from memory import Memory
import torch

import environment

import numpy as np
from parameterServer import ParameterServer
from worker import DataWorker
import ray
import wandb

def selfContainedmain(): #Fix the Ros nodes (2 )

    env = environment.giveEnv("/home/martin_holen/Documents/LinuxBuildMLAgents/LinuxBuildBoatMLAgents.x86_64") #, allow_multiple_obs=True #
    image = env.observation_space.spaces[0]
    state_dim = image.shape[0]*image.shape[1]*image.shape[2]#8 #164x164x3
    action_dim = 3
    n_latent_var = 64
    lr = 0.001
    betas = (0.9, 0.99)
    gamma = 0.99
    device = "cpu"
    model = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, ConvNet, device, K_epochs=4, eps_clip=0.2)
    memory = Memory(2000)

    for episode in range(1):
        state = env.reset()
        img = np.array(state[0], dtype=np.float16)
        imugnss = np.array(state[1], dtype=np.float16)
        totRew = 0.0

        for step in range (2000):
            prevState = torch.from_numpy(img.copy())
            action, dist = model.policy.act(np.expand_dims(img, axis=0))

            state, rew, done, info = env.step(action.detach().numpy())

            prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.unsqueeze(torch.from_numpy(state[0].copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            done = torch.unsqueeze(torch.tensor(done), 0)

            memory.push(prevState, action, stateMem, rew, done, dist)

            totRew += rew
            # print(type(done.item()), done.item())
            if done.item():
                print("Episode {}, gave reward: {}".format(episode, totRew.item()))
                # env.reset()
                model.update(memory)
                memory = Memory(2000)
                break

def main():
    envName = "/home/martin_holen/Documents/LinuxBuildMLAgents/LinuxBuildBoatMLAgents.x86_64"
    env = environment.giveEnv(envName)  # , allow_multiple_obs=True #

    image = env.observation_space.spaces[0]

    state_dim = image.shape[0] * image.shape[1] * image.shape[2]  # 8 #164x164x3
    action_dim = 3
    n_latent_var = 64
    lr = 0.001
    betas = (0.9, 0.99)
    gamma = 0.99

    device = "cpu"
    numAgents = 4
    epochs = 1

    ray.init(ignore_reinit_error=True)

    print("Running synchronous parameter server training.")
    #
    wandy = wandb.init(project="Federated-learning-PPO5L-{}-SGD".format(envName),
                       config={
                           "batch_size": 16,
                           "learning_rate": lr,
                           "dataset": envName,
                           "model": "Highest reward single actor weight",
                       })

    for run in range(10):
        ps = ParameterServer(state_dim, action_dim, n_latent_var, lr, betas, gamma, envName, numAgents)
        workers = [DataWorker.remote(state_dim, action_dim, n_latent_var, lr, betas, gamma, envName) for i in
                   range(numAgents)]

        current_weights = ps.get_weights()
        print("Run {}".format(run))
        for i in range(epochs):
            gradients = ray.get([worker.compute_gradients.remote(current_weights) for worker in workers])
            grads, loss, reward = [], [], []
            for output in gradients:
                grads.append(output[0])
                loss.append(output[1].item())
                reward.append(output[2].item())

            avgLoss = sum(loss) / numAgents
            avgRew = sum(reward) / numAgents
            wandb.log({"training loss": avgLoss}, step=i)
            wandb.log({"training reward": avgRew}, step=i)

            current_weights = ps.highestRewGrad(grads, reward)

            if i % 10 == 9:
                rew = ps.performNActions(1000)
                print("Epoch {}, gave reward {}".format(i, rew.item()))
                wandb.log({"testing reward": rew.item()}, step=i)


    wandy.finish()

    ray.shutdown()


if __name__ == '__main__':
    main()

