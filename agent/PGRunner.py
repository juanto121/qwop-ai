from Game import Game

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import time

H = 200
D = 80 * 69

gamma = 0.99  # discount factor
learning_rate = 1e-3
batch_size = 5
END_GAME_FLAG = 1210


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.input = nn.Linear(D, H)
        self.hidden = nn.Linear(H, 5)

    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.softmax(self.hidden(x), dim=0)
        return x


def map_action(action):
    return (['q', 'w', 'o', 'p', 'n'])[action]


def discount_rewards(reward_log):
    discount = 0
    discounted_rewards = np.zeros_like(reward_log)
    for idx in reversed(range(0, discounted_rewards.size)):
        if reward_log[idx] != 0: discount = 0
        discount = gamma * discount + reward_log[idx]
        discounted_rewards[idx] = discount
    return discounted_rewards


def main():
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    env = Game()
    env.start()

    observation, reward, done = env.execute_action('n')

    prev_x = None
    curr_x = observation
    reward_pool = []
    prob_pool = []
    steps = 0
    reward_sum = 0
    game = 0
    last_reward = 0
    running_reward = 0
    in_game_step = 0

    while True:

        for step in range(500):

            in_game_step += 1

            x_diff = curr_x - prev_x if prev_x is not None else curr_x
            prev_x = curr_x
            x = Variable(torch.from_numpy(x_diff).float())

            forward = policy(x)
            out_dist = Categorical(forward)
            action = out_dist.sample()

            curr_x, reward, done = env.execute_action(map_action(action.item()))

            if done: in_game_step = 0

            if in_game_step > 0 and in_game_step % 100 == 0:
                print('should have reloaded')
                reward = env.get_score()
                env.soft_reload()

            prob_pool.append(out_dist.log_prob(action))

            reward_pool.append(reward)
            reward_sum += reward

            if reward != last_reward and reward != 0:
                game += 1
                print(f"{time.time()} episode: {steps}, game: ${game} reward {reward} {'😋' if reward == 1 else ''}")
                last_reward = reward

        steps += 1
        print(f'End of sub batch {steps}')

        if steps > 0 and steps % batch_size == 0:
            print(f'calling backprop batch {steps}')
            discounted_rewards = np.array(discount_rewards(reward_pool)).ravel()
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            optimizer.zero_grad()
            policy_loss = []

            for prob, dis_reward in zip(prob_pool, discounted_rewards):
                policy_loss.append(-prob * dis_reward)

            loss_fn = torch.cat(policy_loss).sum()
            loss_fn.backward()
            optimizer.step()

            del prob_pool[:]
            del reward_pool[:]

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            reward_sum = 0
            prev_x = None
            curr_x = env.reload()
            last_reward = 0


if __name__ == '__main__':
    main()
