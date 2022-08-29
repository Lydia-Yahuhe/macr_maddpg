from copy import deepcopy

from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from algo.network import Critic, Actor
from algo.memory import ReplayMemory, Experience
from algo.misc import *


def make_exp_id(args):
    return 'train_{}_{}_{}_{}_{}_{}'.format(args.inner_iter, args.meta_final, args.gamma,
                                            args.a_lr, args.c_lr, args.batch_size)


class MADDPG:
    def __init__(self, dim_obs, dim_act, args, record=True):
        self.args = args
        self.var = 1.0

        self.actor = Actor(dim_obs, dim_act)
        self.critic = Critic(dim_obs, dim_act)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.c_lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.a_lr)

        self.memory = ReplayMemory(args.memory_length)
        self.c_loss, self.a_loss = [], []

        if record:
            self.writer = SummaryWriter(logs_path + make_exp_id(args))
        else:
            self.writer = None

        # net_visual([(1, dim_obs)], self.actor, 'actor')
        # net_visual([(1, 2, dim_obs), (1, 2, dim_act)], self.critic, 'critic')

    def load_model(self):
        print("load model!")
        actor = th.load(model_path + "actor.pth")
        critic = th.load(model_path + "critic.pth")
        self.actor.load_state_dict(actor.state_dict())
        self.critic.load_state_dict(critic.state_dict())
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def save_model(self):
        th.save(self.actor, model_path + 'actor.pth')
        th.save(self.critic, model_path + 'critic.pth')

    def scalars(self, key, value, episode):
        self.writer.add_scalars(key, value, episode)

    def scalar(self, key, value, episode):
        self.writer.add_scalar(key, value, episode)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def update(self, step, step_size):
        actor_old_vars = self.actor.state_dict()
        critic_old_vars = self.critic.state_dict()

        actor_new_vars, critic_new_vars = [], []
        for n_agent, transitions in self.memory.sample(self.args.batch_size, num_iter=self.args.inner_iter):
            self.actor.load_state_dict(actor_old_vars)
            self.critic.load_state_dict(critic_old_vars)

            for transition in transitions:
                batch = Experience(*zip(*transition))

                state_batch = th.stack(batch.states).type(FloatTensor)
                action_batch = th.stack(batch.actions).type(FloatTensor)
                reward_batch = th.stack(batch.rewards).type(FloatTensor)
                next_states = th.stack(batch.next_states).type(FloatTensor)

                # 更新Critic
                self.actor.zero_grad()
                self.critic.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                current_q = self.critic(state_batch, action_batch)
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                target_q = target_q * self.args.gamma + reward_batch

                q_loss = nn.MSELoss()(current_q, target_q.detach())
                q_loss.backward()
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.critic_optimizer.step()
                self.c_loss.append(q_loss.detach().numpy())

                # 更新Actor
                self.actor.zero_grad()
                self.critic.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                ac = self.actor(state_batch)

                actor_loss = -self.critic(state_batch, ac).mean()
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.actor_optimizer.step()
                self.a_loss.append(actor_loss.detach().numpy())

            actor_new_vars.append(self.actor.state_dict())
            critic_new_vars.append(self.critic.state_dict())

        self.actor.load_state_dict(interpolate_vars(actor_old_vars, actor_new_vars, step_size))
        self.critic.load_state_dict(interpolate_vars(critic_old_vars, critic_new_vars, step_size))

        if step % 100 == 0:
            soft_update(self.critic_target, self.critic, self.args.tau)
            soft_update(self.actor_target, self.actor, self.args.tau)

            self.writer.add_scalars('L', {'c': np.mean(self.c_loss), 'a': np.mean(self.a_loss)}, step)
            self.c_loss, self.a_loss = [], []

    def choose_action(self, states, noisy=True):
        states = th.from_numpy(np.stack(states)).float().to(device)

        actions = []
        for state in states:
            act = self.actor(state.detach().unsqueeze(0)).squeeze()
            if noisy:
                act += th.from_numpy(np.random.randn(act.shape[-1]) * self.var).type(FloatTensor)
            actions.append(act)
        actions = th.stack(actions)
        actions = th.clamp(actions, -1, 1)

        if self.var > 0.05:
            self.var *= 0.99995

        return actions.data.cpu().numpy()
