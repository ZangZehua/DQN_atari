import torch
import datetime
from tensorboardX.writer import SummaryWriter

from algos.dqn_agent import DQN
from common.atari_wrappers import *
from common.config import Config


class Runner:
    def __init__(self):
        self.config = Config()
        self.env = make_atari(self.config.env_name)
        self.env = wrap_deepmind(self.env, scale=False, frame_stack=self.config.frame_stack)
        self.action_space = self.env.action_space
        self.state_dim = self.env.observation_space.shape[0]
        self.state_channel = self.env.observation_space.shape[2]
        self.dqn_agent = DQN(self.config.device, 4, self.action_space, self.config.memory_size,
                             epsilon_max=self.config.epsilon_max, epsilon_min=self.config.epsilon_min,
                             epsilon_decay=self.config.eps_decay, lr=self.config.lr, gamma=self.config.gamma)

    def train(self):
        reward_writer = SummaryWriter(self.config.reward_writer_path)
        loss_writer = SummaryWriter(self.config.loss_writer_path)
        log_writer = open(self.config.log_path, "w+")
        episode = 0
        all_rewards = []
        episode_reward = 0
        losses = []

        step = 0
        learning_start = 5000
        loss = 0

        train_begin_time = datetime.datetime.now()
        last_train_time = train_begin_time
        observation = self.env.reset()
        while step < self.config.max_step:
            state = self.dqn_agent.get_state(observation)
            epsilon = self.dqn_agent.epsilon_by_step(step)
            action = self.dqn_agent.select_action(state, epsilon)
            next_observation, reward, done, _ = self.env.step(action)
            episode_reward += reward

            self.dqn_agent.memory.append([observation, action, reward, next_observation, done])
            observation = next_observation
            step += 1

            if self.dqn_agent.memory.size() > learning_start:
                loss = self.dqn_agent.learn(self.config.batch_size)
                losses.append(loss)
                loss_writer.add_scalar("loss", loss, step)

            if step % self.config.print_interval == 0:
                train_time = datetime.datetime.now()
                log_writer.write("step:{}\tepisode:{}\treward:{:.4f}\tloss:{:.8f}\tepsilon:{:.8f}\ttime_spent:{}\t{}"
                                 .format(step, episode, np.mean(all_rewards[-10:]), loss, epsilon, train_time-train_begin_time, train_time-last_train_time))
                log_writer.flush()
                print("step:{}\tepisode:{}\treward:{:.4f}\tloss:{:.8f}\tepsilon:{:.8f}\ttime_spent:{}\t{}"
                      .format(step, episode, np.mean(all_rewards[-10:]), loss, epsilon, train_time-train_begin_time, train_time-last_train_time))
                last_train_time = train_time

            if step % self.config.update_target_interval == 0:
                self.dqn_agent.target_net.load_state_dict(self.dqn_agent.policy_net.state_dict())

            if done:
                if episode_reward > max(all_rewards):
                    log_writer.write("net saved at:{}\n".format(step))
                    log_writer.flush()
                    print("net saved at:{}".format(step))
                    torch.save(self.dqn_agent.policy_net.state_dict(), self.config.model_save_path)
                observation = self.env.reset()
                all_rewards.append(episode_reward)
                reward_writer.add_scalar("train_reward", episode_reward, episode)
                episode_reward = 0
                episode += 1
        reward_writer.close()
        loss_writer.close()
        log_writer.close()

    def eval(self, model_path,eval_times=3):
        self.dqn_agent.policy_net.load_state_dict(model_path)
        self.dqn_agent.policy_net.eval()
        all_rewards = []
        episode_reward = 0
        all_steps = []
        episode_step = 0

        observation = self.env.reset()
        for episode in range(eval_times):
            state = self.dqn_agent.get_state(observation)
            action = self.dqn_agent.select_action(state, 0)  # don't greedy
            next_observation, reward, done, _ = self.env.step(action)
            episode_reward += reward
            observation = next_observation

            if done:
                observation = self.env.reset()
                all_rewards.append(episode_reward)
                all_steps.append(episode_step)
                print("episode:{}\tstep:{}\treward:{}")
                episode_reward = 0
                episode_step = 0
                episode += 1
        print("Test Done!")
        print("Rewards mean:{}".format(np.mean(all_rewards)))
