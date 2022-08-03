import gym_2048
import gym
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from agent import Agent



if __name__ == '__main__':
    env = gym.make('2048-v0')


    agent = Agent()

    CONTINUE_TRAINING = False
    DEMONSTRATION = False


    ep_reward_list = []
    avg_reward_list = []

    if CONTINUE_TRAINING or DEMONSTRATION:
        agent.actor_model.load_weights(f"weights/stealth_actor.h5")
        agent.critic_model.load_weights(f"weights/stealth_critic.h5")

        agent.actor_target_model.load_weights(f"weights/stealth_target_actor.h5")
        agent.critic_target_model.load_weights(f"weights/stealth_target_critic.h5")


    for i in tqdm(range(1000000)):
        prev_state = env.reset()
        prev_state = prev_state.flatten()
        # env.render()
        ep_reward = 0
        invalid_moves = 0
        invalid_move_amount = 10

        done = False
        moves = 0

        while not done:
            prev_state = prev_state
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = agent.policy(tf_prev_state, agent.ou_noise)

            state, reward, done, info = env.step(action)

            if (reward != 0):
                invalid_moves = invalid_move_amount
            if (reward == 0):
                invalid_moves += 1
            
            if invalid_moves > invalid_move_amount:
                break


            state = state.flatten()

            if not DEMONSTRATION:
                agent.memory.record((prev_state, action, reward, state))
                ep_reward += reward
                agent.learn()
                agent.update_target(agent.actor_target_model.variables, agent.actor_model.variables, agent.tau)
                agent.update_target(agent.critic_target_model.variables, agent.critic_model.variables, agent.tau)

            prev_state = state

            moves += 1

            if DEMONSTRATION:
                print('Next Action: "{}"\n\nReward: {}'.format(
                    gym_2048.Base2048Env.ACTION_STRING[action - 1], reward))
                env.render()

        ep_reward_list.append(ep_reward)
        avg_reward_list.append(np.mean(ep_reward_list[-40:]))
        # print(f"Episode * {i} * Avg Reward is ==> {avg_reward_list[-1]}")
        # print('\nTotal Moves: {}'.format(moves))
        if i % 500 == 0 and not DEMONSTRATION:
            agent.actor_model.save_weights(f"weights/stealth_actor.h5")
            agent.critic_model.save_weights(f"weights/stealth_critic.h5")

            agent.actor_target_model.save_weights(f"weights/stealth_target_actor.h5")
            agent.critic_target_model.save_weights(f"weights/stealth_target_critic.h5")
            

    if not DEMONSTRATION:
        agent.actor_model.save_weights(f"weights/stealth_actor.h5")
        agent.critic_model.save_weights(f"weights/stealth_critic.h5")

        agent.actor_target_model.save_weights(f"weights/stealth_target_actor.h5")
        agent.critic_target_model.save_weights(f"weights/stealth_target_critic.h5")

        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.savefig("Avg_rewards")
        plt.show()
