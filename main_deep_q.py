from time import sleep
import gym_2048
import gym
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from deep_q_agent import DeepQ


def demonstrate(agent):
    state = env.reset()
    state = state.flatten()
    # env.render()
    ep_reward = 0
    invalid_moves = 0
    invalid_move_amount = 10

    done = False
    moves = 0

    while not done:
        invalid_moves = 0
        invalid_move_amount = 10
        env.render()
        print("===========================================")
        
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten()
        
        if (reward != 0):
            invalid_moves = invalid_move_amount
        if (reward == 0):
            invalid_moves += 1
        
        if invalid_moves > invalid_move_amount:
            break

        
        state = next_state
        sleep(0.1)



# if __name__ == '__main__':
env = gym.make('2048-v0')


agent = DeepQ()


CONTINUE_TRAINING = False
DEMONSTRATION = False


ep_reward_list = []
avg_reward_list = []

if CONTINUE_TRAINING or DEMONSTRATION:
    agent.agent.load_weights(f"weights/deep_q.h5")
    agent.target_network.load_weights(f"weights/deep_q.h5")

if DEMONSTRATION:
    demonstrate(agent)
    exit()


for i in tqdm(range(10_000)):
    state = env.reset()
    state = state.flatten()
    # env.render()
    ep_reward = 0
    invalid_moves = 0
    invalid_move_amount = 10

    done = False
    moves = 0

    while not done:
        invalid_moves = 0
        invalid_move_amount = 10
        
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        next_state = next_state.flatten()
        


        if (reward != 0):
            invalid_moves = invalid_move_amount
        if (reward == 0):
            invalid_moves += 1
        
        if invalid_moves > invalid_move_amount:
            break


        if not DEMONSTRATION:
            agent.remember(state, action, reward, next_state, done)
            ep_reward += reward
            agent.train()
        
        state = next_state
    
    if i % 5 == 0:
        agent.target_network.save_weights(f"weights/deep_q.h5")
        agent.target_network.set_weights(agent.agent.get_weights())
        agent.agent.save_weights(f"weights/deep_q.h5")
    
    agent.epsilon *= agent.epsilon_decay


if not DEMONSTRATION:
    agent.agent.save_weights(f"weights/deep_q.h5")
    agent.target_network.save_weights(f"weights/deep_q.h5")

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.savefig("Avg_rewards")
    plt.show()




