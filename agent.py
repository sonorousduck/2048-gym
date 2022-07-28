from time import time
from matplotlib import pyplot as plt
from tensorflow.keras.models import clone_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from modules.OUActionNoise import OUActionNoise
import asyncio
from tqdm import tqdm
from modules.Buffer import Buffer

class Agent:
    def __init__(self, ):
       
        # TODO: Determine size of state space
        self.state_space = 16
        self.action_space = 4
        
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = 0.99
        self.batch_size = 128

        self.tau = 0.005
        self.epochs = 10_000
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        self.critic_learning_rate = 0.002
        self.actor_learning_rate = 0.001
        self.learns = 0
        self.actor_model = self.create_actor()
        self.critic_model = self.create_critic()
        self.actor_target_model = clone_model(self.actor_model)
        self.critic_target_model = clone_model(self.critic_model)
        self.rewards = []
        self.memory = Buffer(self.state_space, self.action_space, batch_size=self.batch_size)

        self.train_time_in_seconds = 60

        self.critic_optimizer = Adam(self.critic_learning_rate)
        self.actor_optimizer = Adam(self.actor_learning_rate)


    def create_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = Input(shape=(self.state_space,))
        out = Dense(256, activation="relu")(inputs)
        out = Dense(256, activation="relu")(out)

        outputs = Dense(self.action_space, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs

        model = Model(inputs, outputs)
        return model

    
    def create_critic(self):
        state_input = Input(shape=(self.state_space,))
        state_out = Dense(16, activation="relu")(state_input)
        state_out = Dense(32, activation="relu")(state_out)

        action_input = Input(shape=(self.action_space,))
        action_out = Dense(32, activation='relu')(action_input)

        concat_layer = Concatenate()([state_out, action_out])

        out = Dense(256, activation="relu")(concat_layer)
        out = Dense(512, activation="relu")(out)
        out = Dense(1024, activation="relu")(out)
        outputs = Dense(1)(out)

        model = Model([state_input, action_input], outputs)
        return model


    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.actor_target_model(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.critic_target_model([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
    
    def learn(self):

        # Get Sampling Range
        record_range = min(self.memory.buffer_counter, self.memory.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to Tensors
        state_batch = tf.convert_to_tensor(self.memory.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.memory.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.memory.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        next_state_batch = tf.convert_to_tensor(self.memory.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    @tf.function
    def update_target(target_weights, weights, tau):
        # LERP
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
    
    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()

        sampled_actions = sampled_actions.numpy() + noise

        legal_action = np.clip(sampled_actions, -1.0, 1.0)

        return np.squeeze(legal_action)

    



    # def train(self, camera_feed, collision_feed, danger_feed, lidar_feed, location_feed, messages):
    #
    #     self.camera_feed = camera_feed
    #     self.collision_feed = collision_feed
    #     self.danger_feed = danger_feed
    #     self.lidar_feed = lidar_feed
    #     self.location_feed = location_feed
    #     self.messages = messages
    #
    #     # while True:
    #     #     test = self.get_state()
    #     #     print(test)
    #
    #
    #
    #
    #     for i in tqdm(range(self.epochs)):
    #         done = self.get_done()
    #
    #         start_time = time()
    #
    #
    #         prev_state = self.get_state()
    #         episodic_reward = 0
    #
    #         while True:
    #
    #             tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
    #             action = self.policy(tf_prev_state, self.ou_noise)
    #
    #             self.messages.put(action)
    #
    #             state = self.get_state()
    #             reward = self.get_reward()
    #             done = self.get_done()
    #
    #             current_time = time() - start_time
    #
    #             if current_time / 1000 > self.train_time_in_seconds:
    #                 done = True
    #
    #
    #
    #
    #             if not DEMONSTRATION:
    #                 self.memory.record(())
    #                 episodic_reward += reward
    #
    #                 self.learn()
    #                 self.update_target(self.actor_target_model.variables, self.actor_model.variables, self.tau)
    #                 self.update_target(self.critic_target_model.variables, self.critic_model.variables, self.tau)
    #
    #             if done:
    #                 break
    #
    #             prev_state = state
    #
    #         ep_reward_list.append(episodic_reward)
    #         avg_reward = np.mean(ep_reward_list[-40:])
    #         print(f"Episode * {i} * Avg Reward is ==> {avg_reward}")
    #         avg_reward_list.append(avg_reward)
    #
    #     if not DEMONSTRATION:
    #         self.actor_model.save_weights(f"weights/stealth_actor.h5")
    #         self.critic_model.save_weights(f"weights/stealth_critic.h5")
    #
    #         self.actor_target_model.save_weights(f"weights/stealth_target_actor.h5")
    #         self.critic_target_model.save_weights(f"weights/stealth_target_critic.h5")
    #
    #         plt.plot(avg_reward_list)
    #         plt.xlabel("Episode")
    #         plt.ylabel("Avg. Episodic Reward")
    #         plt.show()

            















