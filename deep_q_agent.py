from collections import deque
import enum
import numpy as np
import random
from keras.layers import Dense, Input
from keras.models import Sequential 
from tensorflow.keras.optimizers import Adam

class DeepQ:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.replay_buffer = deque(maxlen=200_000)
        self.state_space = 16
        self.action_space = 4
        self.required_buffer_size = 32
        self.agent = self.create_model()
        self.target_network = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Input(self.state_space))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def act(self, state, demonstration=False):
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if demonstration:
            return np.argmax(self.agent.predict(state.reshape(1, 16)))

        else:
            if np.random.rand(1) < self.epsilon:
                return np.random.randint(0, 3)
            else:
                return np.argmax(self.agent.predict(state.reshape(1, 16)))
        
    def train(self):
        if len(self.replay_buffer) < self.required_buffer_size:
            return
        
        samples = random.sample(self.replay_buffer, self.required_buffer_size)

        states = []
        next_states = []

        # Look into refactoring this code to combine the two for loops. That isn't that great

        for sample in samples:
            state, action, reward, next_state, done = sample
            states.append(state)
            next_states.append(next_state)
        
        states = np.array(states)
        next_states = np.array(next_states)

        targets = np.array(self.agent.predict(states))
        next_state_targets = np.array(self.target_network.predict(next_states))

        for i, sample in enumerate(samples):
            state, action, reward, next_state, done = sample
            target = targets[i]

            if done:
                target[action] = reward
            else:
                q_future = max(next_state_targets[i])
                target[action] = reward + q_future * self.gamma
        
        self.agent.fit(states, targets, epochs=1, verbose=0)
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
