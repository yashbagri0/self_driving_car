import torch
import torch.nn as nn 
import torch.optim as optim
from collections import deque 
import numpy as np
import math
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.GroupNorm(1, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.GroupNorm(1, 128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.GroupNorm(1, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.GroupNorm(1, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.GroupNorm(1, 64),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)
        # super().__init__()

        # self.fc1 = nn.Linear(state_size, 128)
        # self.norm1 = nn.GroupNorm(1, 128)
        
        # self.fc2 = nn.Linear(128, 256)
        # self.norm2 = nn.GroupNorm(1, 256)
        
        # self.fc3 = nn.Linear(256, 512)
        # self.norm3 = nn.GroupNorm(1, 512)

        # self.proj1 = nn.Linear(128, 512)  # Projection layer to match x1 size with x2
        
        # self.fc4 = nn.Linear(512, 256)
        # self.norm4 = nn.GroupNorm(1, 256)

        # self.fc5 = nn.Linear(256, 128)
        # self.norm5 = nn.GroupNorm(1, 128)
        
        # self.proj2 = nn.Linear(512, 128)  # Projection layer for second skip connection

        # self.fc_out = nn.Linear(128, action_size)

        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()

    # def forward(self, x):
    #     # Input layer
    #     x1 = self.relu(self.norm1(self.fc1(x)))
        
    #     # Residual connection 1
    #     x2 = self.relu(self.norm2(self.fc2(x1)))
    #     x2 = self.relu(self.norm3(self.fc3(x2)))
    #     x2 = x2 + self.proj1(x1)  # Skip connection
        
    #     # Residual connection 2
    #     x3 = self.relu(self.norm4(self.fc4(x2)))
    #     x3 = self.relu(self.norm5(self.fc5(x3)))
    #     x3 = x3 + self.proj2(x2)  # Skip connection
        
    #     # Output layer
    #     out = self.fc_out(x3)
    #     return out

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, deque_max_len=10000, discount=0.99, epsilon=1.0, min_epsilon=0.0001,
                 epsilon_decay=0.995, target_update=5000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_network = DQN(self.state_size, self.action_size).to(self.device)
        self.target_network = DQN(self.state_size, self.action_size).to(self.device)

        self.lr = lr
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.memory = deque(maxlen=deque_max_len)

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size

        self.train_step = 0

    def positional_encoding(self, index, total_checkpoints):
        pos = index / total_checkpoints  # Normalize index (0 to 1)
        enc = [np.sin(2 * np.pi * pos), np.cos(2 * np.pi * pos)]
        return enc

    def get_states(self, car, track):
        state = []
        state.append(car.car_x/track.screen_size[0])
        state.append(car.car_y/track.screen_size[1])
        state.append(car.front_x/track.screen_size[0])
        state.append(car.front_y/track.screen_size[1])

        # state.append(1) #velocity to be 1
        state.append(np.sin(np.radians(car.car_angle+90)))
        state.append(np.cos(np.radians(car.car_angle+90)))

        num_next_checkpoints = 5

        unpassed_checkpoints = [i for i in range(len(track.reward_checkpoints)) if i not in car.passed_reward_checkpoints_index]
        # print(unpassed_checkpoints)
        # print(car.passed_reward_checkpoints_index)
        while len(unpassed_checkpoints) < num_next_checkpoints:
            unpassed_checkpoints.append(-1)
        # print(unpassed_checkpoints[0])
        # print(track.reward_checkpoints)
        

        for i in range(num_next_checkpoints):
            enc = self.positional_encoding(unpassed_checkpoints[i], len(track.reward_checkpoints))
            chk_x, chk_y = np.mean(track.reward_checkpoints[i], axis=0)  # Get checkpoint center
            chk_x /= track.screen_size[0]  # Normalize X
            chk_y /= track.screen_size[1]  # Normalize Y
            
            state.extend(enc)  # Add positional encoding
            state.extend([chk_x, chk_y])  # Add coordinates

        # angles = [-180, -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15, 
        #           0,
        #           15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        # angles = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        
        angles = [i for i in range (-180, 180+1, 15)] #25
        for ray_angle in angles:
            distance = self.get_distance_to_edge(car, track, ray_angle)
            state.append(distance / np.sqrt(track.screen_size[0]**2 + track.screen_size[1]**2))
        
        return np.array(state, dtype=np.float32)
    
    def get_distance_to_edge(self, car, track, relative_angle):
        ray_x = car.car_x
        ray_y = car.car_y
        angle = car.car_angle + relative_angle
        step_size = 1
        max_steps = 1000  # prevent infinite loops
        steps = 0
        
        while not track.is_off_track(ray_x, ray_y) and steps < max_steps:
            ray_x += step_size * math.cos(math.radians(angle))
            ray_y -= step_size * math.sin(math.radians(angle))
            steps += 1
        
        return np.sqrt((ray_x - car.car_x)**2 + (ray_y - car.car_y)**2)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.policy_network(state)
            return torch.argmax(action_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)

        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update == 0:
            # print(self.target_update)
            # print('Changed target network')
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    # step 1: make the states and nn, and output is actions, henceforth called policy network, pn
    # step 2: make a copy of nn, target network, tn
    # step 3: normal navigation(play game)
    # step 3a: memory use deque
    # step 3b: memory purge after a set number 
    # step 4: use our pn
    # step 5: use tn
    # step 6: calc q value using: q[state, action] = reward if new_state is terminal else eward + discount*max(q[state,:])
    # step 7: get action in tn from output and change to output of q
    # step 8: use the target value action tn in output to train policy network 
    # step 9: repeat 3-8
    # step 10: sync policy and tn (that means we need to run both of them)
    # step 11: repeat 9-10