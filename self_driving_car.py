import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from collections import deque
import sys


SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE = 10

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Car:
    def __init__(self, device):
        self.device = device
        self.surface = pygame.image.load("car.png")
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = [650, 650]
        self.angle = 0
        self.speed = 10
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.is_alive = True
        self.distance = 0
        self.time_spent = 0

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, map):
        self.is_alive = True
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not map.get_at((x, y)) == (255, 255, 255, 255) and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map):
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        
        
        self.distance += self.speed
        self.time_spent += 1

        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > SCREEN_WIDTH - 120:
            self.pos[0] = SCREEN_WIDTH - 120

        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > SCREEN_HEIGHT - 120:
            self.pos[1] = SCREEN_HEIGHT - 120

        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len,
                   self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len,
                      self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map)

    def get_state(self):
        radars = self.radars
        state = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            state[i] = int(r[1] / 30)
        return torch.FloatTensor(state).to(self.device)

    def get_reward(self):
        reward = self.speed
        if not self.is_alive:
            reward = -100
        return torch.tensor([reward / 50.0], device=self.device)

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

class CarAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 5
        self.output_size = 2
        
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        self.epsilon = EPSILON_START

    def select_action(self, state):
        sample = random.random()
        self.epsilon = max(EPSILON_END, EPSILON_DECAY * self.epsilon)
        
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(0)[1].view(1, 1).to(self.device)
        else:
            return torch.tensor([[random.randrange(2)]], dtype=torch.long, device=self.device)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.stack(batch[0]).to(self.device)
        action_batch = torch.cat(batch[1]).to(self.device)
        reward_batch = torch.cat(batch[2]).to(self.device)
        next_state_batch = torch.stack(batch[3]).to(self.device)
        done_batch = torch.cat(batch[4]).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        next_state_values[done_batch] = 0
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

def train():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    game_map = pygame.image.load('map.png')
    
    
    car_ai = CarAI()
    episode_rewards = []
    
    for episode in range(1000):
        car = Car(car_ai.device)  
        episode_reward = 0
        step = 0
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = car.get_state()
            action = car_ai.select_action(state)
            
            if action.item() == 0:
                car.angle += 15
            else:
                car.angle -= 15


            car.update(game_map)
            
            reward = car.get_reward()
            done = torch.tensor([not car.is_alive], dtype=torch.bool, device=car_ai.device)
            next_state = car.get_state()
            
            car_ai.memory.push(state, action, reward, next_state, done)
            car_ai.optimize_model()
            
            episode_reward += reward.item()
            
            screen.blit(game_map, (0, 0))
            car.draw(screen)
            
            font = pygame.font.SysFont("Arial", 30)
            text = font.render(f"Episode: {episode} Step: {step} Reward: {episode_reward:.2f}", 
                             True, (255, 255, 0))
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(30)
            
            step += 1
            
            if done.item():
                episode_rewards.append(episode_reward)
                if episode % TARGET_UPDATE == 0:
                    car_ai.target_net.load_state_dict(car_ai.policy_net.state_dict())
                break

        if episode % 100 == 0:
            torch.save(car_ai.policy_net.state_dict(), f'car_model_episode_{episode}.pth')

if __name__ == "__main__":
    train()