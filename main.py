import pygame
import os
import torch
import math

import matplotlib.pyplot as plt 

from car import Car
from track import Track
from DQN import DQNAgent


def save_checkpoint(agent, episode, intersected, filename):
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'memory': agent.memory,
        'intersected_path': intersected
    }
    torch.save(checkpoint, filename)

def load_checkpoint(agent, filename):
    checkpoint = torch.load(filename, weights_only=False)
    agent.policy_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.memory = checkpoint['memory']
    return checkpoint['episode'], checkpoint['intersected_path']


intersected = []
finished_times_list = []
finished_biggest_episodes_rewards = {}
def train_agent(agent, car, track, ep_start=0, episodes=100000, early_stop=10000, render=True, dir_num=1):
    for episode in range(ep_start, episodes):
        global running
        state = agent.get_states(car, track)
        total_reward = 0
        done = False
        steps = 0
        is_finished = False
        reintersected = 0
        finished_times = 0
        
        while not done:
            action = agent.act(state)
            
            # Convert action to game controls
            if render:
                keys = {pygame.K_UP: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
                if action == 0:  # up
                    keys[pygame.K_UP] = True
                elif action == 1:  # left
                    keys[pygame.K_LEFT] = True
                elif action == 2:  # right
                    keys[pygame.K_RIGHT] = True
                elif action == 3:  # up+left
                    keys[pygame.K_UP] = True
                    keys[pygame.K_LEFT] = True
                elif action == 4:  # up+right
                    keys[pygame.K_UP] = True
                    keys[pygame.K_RIGHT] = True
            else:
                keys = []
                if action == 0:  # up
                    keys.append(0)
                elif action == 1:  # left
                    keys.append(1)
                elif action ==2 :  # right
                    keys.append(2)
                elif action == 3:  # up+left
                    keys.append(0)
                    keys.append(1)
                elif action == 4:  # up+right
                    keys.append(0)
                    keys.append(2)
            
            # Take action
            old_pos = (car.car_x, car.car_y)
            return_smth = car.move(keys, track, render=render)
            reward = 0
            if return_smth == 'is_off_track':
                reward += -10

            # Calculate reward
            intersect = track.intersect_reward_checkpoint(old_pos[0], old_pos[1], car.car_x, car.car_y)
            # Reward for moving forward along track
            if intersect == 'finished':
                print('IT FINISHED')
                is_finished = True
                # finished.append(episode)
                reward += 60
            elif intersect == 'reintersect':
                reintersected += 1
                print('reintersected')
                if reintersected % 2 == 0:
                    reward += -5
                else: reward += -20
            elif intersect == 'wrong_order':
                print('wrong order')
                reward += -40
            elif intersect == 'intersect':
                # print('intersected')
                reward += 21
            elif intersect == 'none':
                reward += -0.25
            
            # Get next state
            next_state = agent.get_states(car, track)
            
            # Store transition
            agent.remember(state, action, reward, next_state, done)
            
            # Train
            if len(agent.memory) > agent.batch_size:
                agent.train()
            
            state = next_state
            total_reward += reward
            steps += 1
            # print(f'reards:{total_reward}')

            if render:
                screen.fill((255, 255, 255))
                track.draw(screen)
                car.draw(screen)
                pygame.display.flip()
                clock.tick(60)
            
            if total_reward < -250 or is_finished:
                print(f'stopped early {steps}')
                # early_stop = min(round(early_stop+1), 40000)
                done = True

                # increase_by = 200
                # every_step = 100
                # multiplier = (early_stop // every_step)
                # agent.target_update = min(1000 + (multiplier * increase_by), 3000)
            # if steps % 1000 == 0:
            #     print(f'steps: {steps}')

            if is_finished:
                finished_times += 1
                is_finished = False
                print(f'finished: {finished_times} times')
                intersected.append(100)
                print(f"Intersected: {100} %")
                track.passed_reward_checkpoints_index = []
                finished_biggest_episodes_rewards.update({episode+1: int(total_reward)})
        if not is_finished:
            print(f"Intersected: {len(track.passed_reward_checkpoints_index)*100/52} %")
            intersected.append(len(track.passed_reward_checkpoints_index)*100/52)


        reintersected = 0
        car.car_x = 460  # Starting x
        car.car_y = 697  # Starting y
        car.car_angle = 90
        car.front_x = car.car_x + (car.car_height / 2) * math.cos(math.radians(car.car_angle+90))
        car.front_y = car.car_y - (car.car_height / 2) * math.sin(math.radians(car.car_angle+90))
        track.passed_reward_checkpoints_index = []
        # finished_times_list.append(finished_times)

        if (episode+1) % 10 == 0:
            # print(finished)
            # finished = []
            
            plt.figure(figsize=(10, 5))
            plt.plot(intersected, marker='o', linestyle='-')
            plt.xlabel('Episode')
            plt.ylabel('Times passed finished line in one episode')
            plt.ylim(0, 100)
            plt.title('Progress Over Time')
            plt.grid(True)
            plt.savefig(f'imgs{dir_num}/intersected_{episode+1}.png')

            plt.close()
        save_checkpoint(agent, episode, intersected, f'ckp{dir_num}/checkpoint_{episode+1}.pth')
        save_checkpoint(agent, episode, intersected, f'ckp{dir_num}/checkpoint_latest.pth')
        print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}, with {finished_biggest_episodes_rewards}")

render = True

WIDTH, HEIGHT = 1092, 774
if render:
    pygame.init()
    TITLE = 'Racetrack'
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()


path = 'assets'
path_to_bg = 'track.png'
collision_path_to_bg = 'track_collision.png'
track = Track(f'{path}/{path_to_bg}', f'{path}/{collision_path_to_bg}', (WIDTH, HEIGHT), render=render)
cars = [f'{path}/{car}' for car in os.listdir(path) if 'car' in car]
# car = []
# for index, car_img in enumerate(cars):
#     if index == 0: # oput thids in main loop or smth
#         car_class = Car((460, 697), car_img, (25, 40), velocity=4)
#         car.append(car_class)
#     else: break
car = Car((460, 697), cars[0], (25, 40), velocity=10, turn_speed=10)

state_size = 51 # 6 car states + 25 raycasts + 20 poe (2+2)*5
action_size = 5  # [up, left, right, up+left, up+right]
running = True
# pos = []
finished = []
# print([i for i in range (-180, 180+1, 15)])
agent = DQNAgent(state_size, action_size, target_update=1000, min_epsilon=0.001)
if not render:
    episode, intersected = load_checkpoint(agent, 'ckp14/checkpoint_233.pth')            
    train_agent(agent, car, track, episodes=15000, render=render, dir_num=14, ep_start=224)
else:
    agent = DQNAgent(state_size, action_size, target_update=1000, epsilon=0, min_epsilon=0)
    episode, _ = load_checkpoint(agent, 'ckp14/checkpoint_233.pth')
    # episode, _ = load_checkpoint(agent, 'ckp14/checkpoint_194.pth')
    # episode, _ = load_checkpoint(agent, 'ckp14/checkpoint_122.pth')

    # print(_)
test = 0
while render and running:
    clock.tick(60)
    track.draw(screen)
    state = agent.get_states(car, track)
    # print(state)
    action = agent.act(state)
    # old_pos = (car.car_x, car.car_y)
    # track.intersect_reward_checkpoint(old_pos[0], old_pos[1], car.car_x, car.car_y)
        
    if render:
        keys = {pygame.K_UP: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
        # print(action)

        if action == 0:  # up
            keys[pygame.K_UP] = True
        elif action == 1:  # left
            keys[pygame.K_LEFT] = True
        elif action == 2:  # right
            keys[pygame.K_RIGHT] = True
        elif action == 3:  # up+left
            keys[pygame.K_UP] = True
            keys[pygame.K_LEFT] = True
        elif action == 4:  # up+right
            keys[pygame.K_UP] = True
            keys[pygame.K_RIGHT] = True

    # keys = pygame.key.get_pressed()
    prev_x, prev_y = car.car_x, car.car_y
    smth = car.move(keys, track)
    if smth == 'is_off_track':
        test +=1
        print(smth)
        print(test)
    _ = track.intersect_reward_checkpoint(prev_x, prev_y, car.car_x, car.car_y)
    car.draw(screen)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            x, y = pygame.mouse.get_pos()
            print(f'({x}, {y})')
if render:
    pygame.quit()