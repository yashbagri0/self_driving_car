import pygame
import os
import torch

from car import Car
from track import Track
from DQN import DQNAgent

player_count = 2
is_ai_playing = True


def load_checkpoint(agent, filename):
    checkpoint = torch.load(filename, weights_only=False)
    agent.policy_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.memory = checkpoint['memory']
    return checkpoint['episode'], checkpoint['intersected_path']

WIDTH, HEIGHT = 1092, 774

pygame.init()
TITLE = 'Racetrack'
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()


path = 'assets'
path_to_bg = 'track.png'
collision_path_to_bg = 'track_collision.png'
track = Track(f'{path}/{path_to_bg}', f'{path}/{collision_path_to_bg}', (WIDTH, HEIGHT))
cars = [f'{path}/{car}' for car in os.listdir(path) if 'car' in car]

if not is_ai_playing:
    car2 = Car((460, 679), cars[1], (25, 40), velocity=10, turn_speed=5)
    car3 = Car((460, 711), cars[2], (25, 40), velocity=10, turn_speed=5)    
elif player_count == 2:
    car1 = Car((460, 679), cars[4], (25, 40), velocity=10, turn_speed=10)
    car2 = Car((460, 711), cars[1], (25, 40), velocity=10, turn_speed=5)
elif player_count == 3:
    car1 = Car((460, 679), cars[0], (25, 40), velocity=10, turn_speed=10)
    car2 = Car((460, 711), cars[1], (25, 40), velocity=10, turn_speed=5)
    car3 = Car((460, 695), cars[2], (25, 40), velocity=10, turn_speed=5)

state_size = 51
action_size = 5
running = True
if is_ai_playing:
    agent = DQNAgent(state_size, action_size, target_update=1000, epsilon=0, min_epsilon=0)
    load_checkpoint(agent, 'ckpt.pth')
ai_won = car2_won = car3_won = False
ai_animation = car2_animation = car3_animation = False
game_started = False
while running:
    clock.tick(60)
    if ai_animation:
        track.draw(screen, animation=ai_animation)
    elif car2_animation:
        track.draw(screen, animation=car2_animation, text_to_render='CAR 1 WON')
    elif car3_animation:
        track.draw(screen, animation=car3_animation, text_to_render='CAR 2 WON')
    else:
        track.draw(screen)
    if is_ai_playing and game_started:
        state = agent.get_states(car1, track)
        action = agent.act(state)
        
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

        prev_x, prev_y = car1.car_x, car1.car_y
        car1.move(keys, track)
        is_finished = track.intersect_reward_checkpoint(car1, prev_x, prev_y, car1.car_x, car1.car_y)
        if is_finished == 'finished':
            is_finished = ''
            if not (car2_won or car3_won):
                ai_won = True
                ai_animation = True
                ai_timer = pygame.time.get_ticks()
    if ai_animation and pygame.time.get_ticks() - ai_timer > 3000:
        ai_animation = False 


    keys_player = pygame.key.get_pressed()
    prev_x2, prev_y2 = car2.car_x, car2.car_y
    car2.move(keys_player, track, key='arrow')
    is_finished_c2 = track.intersect_reward_checkpoint(car2, prev_x2, prev_y2, car2.car_x, car2.car_y)

    if is_finished_c2 == 'finished':
        is_finished_c2 = ''
        if not (ai_won or car3_won):
            car2_won = True
            car2_animation = True
            car2_timer = pygame.time.get_ticks()
    if car2_animation and pygame.time.get_ticks() - car2_timer > 8000:
        car2_animation = False 
    if player_count == 3 or not is_ai_playing:
        keys_player = pygame.key.get_pressed()
        prev_x3, prev_y3 = car3.car_x, car3.car_y
        car3.move(keys_player, track, key='wad')
        is_finished_c3 = track.intersect_reward_checkpoint(car3, prev_x3, prev_y3, car3.car_x, car3.car_y)
        if is_finished_c3 == 'finished':
            is_finished_c3 = ''
            if not (car2_won or ai_won):
                car3_won = True
                car3_animation = True
                car3_timer = pygame.time.get_ticks()
        if car3_animation and pygame.time.get_ticks() - car3_timer > 8000:
            car3_animation = False 

    if any(keys_player):
        game_started = True
    

    # car1.raycasts(screen, track)
    if is_ai_playing:
        car1.draw(screen)
    car2.draw(screen)
    if player_count == 3 or not is_ai_playing:
        car3.draw(screen)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()