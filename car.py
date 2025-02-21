import pygame
import math

class Car:
    def __init__(self, car_x_y, car_image, car_width_height, velocity=0, max_velocity=5, car_angle=90, turn_speed=3):
        self.car_x, self.car_y = car_x_y
        self.car_width, self.car_height = car_width_height
        self.car_image = pygame.image.load(car_image)
        self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_height))
        self.mask = pygame.mask.from_surface(self.car_image)
        self.velocity = velocity
        self.max_velocity = max_velocity
        self.car_angle = car_angle
        self.turn_speed = turn_speed

        self.passed_reward_checkpoints_index = [] # store indexes 

        self.front_x = self.car_x + (self.car_height // 2) * math.cos(math.radians(self.car_angle+90))
        self.front_y = self.car_y - (self.car_height // 2) * math.sin(math.radians(self.car_angle+90))
            

    def draw(self, surface):
        # Rotate the car image
        rotated_car = pygame.transform.rotate(self.car_image, self.car_angle)
        # Get the new rect and position
        new_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        
        # Create mask from rotated image
        self.rotated_mask = pygame.mask.from_surface(rotated_car)        
        
        surface.blit(rotated_car, new_rect.topleft)
        # front_x = self.car_x + (self.car_height // 2) * math.cos(math.radians(self.car_angle+90))
        # front_y = self.car_y - (self.car_height // 2) * math.sin(math.radians(self.car_angle+90))

        # pygame.draw.circle(surface, (0, 0, 255), (self.front_x, self.front_y), 2)


    def move(self, keys, track, key=None):
        return_smth = 'no'
        old_x = self.car_x
        old_y = self.car_y
        # +90 cause we summon our car at 90 degress
        x_move = self.velocity * math.cos(math.radians(self.car_angle+90))
        y_move = self.velocity * math.sin(math.radians(self.car_angle+90))
        if key=='arrow':
            if keys[pygame.K_UP]:
                self.car_x += x_move
                self.car_y -= y_move
                # print('moved up')
            # elif keys[pygame.K_DOWN]:
            #     self.car_x -= x_move
            #     self.car_y += y_move
            if keys[pygame.K_LEFT]:
                self.car_angle += self.turn_speed
                # print('moved left')
            elif keys[pygame.K_RIGHT]:
                self.car_angle -= self.turn_speed
                # print('moved right')
        elif key == 'wad':
            if keys[pygame.K_w]:
                self.car_x += x_move
                self.car_y -= y_move
                # print('moved up')
            # elif keys[pygame.K_DOWN]:
            #     self.car_x -= x_move
            #     self.car_y += y_move
            if keys[pygame.K_a]:
                self.car_angle += self.turn_speed
                # print('moved left')
            elif keys[pygame.K_d]:
                self.car_angle -= self.turn_speed
        else:
            if 0 in keys:
                self.car_x += x_move
                self.car_y -= y_move
            if 1 in keys:
                self.car_angle += self.turn_speed
            elif 2 in keys:
                self.car_angle -= self.turn_speed

        self.front_x = self.car_x + (self.car_height // 2) * math.cos(math.radians(self.car_angle+90))
        self.front_y = self.car_y - (self.car_height // 2) * math.sin(math.radians(self.car_angle+90))
            
            
        if track.is_off_track(self.front_x, self.front_y):
            self.car_x = old_x
            self.car_y = old_y
            return_smth = 'is_off_track'
            
        self.car_angle = self.car_angle % 360 
        return return_smth

    def raycasts(self, screen, track):
        angles = [i for i in range(-180, 181, 15)]
        for ray_angle in angles:
            distance = self.get_distance_to_edge(track, ray_angle)
            # print(distance / math.sqrt(track.screen_size[0]**2 + track.screen_size[1]**2))
            end_x = self.car_x + distance * math.cos(math.radians(self.car_angle + ray_angle))
            end_y = self.car_y - distance * math.sin(math.radians(self.car_angle + ray_angle))
            pygame.draw.line(screen, (255, 0, 0), (self.car_x, self.car_y), (end_x, end_y), 2)  # Red laser
            pygame.draw.circle(screen, (255, 255, 0), (int(end_x), int(end_y)), 3)  # Small dot at hit point

    def get_distance_to_edge(self, track, relative_angle):
        ray_x = self.car_x
        ray_y = self.car_y
        angle = self.car_angle + relative_angle
        step_size = 1
        max_steps = 1000  # prevent infinite loops
        steps = 0
        
        while not track.is_off_track(ray_x, ray_y) and steps < max_steps:
            ray_x += step_size * math.cos(math.radians(angle))
            ray_y -= step_size * math.sin(math.radians(angle))
            steps += 1
        
        return math.sqrt((ray_x - self.car_x)**2 + (ray_y - self.car_y)**2)