import pygame

class Track:
    def __init__(self, display_image_path, collision_image_path, screen_size, render=True):
        self.render = render
        self.screen_size = screen_size
        if self.render:
            self.display_image = pygame.image.load(display_image_path)
            self.display_image = pygame.transform.scale(self.display_image, self.screen_size)
            
        self.collision_image = pygame.image.load(collision_image_path)
        self.collision_image = pygame.transform.scale(self.collision_image, self.screen_size)

        # self.passed_reward_checkpoints_index = [] # store indexes 

        self.reward_checkpoints = [
            # [(438, 652), (437, 743)], 
                                   [(345, 651), (344, 744)], [(211, 649), (205, 745)], [(166, 645), (106, 721)], [(134, 613), (44, 634)], [(136, 541), (44, 541)], [(154, 511), (92, 445)], [(172, 416), (190, 501)], [(280, 433), (227, 513)], [(328, 469), (292, 553)], [(388, 502), (390, 604)], [(426, 508), (448, 600)], [(444, 496), (533, 529)], [(534, 419), (452, 461)], [(476, 365), (422, 437)], [(351, 301), (346, 389)], [(253, 312), (282, 401)], [(162, 322), (138, 417)], [(140, 313), (62, 375)], [(34, 300), (135, 304)], [(136, 192), (44, 177)], [(76, 108), (169, 169)], [(276, 73), (272, 177)], [(488, 241), (429, 302)], [(604, 341), (552, 414)], [(692, 401), (735, 484)], [(705, 390), (789, 355)], [(693, 269), (642, 353)], [(612, 216), (550, 288)], [(510, 146), (451, 209)], [(437, 73), (516, 122)], [(536, 37), (542, 121)], [(664, 40), (644, 131)], [(753, 130), (668, 177)], [(811, 232), (757, 305)], [(827, 237), (868, 319)], [(824, 201), (913, 209)], [(832, 112), (916, 151)], [(899, 40), (930, 134)], [(1016, 61), (957, 136)], [(1052, 145), (956, 158)], [(956, 196), (1050, 189)], [(957, 340), (1052, 343)], [(952, 372), (988, 456)], [(839, 382), (876, 472)], [(867, 492), (779, 537)], [(880, 505), (875, 598)], [(996, 524), (951, 609)], [(1050, 647), (961, 629)], [(951, 641), (993, 735)], [(912, 649), (913, 744)], [(752, 652), (759, 745)], [(588, 653), (594, 745)]]
        self.finish_line = [(444, 651), (444, 745)]
    
    def get_height(self):
        return self.screen_size[1]

    def get_width(self):
        return self.screen_size[0]

    def draw(self, surface, animation=False, text_to_render='AI WON'):
        surface.blit(self.display_image, (0, 0))
        if animation:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            overlay.fill((50, 50, 50, 180))  # Dark translucent background
            surface.blit(overlay, (0, 0))  # Draw the overlay

            # Render the "You Won" text
            font = pygame.font.Font(None, 80)  # Default font
            text = font.render(text_to_render, True, (255, 255, 255))

            # Animate text (Sliding effect)
            if not hasattr(self, 'text_y'):
                self.text_y = surface.get_height()

            if self.text_y > surface.get_height() // 2 - 40:
                self.text_y -= 5  # Adjust speed

            # Center text
            text_x = (surface.get_width() - text.get_width()) // 2
            surface.blit(text, (text_x, self.text_y))

        # debugging
        # for reward_ptn in self.reward_checkpoints:
        #     start_x, start_y = reward_ptn[0]
        #     end_x, end_y = reward_ptn[1]
        #     pygame.draw.line(surface, (255, 0, 0), (start_x, start_y), (end_x, end_y), 2)  # Red line
        
        # start_x, start_y = self.finish_line[0]
        # end_x, end_y = self.finish_line[1]
        # pygame.draw.line(surface, (0, 0, 255), (start_x, start_y), (end_x, end_y), 2)
            

    def is_off_track(self, x, y):
        try:
            color = self.collision_image.get_at((int(x), int(y)))
            # print(color != (0, 0, 0, 255)) #check color, it takes the center x, y, and not the frontal. fix that 
            # try other network with residue
            # next try ckp 7
            # if color is not black (0,0,0), we're off track
            return color != (0, 0, 0)
        except IndexError:
            return True
        
    def do_intersect(self, A, B, C, D):
        """Returns True if line segment AB and CD intersect."""
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def intersect_reward_checkpoint(self, car, prev_x, prev_y, x, y):
        length_checkpoint = len(self.reward_checkpoints)
        if len(car.passed_reward_checkpoints_index) == length_checkpoint:
            if self.do_intersect((prev_x, prev_y), (x, y), self.finish_line[0], self.finish_line[1]):
                car.passed_reward_checkpoints_index = []
                # print(f'finished at {self.finish_line[0]} and {self.finish_line[1]}')
                return 'finished'
        for index,checkpoint in enumerate(self.reward_checkpoints):
            (x1, y1), (x2, y2) = checkpoint
            if self.do_intersect((prev_x, prev_y), (x, y), (x1, y1), (x2, y2)):
                # print(car.passed_reward_checkpoints_index)
                if index in car.passed_reward_checkpoints_index:
                    return 'reintersect'
                if index > 0 and (index - 1) not in car.passed_reward_checkpoints_index:
                    return 'wrong_order'
                car.passed_reward_checkpoints_index.append(index)
                # print(f'passed at ({x1}, {y1}) and ({x2}, {y2})')
                return 'intersect'
        return 'none'