# scratch space for testing out dspy automation ideas

import dspy
import openai
import re
import numpy as np
import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import logging
import time


turbo = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.2)
dspy.settings.configure(lm=turbo)
openai.api_key_path = '../../openai_secret_key.txt'

class TankCaptain(dspy.Signature):
    """Gives direction and inspiration to the Tank Pilot."""

    directive = dspy.InputField()
    order = dspy.OutputField(desc="Short speech to the tank pilot communicating the captain's intent and setting the stage for the ensuing combat.")

class TankPilot(dspy.Signature):
    """Takes direction from the Tank Captain and chooses actions to pilot the tank."""

    intent_and_status = dspy.InputField()
    action = dspy.OutputField(desc="Choice of one action out of the limited available actions.")

blue_captain = dspy.Predict(TankCaptain)
blue_order = blue_captain(directive='You are the captain giving orders over radio to a tank asset that is about ' + \
    'to engage in combat with an adversary tank. The battlefield is a flat plain encircled by an impassable river. ' + \
    'You know the other tank is at the other end of the battlefield but not precisely where it is. Your tank pilot cannot see it because of fog. ' + \
    'Somewhere near the middle, there are small groves of trees with thick underbrush where a tank would be able to hide. ' + \
    'You know that both tank models can only see a limited view to their front, and the turrets cannot swivel. ' + \
    'The tanks are able to turn, move forward or backwards, and fire. ' + \
    'You are one of the good guys. You value honor, chivalry, and want to give the adversary a fair fight and ultimately a noble death. ' + \
    'Your task now is to come up with a short speech that will inspire the tank pilot to follow your ideals and achieve victory.' + \
    'Please use virtuous, flowery, and militaristic language in your speech, and limit it to 5 sentences or less.')

red_captain = dspy.Predict(TankCaptain)
red_order = red_captain(directive='You are the captain giving orders over radio to a tank asset that is about ' + \
    'to engage in combat with an adversary tank. The battlefield is a flat plain encircled by an impassable river. ' + \
    'You know the other tank is at the other end of the battlefield but not precisely where it is. Your tank pilot cannot see it because of fog. ' + \
    'Somewhere near the middle, there are small groves of trees with thick underbrush where a tank would be able to hide. ' + \
    'You know that both tank models can only see a limited view to their front, and the turrets cannot swivel. ' + \
    'The tanks are able to turn, move forward or backwards, and fire. ' + \
    'You are one of the bad guys, though you might not see yourself that way. You value cunning, dirty tactics, and want to destroy your opponent at any cost. ' + \
    'Your task now is to come up with a short speech that will force the lazy, no-good tank pilot to follow your ideals and achieve victory.' + \
    'Please use harsh, derogatory, and militaristic language in your speech, and limit it to 5 sentences or less.')

# %% simulation loop

class Tank:
    def __init__(self, actor='blue') -> None:
        self.pilot = dspy.Predict(TankPilot)
        if actor == 'blue':
            # tank starting location
            self.loc_xy = [0, 490]
            # initial heading (0 is North or +y, 90 East or +x)
            self.heading = 180
            # vision radius in m
            self.viewing_radius = 500
            # vision half width in degrees
            self.viewing_hwidth = 90
        else:
            # tank starting location
            self.loc_xy = [0, -490]
            # initial heading (0 is North or +y, 90 East or +x)
            self.heading = 0
            # vision radius in m
            self.viewing_radius = 500
            # vision half width in degrees
            self.viewing_hwidth = 90
        self.hidden = False # start unhidden
        
    def update_status(self, action='move forward 50 m') -> tuple([bool, bool]):
        # parse the action and update status
        test_num = re.findall(r'\d+',action)
        if test_num != []:
            num = float(test_num[0]) # only use first number provided
        else:
            # apply default value if no number provided
            num = 50
            
        fired = False
        parsed = True
        if 'turn' in action.lower() and 'left' in action.lower():
            self.heading = self.heading - np.min([num, 45])
        elif 'turn' in action.lower() and 'right' in action.lower():
            self.heading = self.heading + np.min([num, 45])
        elif 'move' in action.lower() and 'forward' in action.lower():
            self.loc_xy[1] = self.loc_xy[1] + np.min([num, 50])*np.cos(np.deg2rad(self.heading))
            self.loc_xy[0] = self.loc_xy[0] + np.min([num, 50])*np.sin(np.deg2rad(self.heading))
        elif 'move' in action.lower() and 'back' in action.lower():
            self.loc_xy[1] = self.loc_xy[1] + np.min([num, 50])*np.cos(np.deg2rad(self.heading+180))
            self.loc_xy[0] = self.loc_xy[0] + np.min([num, 50])*np.sin(np.deg2rad(self.heading+180))
        elif 'fire' in action.lower():
            fired = True
        else:
            parsed = False
            
        return fired, parsed

class SimulationBoard:
    def __init__(self, blue_order, red_order, temperature=0.2) -> None:
        self.temperature = temperature
        turbo = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.2) # apply temperature to LLMs
        dspy.settings.configure(lm=turbo)
        self.step_num = 0 # initialize turn timer (+1 per turn either blue or red take)
        # grab timestamp for writing to folder
        ts = datetime.datetime.now()
        ts_format = str(datetime.datetime.now()).replace('-','_').replace(' ','T').replace(':','').split('.')[0]
        self.save_folder = 'sim_temperature_{}_'.format(self.temperature) + ts_format
        os.makedirs(self.save_folder)
        self.logging_file = os.path.join(self.save_folder,'log.txt')
        logging.basicConfig(filename=self.logging_file, encoding='utf-8', level=logging.DEBUG, force=True)
        # define limits of rectangular board [-x, x, -y, y] in meters
        self.board_limits = [-500, 500, -500, 500]
        # define random location of grove(s) of trees [x, y]
        self.n_grove = 6
        self.grove_xy = []
        self.grove_r = []
        for ii in range(self.n_grove):
            # choose random location, assuming the board is centered on the origin
            self.grove_xy.append([np.random.rand()*1.6*self.board_limits[1]-0.8*self.board_limits[1], 
                                  np.random.rand()*1.6*self.board_limits[3]-0.8*self.board_limits[3]])
            # choose random grove size (radius)
            self.grove_r.append(np.random.rand()*50 + 50)
        # self.grove_xy = [[-200, -200], [100, -350], [0, 0], [150, 150]]
        # # define grove radii
        # self.grove_r = [50, 50, 50, 50]
        # initialize red and blue tanks
        self.blue_tank = Tank('blue')
        self.red_tank = Tank('red')
        # save initial battleground image
        self.write_board_image()
        # initialize red and blue prompt strings
        pilot_init_1 = 'You are the pilot of a tank that is about ' + \
            'to engage in combat with an adversary tank. The battlefield is a flat plain encircled by an impassable river. ' + \
            'You know the other tank is at the other end of the battlefield but cannot see where it is because of fog. ' + \
            'Somewhere near the middle, there are small groves of trees with thick underbrush where a tank would be able to hide. ' + \
            'You hear your Captain''s voice coming through the radio: \n'
        pilot_init_2 = '...\n\nYour Captain''s voice fades into static.\n' + \
            'Your must consider your Captain''s orders and take one of the following possible actions. ' + \
            'You can turn left or right up to 45 degrees, move forward or backward up to 50 m, or fire your turret.\n'
        self.blue_prompt = pilot_init_1 + blue_order.values()[0] + pilot_init_2 + self.get_observation(self.blue_tank, self.red_tank)
        self.red_prompt = pilot_init_1 + red_order.values()[0] + pilot_init_2 + self.get_observation(self.red_tank, self.blue_tank)
    
    def update_board(self, actor='blue', action='move forward 50 m') -> str:
        
        game_end = 'Continue' # by default, do not end the game
        # update tank status
        if actor == 'blue':
            fired, parsed = self.blue_tank.update_status(action)
            # append taken action to the prompt string
            self.blue_prompt += '\n' + action + '\n'
            # test victory conditions
            if fired:
                result = self.check_fire_hit(self.blue_tank, self.red_tank)
                self.blue_prompt +=  result + '\n'
                if 'You win!' in result:
                    game_end = 'Blue victory!'
            result = self.check_board_limits(self.blue_tank)
            if 'sink in the murky depths' in result:
                self.blue_prompt += result + '\n'
                game_end = 'Red victory!'
            if 'victory' not in game_end:
                # test for mis-parsed input
                if not parsed:
                    self.blue_prompt +=  'Sorry, that action was not understood. Please choose from the list of possible actions above.\n'
                # get observation of the game state
                obs = self.get_observation(self.blue_tank, self.red_tank)
                if 'You are currently hidden' in obs:
                    self.blue_tank.hidden = True
                self.blue_prompt += obs
        else:
            fired, parsed = self.red_tank.update_status(action)
            # append taken action to the prompt string
            self.red_prompt += '\n' + action + '\n'
            # test victory conditions
            if fired:
                result = self.check_fire_hit(self.red_tank, self.blue_tank)
                self.red_prompt +=  result + '\n'
                if 'You win!' in result:
                    game_end = 'Red victory!'
            result = self.check_board_limits(self.red_tank)
            if 'sink in the murky depths' in result:
                self.blue_prompt += result + '\n'
                game_end = 'Blue victory!'
            if 'victory' not in game_end:
                # test for mis-parsed input
                if not parsed:
                    self.red_prompt +=  'Sorry, that action was not understood. Please choose from the list of possible actions above.\n'
                # get observation of the game state
                obs = self.get_observation(self.red_tank, self.blue_tank)
                if 'You are currently hidden' in obs:
                    self.red_tank.hidden = True
                self.red_prompt += obs
        
        # write the board state to image
        self.step_num += 1
        self.write_board_image(actor, fired, parsed)
        
        return game_end
        
    def get_observation(self, tank, enemy_tank) -> str:
        # build observations encoded into a message
        obs = '\nHere is the current battlefield status. '
        # test if the gameboard edge is nearby (doesn't need to be within view)
        if np.abs(tank.loc_xy[1] - self.board_limits[2]) <= tank.viewing_radius-10:
            obs += 'There is an impassible river {} m away, {}. '.format(
                int(np.abs(tank.loc_xy[1] - self.board_limits[2])), self.get_dir_str(180 - tank.heading))
        if np.abs(tank.loc_xy[1] - self.board_limits[3]) <= tank.viewing_radius-10:
            obs += 'There is an impassible river {} m away, {}. '.format(
                int(np.abs(tank.loc_xy[1] - self.board_limits[3])), self.get_dir_str(-tank.heading))
        if np.abs(tank.loc_xy[0] - self.board_limits[0]) <= tank.viewing_radius-10:
            obs += 'There is an impassible river {} m away, {}. '.format(
                int(np.abs(tank.loc_xy[0] - self.board_limits[0])), self.get_dir_str(-tank.heading - 90))
        if np.abs(tank.loc_xy[0] - self.board_limits[1]) <= tank.viewing_radius-10:
            obs += 'There is an impassible river {} m away, {}. '.format(
                int(np.abs(tank.loc_xy[0] - self.board_limits[1])), self.get_dir_str(-tank.heading + 90))
        # test if other tank is within view
        dist = np.sqrt(np.square(np.abs(tank.loc_xy[0] - enemy_tank.loc_xy[0])) + 
                       np.square(np.abs(tank.loc_xy[1] - enemy_tank.loc_xy[1])))
        angle = 90 - np.rad2deg(np.arctan2(enemy_tank.loc_xy[1] - tank.loc_xy[1], enemy_tank.loc_xy[0] - tank.loc_xy[0])) # converted to 0=N, 90=E
        angle_diff = np.abs(tank.heading - angle)
        if dist <= tank.viewing_radius and angle_diff <= tank.viewing_hwidth and not enemy_tank.hidden:
            obs += 'You see the enemy tank {} m away, {}! '.format(
                int(dist), self.get_dir_str(angle - tank.heading))
        else:
            obs += 'You don''t see the enemy tank. '
        # test if any groves of trees are in view
        for ii, grove_xy in enumerate(self.grove_xy):
            dist = np.sqrt(np.square(np.abs(tank.loc_xy[0] - grove_xy[0])) + 
                        np.square(np.abs(tank.loc_xy[1] - grove_xy[1])))
            angle = 90 - np.rad2deg(np.arctan2(grove_xy[1] - tank.loc_xy[1], grove_xy[0] - tank.loc_xy[0])) # converted to 0=N, 90=E
            angle_diff = np.abs(tank.heading - angle)
            if dist - self.grove_r[ii] <= tank.viewing_radius and angle_diff <= tank.viewing_hwidth:
                if dist <= self.grove_r[ii]:
                    obs += 'You are currently hidden inside a dense grove of trees. '
                else:
                    obs += 'You see a dense grove of trees {} m away, {}. '.format(
                        int(dist), self.get_dir_str(angle - tank.heading))
        obs += 'You see nothing else in the fog.\nPlease take one of the above possible actions now.\n'
        
        return obs
    
    def get_dir_str(self, direction) -> str:
        dir = np.remainder(direction, 360)
        if dir > 358 or dir < 2:
            dir_str = 'in front of you'
        elif dir > 182:
            dir_str = 'at {} degrees to your left'.format(int(360-dir))
        elif dir > 178:
            dir_str = 'behind you'
        elif dir >= 2:
            dir_str = 'at {} degrees to your right'.format(int(dir))
        return dir_str
    
    def check_fire_hit(self, tank, enemy_tank) -> str:
        # compute distance to enemy tank and difference between heading and vector to the enemy tank
        dist = np.sqrt(np.square(np.abs(tank.loc_xy[0] - enemy_tank.loc_xy[0])) + 
                       np.square(np.abs(tank.loc_xy[1] - enemy_tank.loc_xy[1])))
        angle = 90 - np.rad2deg(np.arctan2(enemy_tank.loc_xy[1] - tank.loc_xy[1], enemy_tank.loc_xy[0] - tank.loc_xy[0])) # converted to 0=N, 90=E
        angle_diff = np.abs(tank.heading - angle)
        
        # test victory condition - must be within 10 degrees at 100 m distance, and within 400 m distance total
        if dist * angle_diff <= 1000.0 and dist <= 400.0:
            result = 'Your shot strikes the enemy tank! You win!'
        else:
            if dist * angle_diff <= 1000.0 and dist > 400.0:
                result = 'Your shot whizzes towards the enemy tank, but falls short and hits the ground. You need to get closer.'
            else:
                result = 'Your shot whizzes through the air, missing the enemy tank.'
        
        return result
    
    def check_board_limits(self, tank) -> str:
        if (tank.loc_xy[0] < self.board_limits[0] or 
            tank.loc_xy[0] > self.board_limits[1] or
            tank.loc_xy[1] < self.board_limits[2] or
            tank.loc_xy[1] > self.board_limits[3]):
            result = 'You have crossed into the river and sink in the murky depths. You lose!'
        else:
            result = ''
        return result
        
    def write_board_image(self, actor='blue', fired=False, parsed=True) -> None:
        fig = plt.figure()
        ax = plt.gca()
        # draw groves
        for ii, grove_xy in enumerate(self.grove_xy):
            circle = plt.Circle(grove_xy, self.grove_r[ii], color='g')
            ax.add_patch(circle)
        # draw tanks and indicate heading
        plt.plot(*self.blue_tank.loc_xy, 'bs', markersize=8)
        plt.plot(self.blue_tank.loc_xy[0] + np.array([0, 100*np.sin(np.deg2rad(self.blue_tank.heading))]), 
                 self.blue_tank.loc_xy[1] + np.array([0, 100*np.cos(np.deg2rad(self.blue_tank.heading))]), 'b-', linewidth=3)
        plt.plot(*self.red_tank.loc_xy, 'rs', markersize=8)
        plt.plot(self.red_tank.loc_xy[0] + np.array([0, 100*np.sin(np.deg2rad(self.red_tank.heading))]), 
                 self.red_tank.loc_xy[1] + np.array([0, 100*np.cos(np.deg2rad(self.red_tank.heading))]), 'r-', linewidth=3)
        plt.grid()
        # draw viewable area per tank
        arc = patches.Arc(self.blue_tank.loc_xy, self.blue_tank.viewing_radius*2, self.blue_tank.viewing_radius*2,
                      angle=90-self.blue_tank.heading,
                      theta1=-self.blue_tank.viewing_hwidth,
                      theta2=self.blue_tank.viewing_hwidth,
                      color='b',
                      linewidth=1)
        ax.add_patch(arc)
        arc = patches.Arc(self.red_tank.loc_xy, self.red_tank.viewing_radius*2, self.red_tank.viewing_radius*2, 
                      angle=90-self.red_tank.heading,
                      theta1=-self.red_tank.viewing_hwidth,
                      theta2=self.red_tank.viewing_hwidth,
                      color='r', 
                      linewidth=1)
        ax.add_patch(arc)
        # write symbols for if a tank fired or did not properly give an action
        if actor == 'blue':
            if fired:
                plt.plot(self.blue_tank.loc_xy[0] + np.array([0, 400*np.sin(np.deg2rad(self.blue_tank.heading))]), 
                         self.blue_tank.loc_xy[1] + np.array([0, 400*np.cos(np.deg2rad(self.blue_tank.heading))]), 'b--', linewidth=3)
            if not parsed:
                plt.plot(*self.blue_tank.loc_xy, 'kx', markersize=8)
        else:
            if fired:
                plt.plot(self.red_tank.loc_xy[0] + np.array([0, 400*np.sin(np.deg2rad(self.red_tank.heading))]), 
                         self.red_tank.loc_xy[1] + np.array([0, 400*np.cos(np.deg2rad(self.red_tank.heading))]), 'r--', linewidth=3)
            if not parsed:
                plt.plot(*self.red_tank.loc_xy, 'kx', markersize=8)
        ax.set_aspect('equal', 'box')
        plt.xlim(self.board_limits[:2])
        plt.ylim(self.board_limits[2:])
        plt.savefig(os.path.join(self.save_folder,'board_step_{}.png'.format(self.step_num)),
                    dpi=300)
        plt.close()
        return
    
    def play_game(self, n_turns=5) -> None:
        print('\nGame Settings: n_turns = {}, temperature = {}\n\n'.format(n_turns, self.temperature))
        logging.info('\nGame Settings: n_turns = {}, temperature = {}\n\n'.format(n_turns, self.temperature))
        for ii in range(n_turns):
            print('Starting turn {} of {}...'.format(ii+1, n_turns))
            logging.info('Starting turn {} of {}...'.format(ii+1, n_turns))
            blue_action = self.blue_tank.pilot(intent_and_status = self.blue_prompt)
            game_end = self.update_board('blue', blue_action.values()[0][0:np.min([100,len(blue_action.values()[0])])])
            if 'victory' in game_end:
                self.blue_prompt += '\n' + game_end + '\n'
                self.red_prompt += '\n' + game_end + '\n'
                break
            time.sleep(5) # add some sleep to avoid server hangups
            red_action = self.red_tank.pilot(intent_and_status = self.red_prompt)
            game_end = self.update_board('red', red_action.values()[0][0:np.min([100,len(red_action.values()[0])])])
            if 'victory' in game_end:
                self.blue_prompt += '\n' + game_end + '\n'
                self.red_prompt += '\n' + game_end + '\n'
                break
            time.sleep(5) # add some sleep to avoid server hangups
            
        # ask the winner and loser why they think they won and lost
        if 'Blue victory' in game_end:
            self.blue_prompt += 'Congratulations on your victory! Please tell your Captain how and why you won the battle. Limit your response to 5 sentences.\n'
            self.red_prompt += 'Too bad! Please tell your Captain how and why you lost the battle. Limit your response to 5 sentences.\n'
            blue_response = self.blue_tank.pilot(intent_and_status = self.blue_prompt)
            self.blue_prompt += blue_response.values()[0][0:np.min([100,len(blue_response.values()[0])])]
            red_response = self.red_tank.pilot(intent_and_status = self.red_prompt)
            self.red_prompt += red_response.values()[0][0:np.min([100,len(red_response.values()[0])])]
        elif 'Red victory' in game_end:
            self.red_prompt += 'Congratulations on your victory! Please tell your Captain how and why you won the battle. Limit your response to 5 sentences.\n'
            self.blue_prompt += 'Too bad! Please tell your Captain how and why you lost the battle. Limit your response to 5 sentences.\n'
            blue_response = self.blue_tank.pilot(intent_and_status = self.blue_prompt)
            self.blue_prompt += blue_response.values()[0][0:np.min([100,len(blue_response.values()[0])])]
            red_response = self.red_tank.pilot(intent_and_status = self.red_prompt)
            self.red_prompt += red_response.values()[0][0:np.min([100,len(red_response.values()[0])])]
            
        print('\n\nGame Results:\n\n')
        print('From blue tank perspective: \n\n{}'.format(self.blue_prompt))
        print('\n\n\nFrom red tank perspective:\n\n{}\n'.format(self.red_prompt))
        logging.info('\n\nGame Results:\n\n')
        logging.info('From blue tank perspective: \n\n{}'.format(self.blue_prompt))
        logging.info('\n\n\nFrom red tank perspective:\n\n{}\n'.format(self.red_prompt))
    
# %% run game with a number of temperatures
time.sleep(3600*5) # run in the middle of the night to avoid server hangups
n_turns = 50 # maximum number of turns per round
temperatures = np.linspace(0.7, 1.0, 4) # list of temperatures to try
for ii, temp in enumerate(temperatures):
    test = SimulationBoard(blue_order, red_order, temperature=temp)
    test.play_game(n_turns=n_turns)

print('\nComplete.')