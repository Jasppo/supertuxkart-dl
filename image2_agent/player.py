from numpy.lib.function_base import angle
import torch
import numpy as np
import math
from .planner import Planner, save_model, load_model
import torchvision.transforms.functional as TF
import itertools
import matplotlib.pyplot as plt

def _save_image(image, step, ball_location):
    import PIL.Image
    import os
    path_to_save = 'match/'

    image_data = []
    fig = plt.figure()
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.clear()
    ax.imshow(image)
    WH2 = np.array([400, 300])/2
    ax.add_artist(plt.Circle(WH2*(ball_location + 1), 2, ec = 'r', fill = False, lw = 15))
    # ax.add_artist(plt.Circle(ball_location, 2, ec = 'r', fill = False, lw = 15))
    ax.axis('off')
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    plt.show()
    plt.close()

    PIL.Image.fromarray(X).save(os.path.join(path_to_save, 'step_%04d.png' % (step)))

def _to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])

    if p[2] < 0 and p[3] < 0:
        return np.array([0., 1.])
    else:
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

def is_offscreen(screen_coords):
    return np.allclose(screen_coords, [0, 1])

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

def extract_features(pstate, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    goal_centers = [torch.tensor([0.0000, -64.5000]), torch.tensor([0.0000, 64.5000])]
    enemy_goal = goal_centers[(team_id+1)%2]
    my_goal = goal_centers[(team_id)%2]
    # features of score-line 
    #goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    kart_to_goal_line = (enemy_goal-kart_center) / torch.norm(enemy_goal-kart_center)
    goal_angle = torch.atan2(kart_to_goal_line[1], kart_to_goal_line[0])
    goal_kart_angle_difference = limit_period((kart_angle - goal_angle)/np.pi)
    return (kart_front, goal_kart_angle_difference)


class Team:
    agent_type = 'image'
    def __init__(self):
        self.team = None
        self.num_players = None
        self.memory = [[], []]
        self.correcting = [False, False]
        self.n_frames = 0
        self.step = 0

        # Load model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model()
        self.model.eval()

    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        actions = [] 
        max_velocity = 19.5
        self.n_frames += 1
        for player_id, pstate in enumerate(player_state):
            _, goal_angle = extract_features(pstate, self.team)

            pimage = TF.to_tensor(player_image[player_id])[None]
            puck_lbl, puck_screen_loc = self.model(pimage)
            puck_screen_loc = puck_screen_loc.squeeze(0).detach().cpu().numpy()

            puck_lbl = (torch.sigmoid(puck_lbl) > 0.25).long().squeeze(0).detach().cpu().numpy().item()

            """
            if puck_lbl == 0:
                puck_screen_loc = np.array([0., 1.])
            """

            """
            # Save images to make video later
            if (player_id % 2) == 0:
                _save_image(image=player_image[player_id], step=self.step, ball_location=puck_screen_loc)
            """

            # start is a little weird, just drive forward
            if self.n_frames < 10:
                self.correcting[player_id] = False
            # puck offscreen
            elif is_offscreen(puck_screen_loc):
                self.correcting[player_id] = True
            # if correcting position, don't stop until the ball is close to the center
            elif self.correcting[player_id] and abs(puck_screen_loc[0] < .4):
                self.correcting[player_id] = False
            # reverse if close to passing the puck
            elif not self.correcting[player_id] and (puck_screen_loc[1] > 0):
                self.correcting[player_id] = True
            
            brake=False
            acc = 1
            steer = 0
            if self.correcting[player_id]:
                brake = True
                acc = 0
                if not is_offscreen(puck_screen_loc):
                    steer = np.clip(-100 * puck_screen_loc[0], -1, 1)
                else:
                    steer = -1
            else:
                brake = False
                acc = .9
                # adjust the angle to try to nudge towards enemy goal
                if puck_screen_loc[1] > -.15:
                    adjust = .1 * goal_angle
                else:
                    adjust = 0
                steer = 12 * (puck_screen_loc[0] - adjust)  

            actions += [dict(acceleration = acc, steer = steer, brake=brake)]
            self.step += 1
        
        return actions