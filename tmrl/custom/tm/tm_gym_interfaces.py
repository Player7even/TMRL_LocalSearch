import logging
import time
from collections import deque
import csv
import os
import traceback
import cv2
import gymnasium.spaces as spaces
import numpy as np
from rtgym import RealTimeGymInterface
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost


#1.GLOBALE EINSTELLUNGEN & HILFSFUNKTIONEN
LOG_FILE_PATH = r"C:\Users\[Username]\TmrlData\local_search_lognormal.csv"

#LOG_FILE_PATH je nach Trainingsmethode manuell zu aendern
#tm_state_log.csv \ local_search_logspeed.csv \ local_search_lognormal.csv \ local_search_logmixed.csv

def GLOBAL_LOG_HELPER(action, data, rew, terminated, label="UNKNOWN"):
    try:
        #Ordner erstellen
        file_exists = os.path.exists(LOG_FILE_PATH)
        
        t, b, s = action[0], action[1], action[2]
        px, py, pz = data[2], data[3], data[4]

        #Datei öffnen und schreiben
        file_exists = os.path.exists(LOG_FILE_PATH)

        row = [t, b, s, px, py, pz, rew, terminated]

        with open(LOG_FILE_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["throttle", "brake", "steer", "pos_x", "pos_y", "pos_z", "reward", "terminated"])
            writer.writerow(row)
            
    except Exception as e:
        print("LOGGING FEHLER TM2020Interface.grab_data_and_img:", e)


#2.BASIS INTERFACE
class TM2020Interface(RealTimeGymInterface):
    def __init__(self, img_hist_len: int = 4, gamepad: bool = True, save_replays: bool = False, grayscale: bool = True, resize_to=(64, 64)):
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']
        self.last_action = [0.0, 0.0, 0.0]
        self.initialized = False

    def initialize_common(self):
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],
                                              nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],
                                              nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],
                                              min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                              max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'])
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        if control is not None:
            self.last_action = control

        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0: actions.append('f')
                if control[1] > 0: actions.append('b')
                if control[2] > 0.5: actions.append('r')
                elif control[2] < -0.5: actions.append('l')
                apply_control(actions)

    def get_last_action(self):
        return getattr(self, "last_action", [0.0, 0.0, 0.0])

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]
        if self.resize_to is not None: img = cv2.resize(img, self.resize_to)
        if self.grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: img = img[:, :, ::-1]
        data = self.client.retrieve_data()
        self.img = img
        return data, img

    def reset_race(self):
        if self.gamepad: gamepad_reset(self.j)
        else: keyres()

    def reset_common(self):
        if not self.initialized: self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)

    def reset(self, seed=None, options=None):
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([data[0],], dtype='float32')
        gear = np.array([data[9],], dtype='float32')
        rpm = np.array([data[10],], dtype='float32')
        for _ in range(self.img_hist_len): self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        if self.gamepad: gamepad_close_finish_pop_up_tm20(self.j)
        else: mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        data, img = self.grab_data_and_img()
        speed = np.array([data[0],], dtype='float32')
        gear = np.array([data[9],], dtype='float32')
        rpm = np.array([data[10],], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty
        rew = np.float32(rew)
        
        #LOGGING
        GLOBAL_LOG_HELPER(self.get_last_action(), data, rew, terminated, label="BASE")
        return obs, rew, terminated, info

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None: w, h = self.resize_to
        else: w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale: img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else: img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False):
        super().__init__(img_hist_len, gamepad, save_replays)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0],], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True
        print("!!! LIDAR INTERFACE INITIALIZED !!!")

    def reset(self, seed=None, options=None):
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len): self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
    
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]), speed=speed)
        
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        
        rew += self.constant_penalty
        rew = np.float32(rew)

        action = self.get_last_action()

        GLOBAL_LOG_HELPER(action, data, rew, terminated)

        current_speed = speed[0]

        #Speed Bonus
        if current_speed > 1.0:
            rew += current_speed * 0.005 
        
        #Rückwärtsfahren bestrafen
        if current_speed < -1.0:
            rew -= 1.0 
            
        #Brems-Strafe
        if action is not None and action[1] > 0.01:
            rew -= cfg.REWARD_CONFIG.get('BRAKE_PENALTY', 0.0) * action[1]
        
        return obs, rew, terminated, info

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19,))
        return spaces.Tuple((speed, imgs))


#4.LIDAR PROGRESS INTERFACE
class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):
    def reset(self, seed=None, options=None):
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len): self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)

        action = self.get_last_action()
        GLOBAL_LOG_HELPER(action, data, rew, terminated)
        
        return obs, rew, terminated, info

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19,))
        return spaces.Tuple((speed, progress, imgs))