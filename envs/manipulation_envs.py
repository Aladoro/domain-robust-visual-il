import random

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import GlfwContext
import cv2
import os
from envs.envs import _FrameBufferEnv


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        GlfwContext(offscreen=True)
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/pusher.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def step(self, a):
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(a).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.125
        self.viewer.cam.elevation = -75

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.20, 0.175])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")

        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('int32')
        return {'obs': ob, 'im': curr_frames}


class PusherHumanSimEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        GlfwContext(offscreen=True)
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/pusher_human_sim.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def step(self, a):
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(a).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.125
        self.viewer.cam.elevation = -75

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.20, 0.175])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")

        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('int32')
        return {'obs': ob, 'im': curr_frames}


class StrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        GlfwContext(offscreen=True)
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False

        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/striker.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2) #5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))

        if np.linalg.norm(vec_1) < self.strike_threshold:
            self._striked = True
            self._strike_pos = self.get_body_com("tips_arm")

        if self._striked:
            vec_3 = self.get_body_com("object") - self._strike_pos
            reward_near = - np.linalg.norm(vec_3)
        else:
            reward_near = - np.linalg.norm(vec_1)

        reward_dist = - np.linalg.norm(self._min_strike_dist)
        reward_ctrl = - np.square(a).sum()
        reward = 3 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4
        self.viewer.cam.elevation = -75

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        self.goal = np.array([0.15, 0.3])

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)

        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('int32')
        return {'obs': ob, 'im': curr_frames}


class StrikerHumanSimEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        GlfwContext(offscreen=True)
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False

        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/striker_human_sim.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2) #5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))

        if np.linalg.norm(vec_1) < self.strike_threshold:
            self._striked = True
            self._strike_pos = self.get_body_com("tips_arm")

        if self._striked:
            vec_3 = self.get_body_com("object") - self._strike_pos
            reward_near = - np.linalg.norm(vec_3)
        else:
            reward_near = - np.linalg.norm(vec_1)

        reward_dist = - np.linalg.norm(self._min_strike_dist)
        reward_ctrl = - np.square(a).sum()
        reward = 3 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4
        self.viewer.cam.elevation = -75

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        self.goal = np.array([0.15, 0.3])

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)

        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('int32')
        return {'obs': ob, 'im': curr_frames}
