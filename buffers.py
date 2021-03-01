import numpy as np
from utils import log_trajectory_statistics


class ReplayBuffer(object):
    """Basic replay buffer."""
    def __init__(self, buffer_size, initial_data={}):
        self.buffer_size = buffer_size
        if initial_data == {}:
            self.N = -1
        else:
            self._initial_setup(initial_data)

    def _initial_setup(self, initial_data={}):
        self.obs = initial_data['obs'].astype('float32')
        self.nobs = initial_data['nobs'].astype('float32')
        self.act = initial_data['act'].astype('float32')
        self.rew = initial_data['rew'].astype('float32')
        self.don = initial_data['don']
        self.N = initial_data['n']

    def add(self, other_data):
        """Add collected data from Sampler."""
        if self.N == -1:
            self._initial_setup(other_data)
        else:
            self.N += other_data['n']
            offset_index = int(np.amax(np.array([self.N - self.buffer_size, 0])))
            self.N -= offset_index
            self.obs = np.concatenate((self.obs[offset_index:],
                                       other_data['obs'].astype('float32')), axis=0)
            self.nobs = np.concatenate((self.nobs[offset_index:],
                                        other_data['nobs'].astype('float32')), axis=0)
            self.act = np.concatenate((self.act[offset_index:],
                                       other_data['act'].astype('float32')), axis=0)
            self.rew = np.concatenate((self.rew[offset_index:],
                                       other_data['rew'].astype('float32')), axis=0)
            self.don = np.concatenate((self.don[offset_index:],
                                       other_data['don']), axis=0)

    def gather_indices(self, indices):
        out_dict = {}
        out_dict['obs'], out_dict['act'], out_dict['nobs'] = (self.obs[indices],
                                                              self.act[indices],
                                                              self.nobs[indices])
        out_dict['rew'], out_dict['don'] = self.rew[indices], self.don[indices]
        return out_dict

    def get_random_batch(self, batch_size):
        """Get random batch of data."""
        indices = np.random.randint(self.N, size=batch_size)
        return self.gather_indices(indices)

    def get_stats_previous_timesteps(self, num_timesteps):
        """Get basic statistics of latest collected data."""
        if num_timesteps > self.N:
            print('Not enough samples in the buffer')
            return
        latest_obs = self.obs[-num_timesteps - 1:]
        latest_rew = self.rew[-num_timesteps - 1:]
        latest_don = self.don[-num_timesteps - 1:]
        trajectory_startpoints = np.where(latest_don)[0] + 1
        number_startpoints = trajectory_startpoints.shape[0]
        trajectory_rewards = []
        for i in range(number_startpoints - 1):
            trajectory_rewards.append(np.sum(
                latest_rew[trajectory_startpoints[i]:trajectory_startpoints[i + 1]]))
        if len(trajectory_rewards) == 0:
            print('Latest number of completed trajectories - {}'.format(
                len(trajectory_rewards)))
        else:
            log_trajectory_statistics(trajectory_rewards)


class VisualReplayBuffer(ReplayBuffer):
    """Replay buffer with added support for visual observations."""
    def __init__(self, buffer_size, initial_data={}):
        super(VisualReplayBuffer, self).__init__(buffer_size, initial_data)

    def _initial_setup(self, initial_data={}):
        super(VisualReplayBuffer, self)._initial_setup(initial_data)
        self.ims = initial_data['ims'].astype(np.uint8)

    def add(self, other_data):
        if self.N == -1:
            self._initial_setup(other_data)
        else:
            self.N += other_data['n']
            offset_index = int(np.amax(np.array([self.N - self.buffer_size, 0])))
            self.N -= offset_index
            self.obs = np.concatenate((self.obs[offset_index:],
                                       other_data['obs'].astype('float32')), axis=0)
            self.nobs = np.concatenate((self.nobs[offset_index:],
                                        other_data['nobs'].astype('float32')), axis=0)
            self.act = np.concatenate((self.act[offset_index:],
                                       other_data['act'].astype('float32')), axis=0)
            self.rew = np.concatenate((self.rew[offset_index:],
                                       other_data['rew'].astype('float32')), axis=0)
            self.don = np.concatenate((self.don[offset_index:],
                                       other_data['don']), axis=0)
            self.ims = np.concatenate((self.ims[offset_index:],
                                       other_data['ims'].astype(np.uint8)), axis=0)

    def gather_indices(self, indices):
        """Get random batch of data."""
        out_dict = super(VisualReplayBuffer, self).gather_indices(indices)
        out_dict['ims'] = ((self.ims[indices].astype('float32') + 0.5) / 256)
        return out_dict


class LearnerAgentReplayBuffer(VisualReplayBuffer):
    """Replay buffer computing calculating the pseudo-rewards from a discriminator."""
    def __init__(self, gail, buffer_size, reward_noise=True,
                 initial_data={}):
        super(LearnerAgentReplayBuffer, self).__init__(buffer_size, initial_data)
        self._pre = gail._pre
        self._disc = gail._disc
        self._rn = reward_noise

    def get_random_batch(self, batch_size, re_eval_rw=True):
        """Get random batch of data.

        Parameters
        ----------
        batch_size : Batch size of experience to collect.
        re_eval_rw : Compute pseudo-rewards for batch, default is True.
        """
        out = super(LearnerAgentReplayBuffer, self).get_random_batch(
            batch_size)
        if re_eval_rw:
            if self._rn:
                out['pre'] = self._pre(out['ims'])
            else:
                out['pre'], _ = self._pre.get_distribution_info(out['ims'])
            out['rew'] = self._disc.get_reward(out['pre'])
        return out


class DemonstrationsReplayBuffer(object):
    """Replay buffer efficiently storing priorly collected visual observations."""
    def __init__(self, initial_data):
        self.ims = initial_data['ims'][:, 0, :, :, :].astype(np.uint8)
        self.N = self.ims.shape[0]
        self.ids = initial_data['ids']
        self.past_frames = initial_data['ims'].shape[1]
        self.idx_shifts = np.expand_dims(np.arange(self.past_frames), axis=0)

        self.pad_image = (np.zeros_like(self.ims[0]).astype('float32') + 0.5) / 256
        _, self.first_indices = np.unique(self.ids, return_index=True)

        self.first_ims = initial_data['ims'][self.first_indices, 1, :, :, :].astype(np.uint8)

        retrieval_indices = np.arange(self.first_indices.shape[0])
        self.padded_retrieval_list = np.zeros([self.N]).astype('int') + 1000000
        self.padded_retrieval_list[self.first_indices] = retrieval_indices

    def gather_indices(self, indices):
        all_indices = np.expand_dims(indices, axis=-1) - self.idx_shifts
        images = (self.ims[all_indices].astype('float32') + 0.5) / 256
        start_indices_mask = np.isin(all_indices[:, :-1], self.first_indices)
        start_indices_x, start_indices_y = np.where(start_indices_mask)
        first_indices_y = start_indices_y + 1

        trajectories_start_indices = all_indices[start_indices_x, start_indices_y]
        images[start_indices_x, first_indices_y] = (self.first_ims[
                                                        self.padded_retrieval_list[trajectories_start_indices]].astype(
            'float32') + 0.5) / 256

        pad_indices_mask = start_indices_mask[:, :-1]
        for i in range(self.past_frames - 3):
            pad_indices_mask[:, i + 1] = np.logical_or(pad_indices_mask[:, i + 1],
                                                       pad_indices_mask[:, i])
        pad_indices_x, pad_indices_y = np.where(pad_indices_mask)
        pad_indices_y += 2
        images[pad_indices_x, pad_indices_y] = self.pad_image

        return {'ims': images}

    def get_random_batch(self, batch_size):
        """Get random batch of data."""
        indices = np.random.randint(self.N, size=batch_size)
        return self.gather_indices(indices)


