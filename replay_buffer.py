import os
import numpy as np

STATE_SHAPE = (8, 84, 84)

class ReplayBuffer:
    def __init__(self, capacity, data_dir="replay_buffer"):
        os.makedirs(data_dir, exist_ok=True)
        self.capacity   = capacity
        self._meta_path = os.path.join(data_dir, "meta.npz")

        if os.path.exists(self._meta_path):
            meta      = np.load(self._meta_path)
            self.ptr  = int(meta["ptr"])
            self.size = int(meta["size"])
            mode      = "r+"
        else:
            self.ptr  = 0
            self.size = 0
            mode      = "w+"

        self.states      = np.memmap(os.path.join(data_dir, "states.dat"),      dtype=np.uint8,   mode=mode, shape=(capacity, *STATE_SHAPE))
        self.actions     = np.memmap(os.path.join(data_dir, "actions.dat"),     dtype=np.int64,   mode=mode, shape=(capacity,))
        self.rewards     = np.memmap(os.path.join(data_dir, "rewards.dat"),     dtype=np.float32, mode=mode, shape=(capacity,))
        self.next_states = np.memmap(os.path.join(data_dir, "next_states.dat"), dtype=np.uint8,   mode=mode, shape=(capacity, *STATE_SHAPE))
        self.dones       = np.memmap(os.path.join(data_dir, "dones.dat"),       dtype=np.float32, mode=mode, shape=(capacity,))

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = (state * 255).astype(np.uint8)
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = (next_state * 255).astype(np.uint8)
        self.dones[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs].astype(np.float32)      / 255.0,
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs].astype(np.float32) / 255.0,
            self.dones[idxs],
        )

    def save(self):
        np.savez(self._meta_path, ptr=self.ptr, size=self.size)

    def __len__(self):
        return self.size
