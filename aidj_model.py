# aidj_model.py
# RL environment, small DQN, quick_train, save/load, action helper

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple

# === Environment (simple synthetic) ===
class AIDJEnvSimple:
    """
    Small synthetic env used for quick training.
    Observation: [tempo_diff, energy_diff, centroid_diff, key_compat, vol_a, vol_b] (6 dims)
    Actions: 0..7 as in earlier design (inc/dec volumes, crossfade etc.)
    """
    def __init__(self, episode_len=40):
        self.episode_len = episode_len
        self.t = 0
        self.max_vol = 1.0
        self.min_vol = 0.0
        self.reset()

    def sample_track(self):
        return {
            'tempo': random.uniform(80, 140),
            'energy': random.uniform(0.2, 1.0),
            'centroid': random.uniform(1000, 6000),
            'key': random.randint(0, 11)
        }

    def _key_compat(self, a, b):
        diff = min((a - b) % 12, (b - a) % 12)
        return max(0.0, 1.0 - (diff / 6.0))

    def reset(self):
        self.t = 0
        self.trackA = self.sample_track()
        self.trackB = self.sample_track()
        self.vol_a = 1.0
        self.vol_b = 0.0
        return self._get_obs()

    def _get_obs(self):
        tempo_diff = abs(self.trackA['tempo'] - self.trackB['tempo']) / 60.0
        energy_diff = abs(self.trackA['energy'] - self.trackB['energy'])
        centroid_diff = abs(self.trackA['centroid'] - self.trackB['centroid']) / 5000.0
        key_compat = self._key_compat(self.trackA['key'], self.trackB['key'])
        return np.array([tempo_diff, energy_diff, centroid_diff, key_compat, self.vol_a, self.vol_b], dtype=np.float32)

    def step(self, action):
        self.t += 1
        pv_a, pv_b = self.vol_a, self.vol_b
        delta = 0.06
        if action == 0: self.vol_a = min(self.max_vol, self.vol_a + delta)
        elif action == 1: self.vol_a = max(self.min_vol, self.vol_a - delta)
        elif action == 2: self.vol_b = min(self.max_vol, self.vol_b + delta)
        elif action == 3: self.vol_b = max(self.min_vol, self.vol_b - delta)
        elif action == 4:
            self.vol_a = max(self.min_vol, self.vol_a - delta); self.vol_b = min(self.max_vol, self.vol_b + delta)
        elif action == 5:
            self.vol_a = min(self.max_vol, self.vol_a + delta); self.vol_b = max(self.min_vol, self.vol_b - delta)
        elif action == 6:
            # small random nudge
            self.vol_a = min(self.max_vol, max(self.min_vol, self.vol_a + (random.random()-0.5)*0.02))
            self.vol_b = min(self.max_vol, max(self.min_vol, self.vol_b + (random.random()-0.5)*0.02))
        else:
            pass

        jerk = abs(self.vol_a - pv_a) + abs(self.vol_b - pv_b)
        smooth_reward = -jerk * 0.6
        compat = (1.0 - abs(self.trackA['tempo'] - self.trackB['tempo']) / 60.0) * (1.0 - abs(self.trackA['energy'] - self.trackB['energy'])) * self._key_compat(self.trackA['key'], self.trackB['key'])
        audibility = min(self.vol_a, self.vol_b)
        compat_reward = audibility * compat * 1.2
        clip_penalty = -1.0 * max(0.0, (self.vol_a + self.vol_b) - 1.2)
        dominance = -abs(self.vol_a - self.vol_b) * 0.04
        reward = smooth_reward + compat_reward + clip_penalty + dominance

        obs = self._get_obs()
        done = (self.t >= self.episode_len)
        return obs, reward, done, {}

# === DQN & replay ===
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(Transition(*args))
    def sample(self, n):
        batch = random.sample(self.buffer, n)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, in_dim=6, out_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self,x): return self.net(x)

# === training helper (quick small training for demo) ===
def quick_train(episodes=300, episode_len=40, device=None):
    """
    Quick training routine (CPU-friendly). Produces a small demo-quality policy.
    Run offline via train_model.py and then start the Flask app.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = AIDJEnvSimple(episode_len=episode_len)
    obs_dim = len(env.reset())
    n_actions = 8
    policy = DQN(obs_dim, n_actions).to(device)
    target = DQN(obs_dim, n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    replay = ReplayBuffer(8000)
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995
    gamma = 0.99
    batch_size = 64
    min_replay = 400

    for ep in range(episodes):
        s = env.reset()
        ep_r = 0.0
        for _ in range(episode_len):
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    tns = torch.FloatTensor(s).unsqueeze(0).to(device)
                    a = int(policy(tns).argmax(1).cpu().numpy()[0])
            ns, r, done, _ = env.step(a)
            replay.push(s,a,r,ns,done)
            s = ns
            ep_r += r
            if len(replay) >= min_replay:
                batch = replay.sample(batch_size)
                states = torch.FloatTensor(batch.state).to(device)
                actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(batch.next_state).to(device)
                dones = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
                q_vals = policy(states).gather(1, actions)
                next_q = target(next_states).max(1)[0].detach().unsqueeze(1)
                expected = rewards + gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_vals, expected)
                opt.zero_grad(); loss.backward(); opt.step()

        eps = max(eps_min, eps * eps_decay)
        if ep % 20 == 0:
            target.load_state_dict(policy.state_dict())
            print(f"[train] ep {ep}/{episodes} ep_reward={ep_r:.3f} eps={eps:.3f}")

    return policy

# === save/load/get action ===
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(path, device=None):
    if not os.path.exists(path):
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DQN(in_dim=6, out_dim=8).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net

def get_action(model, obs, device=None):
    if model is None:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        act = int(model(t).argmax(1).cpu().numpy()[0])
    return act

# If this module is executed directly you can run a short quick check
if __name__ == "__main__":
    print("Quick sanity: training 10 episodes (demo)...")
    m = quick_train(episodes=10)
    print("Done")
