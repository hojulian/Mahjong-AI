import gym
from gym import spaces
import numpy as np
import websocket
import json

class MahjongEnv(gym.Env):
    """Multi-agent RLlib-compatible env for 4-player Japanese Mahjong over WebSocket."""

    def __init__(self, config):
        super().__init__()
        self.ws_url = config.get("ws_url")
        self.num_agents = 4
        self.agent_ids = [f"player_{i}" for i in range(self.num_agents)]
        self.max_tiles = config.get("max_tiles", 14)
        self.action_size = config.get("action_size", 40)

        # Example obs/action spaces
        self.observation_space = spaces.Dict({
            agent_id: spaces.Dict({
                "hand": spaces.MultiBinary(34 * self.max_tiles),
            }) for agent_id in self.agent_ids
        })
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(self.action_size)
            for agent_id in self.agent_ids
        })

        self.ws = None
        self._connect_ws()

    def _connect_ws(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(self.ws_url)

    def reset(self):
        self.ws.send(json.dumps({"cmd": "reset"}))
        obs_json = self.ws.recv()
        obs_dict = json.loads(obs_json)  # should be {agent_id: obs}
        return {aid: self._parse_obs(obs) for aid, obs in obs_dict.items()}

    def step(self, action_dict):
        # Send actions for all agents at once
        self.ws.send(json.dumps({
            "cmd": "step",
            "actions": {aid: int(a) for aid, a in action_dict.items()}
        }))
        resp_json = self.ws.recv()
        resp = json.loads(resp_json)  # Should contain obs, reward, done, info for each agent
        obs = {aid: self._parse_obs(o) for aid, o in resp['observation'].items()}
        rewards = resp.get('reward', {aid: 0 for aid in self.agent_ids})
        dones = resp.get('done', {aid: False for aid in self.agent_ids})
        dones["__all__"] = resp.get('done', {}).get("__all__", False)
        infos = resp.get('info', {aid: {} for aid in self.agent_ids})
        return obs, rewards, dones, infos

    def _parse_obs(self, obs_dict):
        # Implement your obs parsing
        return obs_dict

    def close(self):
        if self.ws:
            self.ws.close()
