# 请将 'environments/alfworld_wrapper.py' 中原有的 AlfWorldWrapper 类完全替换为以下代码

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import json
import os
from typing import Dict, List, Any, Optional
import subprocess
import time
import re
import torch
import yaml
import random

class AlfWorldWrapper(AECEnv):
    """
    AlfWorld环境包装器
    支持文本交互的智能体任务执行
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 data_root: str = None,
                 split: str = 'train',
                 no_graphics: bool = True,
                 time_scale: float = 1.0,
                 seed: int = 1,
                 max_steps: int = 100):
        super().__init__()
        
        self.data_root = data_root or os.path.expanduser("~/alfworld")
        self.split = split
        self.no_graphics = no_graphics
        self.time_scale = time_scale
        self.seed = seed
        self.max_steps = max_steps
        
        # AlfWorld任务类型 (仅作参考)
        self.task_types = [
            'pick_and_place_simple', 'pick_clean_then_place_in_recep',
            'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'pick_two_obj_and_place', 'look_at_obj_in_light',
            'pick_and_place_with_movable_recep'
        ]
        
        # 当前任务状态
        self.current_task = None
        self.current_goal = None
        self.step_count = 0
        self.task_history = []
        self.last_raw_obs = ""
        
        # 初始化agent
        self.possible_agents = ['alfred']
        self.agents = self.possible_agents[:]
        self._agent_sel = agent_selector(self.agents)
        self.agent_selection = self._agent_sel.next()
        
        # 文本编码器（用于状态表示）
        self.text_encoder = None
        self._init_text_encoder()
        
        # 定义观察和动作空间
        self._define_spaces()
        
        # 状态管理
        self.rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        
        # PettingZoo AECEnv必需的属性
        self._cumulative_rewards = {a: 0 for a in self.agents}

        # 初始化环境
        self._init_alfworld()

    def _init_alfworld(self):
        """初始化AlfWorld环境"""
        try:
            import alfworld
            print("AlfWorld已安装")
        except ImportError:
            print("正在安装AlfWorld...")
            subprocess.run(["pip", "install", "alfworld"], check=True)
            import alfworld
        
        os.environ['ALFWORLD_DATA'] = self.data_root
        
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        config_path = os.path.join(os.path.dirname(__file__), '../configs/alfworld_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.alfworld_env = AlfredTWEnv(config, train_eval=self.split)

        if self.alfworld_env.num_games > 0:
            self.alfworld_env.init_env(batch_size=1)
            print(f"AlfWorld环境初始化完成，加载了 {self.alfworld_env.num_games} 个游戏。")
        else:
            print("警告: 未找到AlfWorld游戏数据，环境可能无法正常工作。")
            self.alfworld_env = None

    def _init_text_encoder(self):
        """初始化文本编码器"""
        try:
            from models.encoder import NomicEncoder
            self.text_encoder = NomicEncoder()
            print("文本编码器初始化完成")
        except ImportError:
            print("警告: 无法导入文本编码器，将使用简单文本表示")
            self.text_encoder = None
    
    def _define_spaces(self):
        """定义观察和动作空间"""
        obs_dim = 512 if self.text_encoder else 100
        
        self.observation_spaces = {
            'alfred': spaces.Dict({
                'current_state': spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                'goal_description': spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                'action_history': spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                'inventory': spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                'raw_text': spaces.Text(max_length=1000),
            })
        }
        
        self.action_spaces = {'alfred': spaces.Discrete(20)}
        
        self.action_mapping = {
            0: "look", 1: "move", 2: "pick", 3: "put", 4: "open",
            5: "close", 6: "toggle", 7: "heat", 8: "cool", 9: "clean",
            10: "slice", 11: "examine", 12: "use", 13: "find", 14: "go to",
            15: "take", 16: "place", 17: "turn on", 18: "turn off", 19: "break"
        }
    
    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        if self.text_encoder:
            try:
                with torch.no_grad():
                    encoded = self.text_encoder.encode_text(text)
                    return encoded.cpu().numpy().flatten()
            except Exception:
                pass
        return np.random.randn(512 if self.text_encoder else 100)
    
    def _get_current_observation(self) -> Dict[str, Any]:
        """获取当前观察"""
        if not self.current_task:
            return self._get_empty_observation()
        
        current_state = self.last_raw_obs
        goal_desc = self.current_goal or "Complete the task"
        
        state_encoded = self._encode_text(current_state)
        goal_encoded = self._encode_text(goal_desc)
        history_encoded = self._encode_text(" ".join(self.task_history[-5:]))
        inventory_encoded = self._encode_text(self._get_inventory_text())
        
        return {
            'current_state': state_encoded, 'goal_description': goal_encoded,
            'action_history': history_encoded, 'inventory': inventory_encoded,
            'raw_text': current_state
        }
    
    def _get_empty_observation(self) -> Dict[str, Any]:
        """获取空观察"""
        empty_vec = np.zeros(512 if self.text_encoder else 100)
        return {
            'current_state': empty_vec, 'goal_description': empty_vec,
            'action_history': empty_vec, 'inventory': empty_vec,
            'raw_text': ""
        }
    
    def _get_inventory_text(self) -> str:
        """获取库存文本描述"""
        if "You are carrying:" in self.last_raw_obs:
            return self.last_raw_obs.split("You are carrying:")[1].strip()
        return "nothing"
    
    def reset(self, seed=None, options=None):
        """重置环境并加载新任务"""
        if self.alfworld_env is None:
            print("错误: AlfWorld环境未成功初始化或无游戏数据，无法重置。")
            return self._get_empty_observation()

        if seed is not None:
            self.seed = seed
            
        self.agents = self.possible_agents[:]
        self._agent_sel = agent_selector(self.agents)
        self.agent_selection = self._agent_sel.next()

        self.rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.step_count = 0
        self.task_history = []

        try:
            obs, infos = self.alfworld_env.reset()
            self.last_raw_obs = obs[0] if obs else "Initial observation failed."

            game_file = infos['extra.gamefile'][0]
            traj_data_path = os.path.join(os.path.dirname(game_file), 'traj_data.json')
            with open(traj_data_path, 'r') as f:
                traj_data = json.load(f)
            
            self.current_goal = random.choice(traj_data['turk_annotations']['anns'])['task_desc']
            self.current_task = traj_data['task_type']
            print(f"加载任务: {self.current_task}")
            print(f"目标: {self.current_goal}")

        except Exception as e:
            print(f"警告: AlfWorld环境重置失败 - {e}")
            self.last_raw_obs = "Error resetting environment."
            self.current_goal = "No goal loaded."
            self.current_task = None

        return self._get_current_observation()

    def step(self, action):
        """执行一个动作"""
        if self.dones.get(self.agent_selection, False):
            return self._was_done_step(action)
        
        agent = self.agent_selection
        
        action_name = self.action_mapping.get(action, "look")
        self.task_history.append(action_name)
        
        success = self._execute_action(action_name)
        
        reward = 0
        done = False
        if success:
            reward = 1.0
            if self._check_task_completion():
                reward = 10.0
                done = True
        else:
            reward = -1.0
        
        self.rewards[agent] = reward
        self.dones[agent] = done or self.step_count >= self.max_steps
        self.infos[agent] = {'success': success, 'task_complete': self._check_task_completion()}
        
        self.agent_selection = self._agent_sel.next()
        self.step_count += 1
        self._accumulate_rewards()
        
        # 返回当前观察
        return self._get_current_observation(), reward, done, self.infos[agent]

    def _execute_action(self, action_name: str) -> bool:
        """执行动作并返回成功状态"""
        if not self.alfworld_env:
            return False
        try:
            obs, scores, dones, infos = self.alfworld_env.step([action_name])
            self.last_raw_obs = obs[0] if obs else "No observation returned from step."
            return True
        except Exception as e:
            print(f"执行动作 '{action_name}' 失败: {e}")
            return False

    def _check_task_completion(self) -> bool:
        """检查任务是否完成"""
        return "You have won" in self.last_raw_obs.lower()

    def close(self):
        """关闭环境"""
        if self.alfworld_env:
            self.alfworld_env.close()

    def get_task_info(self) -> Dict[str, Any]:
        """获取当前任务信息"""
        return {
            'task_type': self.current_task,
            'goal': self.current_goal,
            'history': self.task_history
        }
