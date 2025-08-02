#!/usr/bin/env python3
"""
AlfWorld评估脚本
评估训练好的RSN智能体在AlfWorld任务上的表现
"""

import argparse
import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import cv2  # 导入OpenCV库用于图像处理

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.alfworld_wrapper import AlfWorldWrapper
from agents.rsn_agent import RSNAgent
from models.llm_trajectory import LLMTrajectory

class AlfWorldEvaluator:
    """AlfWorld评估器"""
    
    def __init__(self, config: Dict, save_frames: bool = False):
        self.config = config
        self.save_frames = save_frames
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 初始化环境
        self.env = AlfWorldWrapper(
            data_root=config.get("data_root", "~/alfworld"),
            max_steps=config.get("max_steps", 100),
            no_graphics=not self.save_frames  # 如果保存帧，则强制开启图形
        )
        
        # 初始化智能体
        self.agent = RSNAgent(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            num_agents=config["num_agents"],
            device=self.device
        )
        
        # 加载训练好的模型
        if config.get("model_path"):
            self.load_model(config["model_path"])
        
        # 评估指标
        self.metrics = defaultdict(list)
        
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def evaluate_episode(self, task_type: str = None) -> Dict[str, Any]:
        """评估单个episode"""
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        actions_taken = []
        
        # 获取任务信息
        task_info = self.env.get_task_info()
        
        # 如果需要保存帧，创建目录
        if self.save_frames:
            task_name = task_info.get('task_type', 'unknown_task').replace('/', '_')
            frame_dir = os.path.join("results", "frames", f"{task_name}_{len(os.listdir('results/frames'))}" if os.path.exists("results/frames") else f"{task_name}_0")
            os.makedirs(frame_dir, exist_ok=True)
            print(f"图像将保存到: {frame_dir}")
        
        while episode_steps < self.config.get("max_steps", 100):
            # 智能体选择动作
            state = torch.tensor(obs['current_state'], dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.agent.select_action(state)
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            
            # 如果需要，保存当前帧
            if self.save_frames:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    # 将RGB转换为BGR以供OpenCV使用
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(frame_dir, f"frame_{episode_steps:04d}.png"), frame_bgr)
            
            episode_reward += reward
            episode_steps += 1
            actions_taken.append(action)
            
            if done:
                break
        
        # 计算评估指标
        success = episode_reward > 0  # 简化成功判断
        efficiency = episode_reward / max(episode_steps, 1)
        
        return {
            'task_type': task_info['task_type'],
            'goal': task_info['goal'],
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'success': success,
            'efficiency': efficiency,
            'actions_taken': actions_taken,
            'action_history': task_info['action_history']
        }
    
    def evaluate_multiple_episodes(self, num_episodes: int = 100) -> Dict[str, Any]:
        """评估多个episodes"""
        print(f"开始评估 {num_episodes} 个episodes...")
        
        results = []
        task_type_results = defaultdict(list)
        
        for i in range(num_episodes):
            if i % 10 == 0:
                print(f"评估进度: {i}/{num_episodes}")
            
            result = self.evaluate_episode()
            results.append(result)
            task_type_results[result['task_type']].append(result)
        
        # 计算总体指标
        total_metrics = self._calculate_metrics(results)
        
        # 计算各任务类型指标
        task_metrics = {}
        for task_type, task_results in task_type_results.items():
            task_metrics[task_type] = self._calculate_metrics(task_results)
        
        return {
            'overall': total_metrics,
            'by_task_type': task_metrics,
            'detailed_results': results
        }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算评估指标"""
        if not results:
            return {}
        
        metrics = {
            'success_rate': np.mean([r['success'] for r in results]),
            'avg_reward': np.mean([r['episode_reward'] for r in results]),
            'avg_steps': np.mean([r['episode_steps'] for r in results]),
            'avg_efficiency': np.mean([r['efficiency'] for r in results]),
            'std_reward': np.std([r['episode_reward'] for r in results]),
            'std_steps': np.std([r['episode_steps'] for r in results]),
            'min_reward': np.min([r['episode_reward'] for r in results]),
            'max_reward': np.max([r['episode_reward'] for r in results])
        }
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        overall = results['overall']
        
        print("\n" + "="*50)
        print("AlfWorld评估结果摘要")
        print("="*50)
        print(f"总体成功率: {overall['success_rate']:.2%}")
        print(f"平均奖励: {overall['avg_reward']:.2f} ± {overall['std_reward']:.2f}")
        print(f"平均步数: {overall['avg_steps']:.1f} ± {overall['std_steps']:.1f}")
        print(f"平均效率: {overall['avg_efficiency']:.3f}")
        print(f"奖励范围: [{overall['min_reward']:.2f}, {overall['max_reward']:.2f}]")
        
        print("\n各任务类型表现:")
        print("-" * 30)
        for task_type, metrics in results['by_task_type'].items():
            print(f"{task_type}:")
            print(f"  成功率: {metrics['success_rate']:.2%}")
            print(f"  平均奖励: {metrics['avg_reward']:.2f}")
            print(f"  平均步数: {metrics['avg_steps']:.1f}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="评估RSN智能体在AlfWorld环境中的表现")
    parser.add_argument("--config", type=str, default="configs/alfworld_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="评估episode数量")
    parser.add_argument("--output_path", type=str, default="results/alfworld_eval.json",
                       help="评估结果保存路径")
    parser.add_argument("--save_frames", action="store_true",
                       help="是否保存评估过程中的视觉帧")
    parser.add_argument("--device", type=str, default="auto",
                       help="评估设备 (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config.update({
        'model_path': args.model_path,
        'num_episodes': args.num_episodes
    })
    
    # 设置设备
    if args.device == "auto":
        config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config['device'] = torch.device(args.device)
    
    print("=== AlfWorld评估配置 ===")
    print(f"模型路径: {config['model_path']}")
    print(f"评估轮数: {config['num_episodes']}")
    print(f"保存图像: {'是' if args.save_frames else '否'}")
    print(f"使用设备: {config['device']}")
    print("=" * 30)
    
    # 创建评估器
    evaluator = AlfWorldEvaluator(config, save_frames=args.save_frames)
    
    # 执行评估
    # 如果保存图像，建议只跑一个episode
    num_eval_episodes = 1 if args.save_frames else args.num_episodes
    results = evaluator.evaluate_multiple_episodes(num_eval_episodes)
    
    # 打印摘要
    evaluator.print_summary(results)
    
    # 保存结果
    evaluator.save_results(results, args.output_path)
    
    # 关闭环境
    evaluator.env.close()

if __name__ == "__main__":
    main() 