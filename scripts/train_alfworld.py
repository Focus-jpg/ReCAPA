#!/usr/bin/env python3
"""
AlfWorld训练脚本
使用RSN框架训练智能体在AlfWorld环境中执行任务
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.alfworld_wrapper import AlfWorldWrapper
from trainers.trainer import Trainer
from models.llm_trajectory import LLMTrajectory

def main():
    parser = argparse.ArgumentParser(description="训练RSN智能体在AlfWorld环境中")
    parser.add_argument("--config", type=str, default="configs/alfworld_config.yaml", 
                       help="配置文件路径")
    parser.add_argument("--data_root", type=str, default="~/alfworld",
                       help="AlfWorld数据根目录")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="每轮最大步数")
    parser.add_argument("--save_dir", type=str, default="results/models/alfworld",
                       help="模型保存目录")
    parser.add_argument("--eval_interval", type=int, default=20,
                       help="评估间隔")
    parser.add_argument("--device", type=str, default="auto",
                       help="训练设备 (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置参数
    config.update({
        'data_root': args.data_root,
        'num_episodes': args.episodes,
        'max_steps': args.max_steps,
        'save_dir': args.save_dir,
        'eval_interval': args.eval_interval,
        'environment_type': 'alfworld'
    })
    
    # 设置设备
    if args.device == "auto":
        config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config['device'] = torch.device(args.device)
    
    # 从环境变量获取API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        config["llm_api_key"] = api_key
    elif not config.get("llm_api_key"):
        print("警告: 未找到OpenAI API密钥")
        print("请设置环境变量 OPENAI_API_KEY")
        return
    
    print("=== AlfWorld RSN训练配置 ===")
    print(f"环境类型: {config['environment_type']}")
    print(f"数据路径: {config['data_root']}")
    print(f"训练轮数: {config['num_episodes']}")
    print(f"最大步数: {config['max_steps']}")
    print(f"使用设备: {config['device']}")
    print(f"HPCR模式: {'启用' if config.get('enable_hpcr', False) else '禁用'}")
    print(f"零样本模式: {'启用' if config.get('zero_shot_mode', False) else '禁用'}")
    print("=" * 30)
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 初始化训练器
    try:
        trainer = Trainer(config)
        print("训练器初始化成功")
    except Exception as e:
        print(f"训练器初始化失败: {e}")
        return
    
    # 开始训练
    print("开始训练...")
    try:
        trainer.train()
        print("训练完成!")
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_alfworld_environment():
    """测试AlfWorld环境"""
    print("测试AlfWorld环境...")
    
    try:
        # 创建AlfWorld环境
        env = AlfWorldWrapper(
            data_root="~/alfworld",
            max_steps=50,
            no_graphics=True
        )
        
        print("AlfWorld环境创建成功")
        
        # 测试重置
        obs = env.reset()
        print(f"环境重置成功，观察维度: {obs['current_state'].shape}")
        
        # 测试几个动作
        for i in range(5):
            action = env.action_spaces['alfred'].sample()
            obs, reward, done, info = env.step(action)
            print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.2f}, 完成={done}")
            
            if done:
                break
        
        env.close()
        print("AlfWorld环境测试完成")
        return True
        
    except Exception as e:
        print(f"AlfWorld环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 首先测试环境
    if test_alfworld_environment():
        main()
    else:
        print("环境测试失败，请检查AlfWorld安装")
        print("安装命令: pip install alfworld") 