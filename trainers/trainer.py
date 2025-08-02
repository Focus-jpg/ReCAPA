import torch
import torch.optim as optim
from collections import defaultdict
import sys
import os
import torch.nn.functional as F
import numpy as np
from typing import Dict
import random
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import NomicEncoder
from models.forward_policy_network import PolicyNetwork
from models.llm_trajectory import LLMTrajectory, MultiTransition
from models.reflection_head import ReflectionHead, HPCRPredictionHead
from models.rce import RceModule
from models.cgf import CGFModule
# from environments.env_wrapper import MuMAToMGymEnv   
# from environments.env_wrapper import VirtualHomeWrapper  

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # --- Instantiate Modules ---
        self.encoder = NomicEncoder(device=self.device)
        
        # 对于VQA任务，我们不需要策略网络，但保留结构
        if config.get("task_type") == "vqa":
            print("VQA模式：跳过策略网络初始化")
            self.policy_net = None
        else:
        self.policy_net = PolicyNetwork(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            num_agents=config["num_agents"]
        ).to(self.device)

        self.rce_module = RceModule(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"],
            temperature=config.get("rce_temperature", 0.1),
            # Pass alignment-specific params
            sinkhorn_epsilon=config["sinkhorn_eps"],
            score_field_hidden_dim=config["score_hidden"],
            prompt_embedding_dim=config["state_dim"], # Assuming prompt emb dim matches state dim
            num_heads=config["num_heads"]
        ).to(self.device)

        self.cgf_module = CGFModule(
            hidden_dim=config["hidden_dim"],
            policy_net=self.policy_net
        ).to(self.device)

        # 根据配置选择使用传统ReflectionHead还是HPCR版本
        if config.get("enable_hpcr", False):
            self.reflection_head = HPCRPredictionHead(
                input_dim=config["state_dim"],
                output_dim=config["state_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=2,
                dropout=0.1,
                temperature=config.get("rce_temperature", 0.1)
            ).to(self.device)
            print("使用HPCR增强版反思头")
        else:
            self.reflection_head = ReflectionHead(
                rce_module=self.rce_module,
                cgf_module=self.cgf_module,
                gamma=config["gamma"]
            ).to(self.device)
            print("使用传统反思头")

        self.llm_traj_gen = LLMTrajectory(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            api_key=config["llm_api_key"],
            num_agents=config["num_agents"],
            device=self.device
        )

        # 根据任务类型选择优化器
        if config.get("task_type") == "vqa":
            self.optimizer = optim.Adam(
                list(self.rce_module.parameters()) +
                list(self.reflection_head.parameters()),
                lr=config["lr"]
            )
        else:
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.reflection_head.parameters()),
            lr=config["lr"]
    )

        # 环境初始化 - VQA模式下跳过
        if config.get("task_type") == "vqa":
            print("VQA模式：跳过环境初始化")
            self.env = None
        else:
            # self.env = VirtualHomeWrapper(config["executable_path"], no_graphics=True)
            print("环境初始化已注释")
            self.env = None
            
        self.best_eval_success_rate = -1.0

        # VQA数据集初始化
        if config.get("task_type") == "vqa":
            print("加载VQA数据集...")
            self.vqa_dataset = load_dataset("vqa_v2", split="train")
            # 限制数据量用于快速测试
            if config.get("max_vqa_samples"):
                self.vqa_dataset = self.vqa_dataset.select(range(config["max_vqa_samples"]))
            print(f"VQA数据集大小: {len(self.vqa_dataset)}")

    def train(self):
        if self.config.get("task_type") == "vqa":
            self.train_vqa()
        else:
            self.train_embodied()

    def train_vqa(self):
        """VQA训练模式"""
        print("开始VQA训练...")
        
        # 动态导入datasets
        try:
            from datasets import load_dataset
        except ImportError:
            print("请安装datasets: pip install datasets")
            return
        
        # 加载VQA数据集
        print("加载VQA数据集...")
        vqa_dataset = load_dataset("vqa_v2", split="train")
        if self.config.get("max_vqa_samples"):
            vqa_dataset = vqa_dataset.select(range(self.config["max_vqa_samples"]))
        print(f"VQA数据集大小: {len(vqa_dataset)}")
        
        for ep in range(1, self.config["num_episodes"] + 1):
            print(f"Epoch {ep}/{self.config['num_episodes']}")
            
            # 动态权重调度
            warmup_episodes = self.config.get("warmup_episodes", self.config["num_episodes"] / 10)
            progress = min(ep / warmup_episodes, 1.0)
            
            lambda_sinkhorn = self.config.get("lambda_sinkhorn_start", 0.0) + \
                              (self.config.get("lambda_sinkhorn_end", 0.1) - self.config.get("lambda_sinkhorn_start", 0.0)) * progress
            
            lambda_score = self.config.get("lambda_score_start", 0.0) + \
                           (self.config.get("lambda_score_end", 0.1) - self.config.get("lambda_score_start", 0.0)) * progress

            total_loss, loss_breakdown = self.train_one_vqa_episode(vqa_dataset, lambda_sinkhorn, lambda_score)
            
            if total_loss is not None:
                print(f"Episode {ep} - Total Loss: {total_loss.item():.4f}")
                for k, v in loss_breakdown.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"Episode {ep} - No valid loss")

            # --- Evaluation Step ---
            if ep % self.config.get("eval_interval", 5) == 0:
                eval_metrics = self.evaluate_vqa(vqa_dataset)
                print(f"\n--- VQA Evaluation at Episode {ep} ---")
                print(f"Accuracy: {eval_metrics['accuracy']:.2f}")
                print(f"Avg Loss: {eval_metrics['avg_loss']:.4f}")
                print("------------------------------------\n")

                # Save the model if it has the best accuracy so far
                if eval_metrics['accuracy'] > self.best_eval_success_rate:
                    self.best_eval_success_rate = eval_metrics['accuracy']
                    save_path = os.path.join(self.config.get("save_dir", "results/models"), "best_vqa_model.pth")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'rce_module': self.rce_module.state_dict(),
                        'reflection_head': self.reflection_head.state_dict(),
                    }, save_path)
                    print(f"New best VQA model saved to {save_path} with accuracy {self.best_eval_success_rate:.2f}")

    def train_embodied(self):
        """原有的embodied训练模式"""
        print("开始Embodied训练...")
        
        for ep in range(1, self.config["num_episodes"] + 1):
            # Dynamic weight scheduling
            warmup_episodes = self.config.get("warmup_episodes", self.config["num_episodes"] / 10)
            progress = min(ep / warmup_episodes, 1.0)
            
            lambda_sinkhorn = self.config.get("lambda_sinkhorn_start", 0.0) + \
                              (self.config.get("lambda_sinkhorn_end", 0.1) - self.config.get("lambda_sinkhorn_start", 0.0)) * progress
            
            lambda_score = self.config.get("lambda_score_start", 0.0) + \
                           (self.config.get("lambda_score_end", 0.1) - self.config.get("lambda_score_start", 0.0)) * progress

            total_loss, loss_breakdown = self.train_one_episode(lambda_sinkhorn, lambda_score)
            
            if total_loss is not None:
                print(f"Episode {ep} - Total Loss: {total_loss.item():.4f}")
                for k, v in loss_breakdown.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"Episode {ep} - No valid loss")

            # --- Evaluation Step ---
            if ep % self.config.get("eval_interval", 20) == 0:
                eval_metrics = self.evaluate()
                print(f"\n--- Evaluation at Episode {ep} ---")
                print(f"Success Rate: {eval_metrics['success_rate']:.2f}")
                print(f"Avg Episode Length: {eval_metrics['avg_ep_length']:.2f}")
                print(f"Avg Reward: {eval_metrics['avg_reward']:.2f}")
                print("------------------------------------\n")

                # Save the model if it has the best success rate so far
                if eval_metrics['success_rate'] > self.best_eval_success_rate:
                    self.best_eval_success_rate = eval_metrics['success_rate']
                    save_path = os.path.join(self.config.get("save_dir", "results/models"), "best_policy.pth")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.policy_net.state_dict(), save_path)
                    print(f"New best policy saved to {save_path} with success rate {self.best_eval_success_rate:.2f}")

        # if self.env:
        #     self.env.close()

    def train_one_vqa_episode(self, vqa_dataset, lambda_sinkhorn: float, lambda_score: float):
        """VQA单轮训练"""
        total_loss = torch.tensor(0.0, device=self.device)
        loss_breakdown = defaultdict(float)
        num_valid_samples = 0

        # 随机选择一批VQA样本
        batch_size = self.config.get("vqa_batch_size", 16)
        indices = np.random.choice(len(vqa_dataset), batch_size, replace=False)
        
        for idx in indices:
            sample = vqa_dataset[idx]
            
            try:
                # 获取图像和问题
                image = sample['image']
                question = sample['question']
                answer = sample['answers'][0]['answer']  # 使用第一个答案
                
                # 编码图像和文本
                image_emb = self.encoder.encode_image([image]).squeeze(0).to(self.device)
                question_emb = self.encoder.encode_text(question).squeeze(0).to(self.device)
                
                # 创建轨迹数据（模拟）
                trajectory_data = torch.cat([image_emb, question_emb]).unsqueeze(0)
                
                # 使用反思头进行训练
                # 这里我们使用问题作为prompt，图像作为状态
                prompt_global_embedding = question_emb
                prompt_token_embeddings = question_emb.unsqueeze(0).repeat(10, 1)
                
                # 生成对比轨迹（使用LLM）
                trajs_with_logprobs = self.llm_traj_gen.generate_contrastive_trajectories(
                    current_traj=trajectory_data,
                    agent_id=0,
                    level='low',
                    num_trajs=3,
                    env_description="VQA task: answering questions about images",
                    strategy_context=f"Question: {question}, Answer: {answer}"
                )
                
                if trajs_with_logprobs:
                    candidate_trajs = [item[0] for item in trajs_with_logprobs]
                    pos_trajs = [trajectory_data]
                    
                    # 计算损失
                    episode_loss, episode_breakdown = self.reflection_head(
                        anchor_traj=trajectory_data,
                        pos_trajs=pos_trajs,
                        neg_trajs=candidate_trajs,
                        prompt_token_embeddings=prompt_token_embeddings,
                        prompt_global_embedding=prompt_global_embedding,
                        sinkhorn_weight=lambda_sinkhorn,
                        score_field_weight=lambda_score,
                        neg_weights=torch.ones(len(candidate_trajs), device=self.device)
                    )
                    
                    if episode_loss is not None:
                        total_loss += episode_loss
                        for k, v in episode_breakdown.items():
                            loss_breakdown[f"vqa_{k}"] += v
                        num_valid_samples += 1
                        
            except Exception as e:
                print(f"Error processing VQA sample {idx}: {e}")
                continue

        if num_valid_samples > 0:
            avg_loss = total_loss / num_valid_samples
            for k in loss_breakdown:
                loss_breakdown[k] /= num_valid_samples
            
            # 反向传播
            avg_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return avg_loss, loss_breakdown
        else:
            return None, None

    def evaluate_vqa(self, vqa_dataset):
        """VQA评估"""
        self.rce_module.eval()
        self.reflection_head.eval()
        
        total_correct = 0
        total_loss = 0.0
        
        with torch.no_grad():
            # 使用验证集或随机选择样本
            indices = np.random.choice(len(vqa_dataset), 100, replace=False)
            
            for idx in indices:
                sample = vqa_dataset[idx]
                
                try:
                    image = sample['image']
                    question = sample['question']
                    answer = sample['answers'][0]['answer']
                    
                    # 编码
                    image_emb = self.encoder.encode_image([image]).squeeze(0).to(self.device)
                    question_emb = self.encoder.encode_text(question).squeeze(0).to(self.device)
                    
                    # 简单的相似度计算（这里可以改进）
                    similarity = F.cosine_similarity(image_emb, question_emb, dim=0)
                    
                    # 简单的评估逻辑（这里需要根据实际任务调整）
                    if similarity > 0.5:  # 阈值需要调整
                        total_correct += 1
                        
                except Exception as e:
                    continue
        
        self.rce_module.train()
        self.reflection_head.train()
        
        return {
            "accuracy": total_correct / 100,
            "avg_loss": total_loss / 100
        }

    def evaluate(self, num_eval_episodes: int = 10):
        """原有的embodied评估"""
        if not self.policy_net or not self.env:
            print("Embodied评估需要策略网络和环境")
            return {"success_rate": 0.0, "avg_ep_length": 0.0, "avg_reward": 0.0}
            
        self.policy_net.eval()
        
        total_successes = 0
        total_ep_length = 0
        total_reward = 0

        with torch.no_grad():
            for _ in range(num_eval_episodes):
                # obs = self.env.reset()
                # done_flags = {aid: False for aid in range(self.config["num_agents"])}
                # ep_reward = 0
                # ep_length = 0
                
                # for step in range(self.config["max_steps"]):
                #     for agent_name in self.env.agent_iter():
                #         if self.env.dones.get(agent_name, True):
                #             self.env.step(None)
                #             continue
                        
                #         agent_id = int(agent_name.split('_')[-1])
                #         raw_obs = obs
                        
                #         img = raw_obs.get('obs_0')
                #         state_emb = self.encoder.encode_image([img]).squeeze(0).to(self.device)
                        
                #         action, _ = self.policy_net.sample_action(state_emb, agent_id)
                        
                #         next_obs, reward, done, info = self.env.step(action.cpu().numpy())

                #         ep_reward += reward
                #         done_flags[agent_id] = done
                #         obs = next_obs
                #         if all(done_flags.values()):
                #             break
                    
                #     ep_length += 1
                #     if all(done_flags.values()):
                #         break
                
                # if all(done_flags.values()):
                #     total_successes += 1
                
                # total_ep_length += ep_length
                # total_reward += ep_reward
                pass  # 环境评估已注释

        self.policy_net.train()

        return {
            "success_rate": total_successes / num_eval_episodes,
            "avg_ep_length": total_ep_length / num_eval_episodes,
            "avg_reward": total_reward / num_eval_episodes
        }

    def train_one_episode(self, lambda_sinkhorn: float, lambda_score: float):
        """原有的embodied单轮训练"""
        if not self.env:
            print("Embodied训练需要环境")
            return None, None
            
        # --- Setup for the episode ---
        episode_buf = defaultdict(list)
        # obs = self.env.reset()
        # done_flags = {aid: False for aid in range(self.config["num_agents"])}

        # 为不同agent定义不同的prompt和职责
        agent_contexts = {
            0: {  # 主执行Agent
                "env_description": "A virtual home environment where the main agent executes primary tasks.",
                "strategy_context": "Execute core actions efficiently and precisely to complete the main objective."
            },
            1: {  # 辅助执行Agent
                "env_description": "A virtual home environment where the assistant agent supports main execution.",
                "strategy_context": "Prepare objects, clear paths, and support the main agent's actions."
            },
            2: {  # 监控Agent
                "env_description": "A virtual home environment where the monitoring agent observes and detects issues.",
                "strategy_context": "Monitor state changes, detect errors, and alert other agents of problems."
            },
            3: {  # 规划Agent
                "env_description": "A virtual home environment where the planning agent performs high-level reflection.",
                "strategy_context": "Analyze situations, perform HPCR reflection, and suggest strategic improvements."
            },
            4: {  # 对齐Agent
                "env_description": "A virtual home environment where the alignment agent ensures prompt-trajectory consistency.",
                "strategy_context": "Verify alignment between task description and execution, apply Sinkhorn and Score Field alignment."
            }
        }

        # --- Data Collection Loop ---
        # for _ in range(self.config["max_steps"]):
        #     for agent_name in self.env.agent_iter():
        #         if self.env.dones.get(agent_name, True):
        #             self.env.step(None)
        #             continue

        #         agent_id = int(agent_name.split('_')[-1])
        #         raw_obs = obs

        #         img = raw_obs.get('obs_0')
        #         state_emb = self.encoder.encode_image([img]).squeeze(0).to(self.device)

        #         # 根据agent_id选择不同的行为策略
        #         if agent_id == 2:  # 监控Agent - 主要观察，执行监控类动作
        #             action, _ = self.policy_net.sample_action(state_emb, agent_id)
        #             # 监控Agent倾向于执行观察类动作，如LOOKAT, FIND等
        #             # 动作选择由策略网络根据角色决定
        #         elif agent_id == 3:  # 规划Agent - 主要思考，执行规划类动作
        #             action, _ = self.policy_net.sample_action(state_emb, agent_id)
        #             # 规划Agent倾向于执行分析类动作，如TURNTO, LOOKAT等
        #             # 动作选择由策略网络根据角色决定
        #         elif agent_id == 4:  # 对齐Agent - 主要验证，执行验证类动作
        #             action, _ = self.policy_net.sample_action(state_emb, agent_id)
        #             # 对齐Agent倾向于执行验证类动作，如CHECK, VERIFY等
        #             # 动作选择由策略网络根据角色决定
        #         else:  # 主执行和辅助执行Agent - 正常执行
        #             action, _ = self.policy_net.sample_action(state_emb, agent_id)
                
        #         next_obs, reward, done, info = self.env.step(action.cpu().numpy())

        #         episode_buf[agent_id].append(MultiTransition(
        #             agent_id=agent_id, state=state_emb, action=action, 
        #             reward=torch.tensor(float(reward), device=self.device)
        #         ))

        #         done_flags[agent_id] = done
        #         obs = next_obs
        #         if all(done_flags.values()): break
        #     if all(done_flags.values()): break

        # --- Reflection and Update Step ---
        total_loss_for_ep = torch.tensor(0.0, device=self.device)
        loss_breakdown_for_ep = defaultdict(float)
        num_valid_agents = 0

        # for aid in range(self.config["num_agents"]):
        #     buf = episode_buf[aid]
        #     if len(buf) < self.config["sub_len"]:
        #         continue

        #     # 获取agent特定的上下文
        #     agent_context = agent_contexts.get(aid, agent_contexts[0])
        #     env_description = agent_context["env_description"]
        #     strategy_context = agent_context["strategy_context"]
            
        #     # 为对齐Agent (Agent 4) 添加特殊的对齐损失
        #     if aid == 4:
        #         # 对齐Agent专门负责Sinkhorn和Score Field对齐
        #         lambda_sinkhorn *= 2.0  # 增强对齐权重
        #         lambda_score *= 2.0

        #     # --- Hierarchical Reflection ---
        #     # A helper function to perform reflection for a given level
        #     def _perform_reflection(level: str, segment_tensor: torch.Tensor, segment_len: int, episode_step: int = 0):
        #         if len(buf) < segment_len: return None, None

        #         # 使用agent-specific prompt生成对比轨迹
        #         trajs_with_logprobs = self.llm_traj_gen.generate_contrastive_trajectories(
        #             current_traj=segment_tensor, agent_id=aid, level=level, num_trajs=5,
        #             env_description=env_description, strategy_context=strategy_context
        #         )
        #         if not trajs_with_logprobs: return None, None

        #         candidate_trajs = [item[0] for item in trajs_with_logprobs]
        #         logprobs_list = [item[1] for item in trajs_with_logprobs]
                
        #         pos_trajs = [segment_tensor]

        #         deviation_points = [
        #             next((i for i, lp in enumerate(lps) if lp.logprob < self.config["logprob_threshold"]), -1)
        #             for lps in logprobs_list
        #         ]
        #         neg_weights = torch.tensor([
        #             1.0 / (p + 2.0) if p != -1 else 1.0 for p in deviation_points
        #         ], device=self.device)

        #         # 如果使用HPCR，需要准备层级数据
        #         hierarchical_data = None
        #         if self.config.get("enable_hpcr", False) and hasattr(self.reflection_head, 'prepare_hierarchical_data'):
        #             # 将episode buffer转换为trajectory格式
        #             full_trajectory = []
        #             for transition in buf:
        #                 step_dict = {
        #                     'state': transition.state,
        #                     'action': transition.action,
        #                     'reward': transition.reward
        #                 }
        #                 full_trajectory.append(step_dict)
                    
        #             # 准备层级数据
        #             hierarchical_data = self.reflection_head.prepare_hierarchical_data(
        #                 full_trajectory, episode_step
        #             )

        #         # 为对齐Agent添加特殊的prompt embedding
        #         if aid == 4:
        #             # 对齐Agent使用特殊的prompt embedding
        #             alignment_prompt = "Ensure prompt-trajectory alignment through Sinkhorn and Score Field mechanisms"
        #             prompt_global_embedding = self.encoder.encode_text(alignment_prompt).squeeze()
        #             prompt_token_embeddings = prompt_global_embedding.unsqueeze(0).repeat(10, 1)
        #         else:
        #             # 其他Agent使用标准prompt
        #             full_prompt_text = f"{env_description} {strategy_context}"
        #             prompt_global_embedding = self.encoder.encode_text(full_prompt_text).squeeze()
        #             prompt_token_embeddings = prompt_global_embedding.unsqueeze(0).repeat(10, 1)

        #         level_loss, level_loss_breakdown = self.reflection_head(
        #             anchor_traj=segment_tensor,
        #             pos_trajs=pos_trajs,
        #             neg_trajs=candidate_trajs,
        #             prompt_token_embeddings=prompt_token_embeddings,
        #             prompt_global_embedding=prompt_global_embedding,
        #             sinkhorn_weight=lambda_sinkhorn,
        #             score_field_weight=lambda_score,
        #             neg_weights=neg_weights,
        #             level=level,
        #             hierarchical_data=hierarchical_data
        #         )
        #         return level_loss, level_loss_breakdown

        #     agent_total_loss = torch.tensor(0.0, device=self.device)
            
        #     # 根据agent角色调整反思策略
        #     if aid == 0:  # 主执行Agent - 所有级别的反思
        #         # 1. Low-level reflection (always)
        #         low_len = self.config["sub_len"]
        #         low_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-low_len:]])
        #         low_loss, low_breakdown = _perform_reflection('low', low_seg_tensor, low_len, len(buf))
        #         if low_loss is not None:
        #             weighted_loss = low_loss * self.config.get('lambda_contrastive_low', 1.0)
        #             agent_total_loss += weighted_loss
        #             for k, v in low_breakdown.items():
        #             loss_breakdown_for_ep[f"low_{k}"] += v

        #         # 2. Mid-level reflection (periodic)
        #         if ep % self.config.get("mid_level_interval", 5) == 0:
        #             mid_len = self.config.get("mid_len", 50)
        #             if len(buf) >= mid_len:
        #                 mid_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-mid_len:]])
        #                 mid_loss, mid_breakdown = _perform_reflection(
        #                     'mid', 
        #                     mid_seg_tensor, 
        #                     mid_len, 
        #                     len(buf),
        #                     subgoal_score_weight=self.config.get('lambda_score_mid', 0.1) # 传入中层分数权重
        #                 )
        #                 if mid_loss is not None:
        #                     weighted_loss = mid_loss * self.config.get('lambda_contrastive_mid', 0.5)
        #                     agent_total_loss += weighted_loss
        #                     for k, v in mid_breakdown.items():
        #                     loss_breakdown_for_ep[f"mid_{k}"] += v

        #         # 3. High-level reflection (periodic)
        #         if ep % self.config.get("high_level_interval", 10) == 0:
        #             high_len = len(buf) # Full episode
        #             high_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf])
        #             high_loss, high_breakdown = _perform_reflection('high', high_seg_tensor, high_len, len(buf))
        #             if high_loss is not None:
        #                 weighted_loss = high_loss * self.config.get('lambda_contrastive_high', 0.2)
        #                 agent_total_loss += weighted_loss
        #                         for k, v in high_breakdown.items():
        #                     loss_breakdown_for_ep[f"high_{k}"] += v

        #     elif aid == 3:  # 规划Agent - 主要进行中高级反思
        #         # 主要进行mid-level和high-level反思
        #         if ep % self.config.get("mid_level_interval", 5) == 0:
        #             mid_len = self.config.get("mid_len", 50)
        #             if len(buf) >= mid_len:
        #                 mid_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-mid_len:]])
        #                 mid_loss, mid_breakdown = _perform_reflection('mid', mid_seg_tensor, mid_len, len(buf))
        #                 if mid_loss is not None:
        #                     weighted_loss = mid_loss * self.config.get('lambda_contrastive_mid', 1.0)  # 增强权重
        #                     agent_total_loss += weighted_loss
        #                     for k, v in mid_breakdown.items():
        #                     loss_breakdown_for_ep[f"planning_mid_{k}"] += v

        #             if ep % self.config.get("high_level_interval", 10) == 0:
        #                 high_len = len(buf)
        #                 high_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf])
        #                 high_loss, high_breakdown = _perform_reflection('high', high_seg_tensor, high_len, len(buf))
        #                 if high_loss is not None:
        #                     weighted_loss = high_loss * self.config.get('lambda_contrastive_high', 1.0)  # 增强权重
        #                     agent_total_loss += weighted_loss
        #                     for k, v in high_breakdown.items():
        #                     loss_breakdown_for_ep[f"planning_high_{k}"] += v

        #     elif aid == 4:  # 对齐Agent - 专门负责对齐损失
        #         # 对齐Agent主要关注对齐相关的损失
        #         low_len = self.config["sub_len"]
        #         low_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-low_len:]])
        #         low_loss, low_breakdown = _perform_reflection('low', low_seg_tensor, low_len, len(buf))
        #         if low_loss is not None:
        #             # 对齐Agent的对齐损失权重更高
        #             weighted_loss = low_loss * self.config.get('lambda_contrastive_low', 2.0)
        #             agent_total_loss += weighted_loss
        #             for k, v in low_breakdown.items():
        #             loss_breakdown_for_ep[f"alignment_{k}"] += v

        #     else:  # 辅助执行和监控Agent - 标准反思
        #         # 1. Low-level reflection (always)
        #         low_len = self.config["sub_len"]
        #         low_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-low_len:]])
        #         low_loss, low_breakdown = _perform_reflection('low', low_seg_tensor, low_len, len(buf))
        #         if low_loss is not None:
        #             weighted_loss = low_loss * self.config.get('lambda_contrastive_low', 0.8)  # 稍微降低权重
        #             agent_total_loss += weighted_loss
        #             for k, v in low_breakdown.items():
        #             loss_breakdown_for_ep[f"low_{k}"] += v

        #     if agent_total_loss.item() > 0:
        #         total_loss_for_ep += agent_total_loss
        #         num_valid_agents += 1

        # if num_valid_agents > 0:
        #     avg_total_loss = total_loss_for_ep / num_valid_agents
        #     for k in loss_breakdown_for_ep:
        #         loss_breakdown_for_ep[k] /= num_valid_agents
        #     return avg_total_loss, loss_breakdown_for_ep
        # else:
        #     return None, None
        
        print("Embodied训练已注释")
            return None, None


def main():
    import yaml
    
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "rsn_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override API key from environment if available
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        config["llm_api_key"] = api_key
    elif not config.get("llm_api_key"):
        print("Warning: No API key found in config or environment. Using placeholder.")
        config["llm_api_key"] = "YOUR_FALLBACK_API_KEY_HERE"
    
    # Ensure device is set
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"配置加载完成:")
    print(f"- 使用设备: {config['device']}")
    print(f"- HPCR模式: {'启用' if config.get('enable_hpcr', False) else '禁用'}")
    print(f"- 训练轮数: {config['num_episodes']}")
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()