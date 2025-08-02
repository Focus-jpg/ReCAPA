import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import openai
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class HPCRPredictionHead(nn.Module):
    """
    HPCR预测头：实现层级间的预测式对比学习
    在层级 l 中，预测下一级（l+1）子轨迹表示，并通过对比学习最大化预测准确度
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        # 构建预测网络 f_θ
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.prediction_network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z_current: torch.Tensor) -> torch.Tensor:
        """
        预测下一层级的表示
        :param z_current: 当前层级的轨迹表示, shape (batch_size, input_dim) or (input_dim,)
        :return: 预测的下一层级表示, shape (batch_size, output_dim) or (output_dim,)
        """
        return self.prediction_network(z_current)
    
    def compute_predictive_infonce_loss(
        self,
        z_current: torch.Tensor,
        z_next_positive: torch.Tensor,
        z_next_negatives: List[torch.Tensor],
        negative_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算预测式InfoNCE损失
        
        L_pred^l = -E[log(exp(<f_θ(z^l), z^{l+1}>/τ) / 
                       Σ_tilde{z^{l+1}} exp(<f_θ(z^l), tilde{z^{l+1}}>/τ))]
        
        :param z_current: 当前层级表示 z^l, shape (batch_size, input_dim)
        :param z_next_positive: 真实下一层级表示 z^{l+1}, shape (batch_size, output_dim)  
        :param z_next_negatives: 负样本列表，每个shape (batch_size, output_dim)
        :param negative_weights: 负样本权重，shape (num_negatives,)
        :return: 预测式InfoNCE损失
        """
        # 预测下一层级表示
        z_pred = self.forward(z_current)  # (batch_size, output_dim)
        
        # 计算正样本相似度
        pos_sim = F.cosine_similarity(z_pred, z_next_positive, dim=-1)  # (batch_size,)
        pos_logits = pos_sim / self.temperature
        
        # 计算负样本相似度
        neg_logits_list = []
        for neg_sample in z_next_negatives:
            neg_sim = F.cosine_similarity(z_pred, neg_sample, dim=-1)  # (batch_size,)
            neg_logits_list.append(neg_sim / self.temperature)
        
        if not neg_logits_list:
            return torch.tensor(0.0, device=z_current.device, requires_grad=True)
        
        neg_logits = torch.stack(neg_logits_list, dim=1)  # (batch_size, num_negatives)
        
        # 应用负样本权重
        if negative_weights is not None:
            neg_logits = neg_logits * negative_weights.unsqueeze(0)  # (batch_size, num_negatives)
        
        # 计算InfoNCE损失
        numerator = torch.exp(pos_logits)  # (batch_size,)
        denominator = numerator + torch.exp(neg_logits).sum(dim=1)  # (batch_size,)
        
        loss = -torch.log(numerator / (denominator + 1e-8))  # (batch_size,)
        
        return loss.mean()


class HPCRFailureSampleGenerator:
    """
    HPCR失败样本生成器：使用GPT-4生成失败子轨迹作为hard negatives
    """
    def __init__(
        self,
        llm_api_key: str,
        llm_api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        max_retries: int = 3,
        encoder=None  # NomicEncoder实例
    ):
        self.api_key = llm_api_key
        self.api_base = llm_api_base
        self.model_name = model_name
        self.max_retries = max_retries
        self.encoder = encoder
        
        # 初始化OpenAI客户端
        if self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            self.client = None
            print("警告: 未提供API密钥，失败样本生成器将使用模拟模式")
    
    def generate_failure_trajectories(
        self,
        successful_trajectory: List[Dict],
        task_description: str = "",
        environment_context: str = "",
        num_failures: int = 3,
        failure_types: List[str] = None,
        level: str = "mid"
    ) -> List[Tuple[List[Dict], torch.Tensor]]:
        """
        基于成功轨迹生成失败版本，并提取embedding
        
        :param successful_trajectory: 成功的轨迹段
        :param task_description: 任务描述
        :param environment_context: 环境上下文
        :param num_failures: 生成失败样本数量
        :param failure_types: 失败类型 ['action_error', 'timing_error', 'logic_error', 'sequence_error']
        :param level: 层级 ('low', 'mid', 'high')
        :return: 失败轨迹和embedding的元组列表 [(trajectory, embedding), ...]
        """
        if failure_types is None:
            failure_types = ['action_error', 'timing_error', 'logic_error', 'sequence_error']
        
        if not self.client:
            print("使用模拟模式生成失败轨迹")
            return self._generate_simulated_failures(successful_trajectory, num_failures, failure_types)
        
        failure_trajectories_with_embeddings = []
        
        # 并行生成失败轨迹
        with ThreadPoolExecutor(max_workers=num_failures) as executor:
            future_to_type = {
                executor.submit(
                    self._generate_single_failure_with_api,
                    successful_trajectory,
                    failure_type,
                    task_description,
                    environment_context,
                    level
                ): failure_type for failure_type in failure_types[:num_failures]
            }
            
            for future in as_completed(future_to_type):
                failure_type = future_to_type[future]
                try:
                    result = future.result()
                    if result:
                        failure_trajectories_with_embeddings.append(result)
                except Exception as e:
                    print(f"生成{failure_type}失败轨迹时出错: {e}")
                    continue
        
        return failure_trajectories_with_embeddings
    
    def _generate_single_failure_with_api(
        self,
        success_traj: List[Dict],
        failure_type: str,
        task_description: str,
        environment_context: str,
        level: str
    ) -> Optional[Tuple[List[Dict], torch.Tensor]]:
        """
        使用GPT-4 API生成单个失败轨迹
        """
        prompt = self._build_failure_generation_prompt(
            success_traj, failure_type, task_description, environment_context, level
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个专业的机器人任务规划专家，专门生成失败的任务轨迹。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                failure_trajectory = self._parse_api_response(response.choices[0].message.content)
                if failure_trajectory:
                    # 提取embedding
                    embedding = self._extract_trajectory_embedding(failure_trajectory)
                    return failure_trajectory, embedding
                
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue
        
        return None
    
    def _build_failure_generation_prompt(
        self,
        success_traj: List[Dict],
        failure_type: str,
        task_description: str,
        environment_context: str,
        level: str
    ) -> str:
        """
        构建失败轨迹生成的prompt
        """
        # 将轨迹转换为可读格式
        trajectory_text = self._trajectory_to_text(success_traj)
        
        failure_type_descriptions = {
            'action_error': '动作错误：使用错误的动作或动作参数',
            'timing_error': '时序错误：动作执行顺序错误或时机不当',
            'logic_error': '逻辑错误：任务逻辑错误或目标理解错误',
            'sequence_error': '序列错误：缺少关键步骤或步骤重复'
        }
        
        failure_desc = failure_type_descriptions.get(failure_type, failure_type)
        
        prompt = f"""
任务描述: {task_description}
环境上下文: {environment_context}
层级: {level}

成功轨迹:
{trajectory_text}

请生成一个包含"{failure_desc}"的失败轨迹。失败轨迹应该：
1. 保持与成功轨迹相似的结构
2. 在关键点引入指定的错误类型
3. 确保错误是合理的，不是随机错误
4. 返回JSON格式的轨迹数据

请只返回JSON格式的轨迹数据，不要包含其他解释。
"""
        return prompt
    
    def _trajectory_to_text(self, trajectory: List[Dict]) -> str:
        """
        将轨迹转换为可读的文本格式
        """
        text_lines = []
        for i, step in enumerate(trajectory):
            if isinstance(step, dict):
                action = step.get('action', 'unknown')
                state = step.get('state', 'unknown')
                reward = step.get('reward', 0)
                text_lines.append(f"步骤{i+1}: 动作={action}, 状态={state}, 奖励={reward}")
            else:
                text_lines.append(f"步骤{i+1}: {step}")
        return "\n".join(text_lines)
    
    def _parse_api_response(self, response_text: str) -> Optional[List[Dict]]:
        """
        解析API响应，提取轨迹数据
        """
        try:
            # 尝试提取JSON部分
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                trajectory = json.loads(json_str)
                return trajectory
            else:
                # 如果没有找到JSON，尝试解析文本格式
                return self._parse_text_trajectory(response_text)
                
        except json.JSONDecodeError:
            print("JSON解析失败，尝试文本解析")
            return self._parse_text_trajectory(response_text)
        except Exception as e:
            print(f"解析API响应失败: {e}")
            return None
    
    def _parse_text_trajectory(self, text: str) -> Optional[List[Dict]]:
        """
        解析文本格式的轨迹
        """
        try:
            lines = text.strip().split('\n')
            trajectory = []
            
            for line in lines:
                if '步骤' in line or 'step' in line.lower():
                    # 解析步骤行
                    step_dict = self._parse_step_line(line)
                    if step_dict:
                        trajectory.append(step_dict)
            
            return trajectory if trajectory else None
            
        except Exception as e:
            print(f"文本轨迹解析失败: {e}")
            return None
    
    def _parse_step_line(self, line: str) -> Optional[Dict]:
        """
        解析单个步骤行
        """
        try:
            # 简单的解析逻辑，可以根据需要扩展
            if '动作=' in line and '状态=' in line:
                parts = line.split(',')
                step_dict = {}
                
                for part in parts:
                    if '动作=' in part:
                        step_dict['action'] = part.split('=')[1].strip()
                    elif '状态=' in part:
                        step_dict['state'] = part.split('=')[1].strip()
                    elif '奖励=' in part:
                        step_dict['reward'] = float(part.split('=')[1].strip())
                
                return step_dict
            return None
            
        except Exception:
            return None
    
    def _extract_trajectory_embedding(self, trajectory: List[Dict]) -> torch.Tensor:
        """
        提取轨迹的embedding
        """
        if not self.encoder:
            # 如果没有编码器，返回随机embedding
            return torch.randn(512)  # 假设embedding维度为512
        
        try:
            # 将轨迹转换为文本
            trajectory_text = self._trajectory_to_text(trajectory)
            
            # 使用NomicEncoder编码
            with torch.no_grad():
                embedding = self.encoder.encode_text(trajectory_text)
                return embedding.squeeze()
                
        except Exception as e:
            print(f"提取轨迹embedding失败: {e}")
            return torch.randn(512)  # 返回随机embedding作为fallback
    
    def _generate_simulated_failures(
        self,
        success_traj: List[Dict],
        num_failures: int,
        failure_types: List[str]
    ) -> List[Tuple[List[Dict], torch.Tensor]]:
        """
        模拟模式：生成简化的失败轨迹
        """
        failure_trajectories = []
        
        for i in range(num_failures):
            failure_type = failure_types[i % len(failure_types)]
            failure_traj = self._generate_simulated_single_failure(success_traj, failure_type)
            if failure_traj:
                # 生成随机embedding
                embedding = torch.randn(512)
                failure_trajectories.append((failure_traj, embedding))
        
        return failure_trajectories
    
    def _generate_simulated_single_failure(
        self,
        success_traj: List[Dict],
        failure_type: str
    ) -> Optional[List[Dict]]:
        """
        生成模拟的单个失败轨迹
        """
        failure_traj = success_traj.copy()
        
        if failure_type == 'action_error':
            # 随机替换某个动作
            if len(failure_traj) > 1:
                idx = torch.randint(0, len(failure_traj), (1,)).item()
                if 'action' in failure_traj[idx]:
                    failure_traj[idx]['action'] = 'WRONG_ACTION'
        
        elif failure_type == 'timing_error':
            # 时序错误：打乱某些步骤的顺序
            if len(failure_traj) > 2:
                i, j = torch.randperm(len(failure_traj))[:2].tolist()
                failure_traj[i], failure_traj[j] = failure_traj[j], failure_traj[i]
        
        elif failure_type == 'logic_error':
            # 逻辑错误：添加错误的步骤
            if len(failure_traj) > 1:
                wrong_step = {'action': 'ILLOGICAL_ACTION', 'state': 'WRONG_STATE', 'reward': -1.0}
                failure_traj.insert(len(failure_traj)//2, wrong_step)
        
        elif failure_type == 'sequence_error':
            # 序列错误：删除或重复某些步骤
            if len(failure_traj) > 1:
                idx = torch.randint(0, len(failure_traj), (1,)).item()
                if torch.rand(1).item() > 0.5:
                    failure_traj.pop(idx)
                else:
                    failure_traj.insert(idx, failure_traj[idx].copy())
        
        return failure_traj


class HPCRHierarchicalSlicing:
    """
    HPCR层级轨迹切片器：实现多层级轨迹切分
    """
    def __init__(
        self,
        low_span: int = 10,
        mid_span: int = 50,
        high_span: int = -1  # -1表示全episode
    ):
        self.low_span = low_span
        self.mid_span = mid_span
        self.high_span = high_span
    
    def slice_trajectory(
        self,
        full_trajectory: List[Dict],
        level: str
    ) -> List[List[Dict]]:
        """
        根据层级切分轨迹
        
        :param full_trajectory: 完整轨迹
        :param level: 层级 ('low', 'mid', 'high')
        :return: 切分后的子轨迹列表
        """
        if level == 'low':
            span = self.low_span
        elif level == 'mid':
            span = self.mid_span
        elif level == 'high':
            span = self.high_span if self.high_span > 0 else len(full_trajectory)
        else:
            raise ValueError(f"Unknown level: {level}")
        
        if span >= len(full_trajectory):
            return [full_trajectory]
        
        # 滑动窗口切分
        slices = []
        for i in range(0, len(full_trajectory) - span + 1, span):
            slice_traj = full_trajectory[i:i + span]
            slices.append(slice_traj)
        
        # 确保最后一个片段被包含
        if len(full_trajectory) % span != 0:
            slices.append(full_trajectory[-(span):])
        
        return slices 