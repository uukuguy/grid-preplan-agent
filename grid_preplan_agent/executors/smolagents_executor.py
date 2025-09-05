"""Smolagents执行器实现"""

from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime

try:
    from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
    from smolagents.tools import Tool as SmolagentsTool
except ImportError:
    # 如果smolagents不可用，创建mock实现
    print("Warning: smolagents not available, using mock implementation")
    
    class SmolagentsTool:
        def __init__(self, name, description): 
            self.name = name
            self.description = description
    
    class CodeAgent:
        def __init__(self, tools, model, **kwargs): 
            self.tools = tools
            self.model = model
        
        def run(self, task): 
            return f"Mock execution result for: {task}"
    
    class LiteLLMModel:
        def __init__(self, model_id): 
            self.model_id = model_id

from .base_executor import BaseExecutor, ExecutorError
from ..core.models import PlanJSON, PlanStep, ExecutionState, ExecutionResult, StepType
from ..tools.api_registry import tool_registry, call_tool
from ..utils.logger import logger


class GridTool(SmolagentsTool):
    """适配电网工具到Smolagents的工具包装器"""
    
    def __init__(self, tool_name: str, tool_instance):
        self.tool_name = tool_name
        self.tool_instance = tool_instance
        
        super().__init__(
            name=tool_name,
            description=tool_instance.description
        )
    
    def __call__(self, **kwargs):
        """同步调用工具"""
        # 创建异步任务并运行
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.tool_instance.execute(**kwargs))
        except RuntimeError:
            # 如果没有运行的事件循环，创建新的
            return asyncio.run(self.tool_instance.execute(**kwargs))


class SmolagentsExecutor(BaseExecutor):
    """Smolagents执行器"""
    
    def __init__(self, model_id: str = "gpt-4-turbo-preview"):
        super().__init__("Smolagents")
        self.model_id = model_id
        self.agents: Dict[str, CodeAgent] = {}
    
    async def execute(
        self, 
        plan: PlanJSON, 
        scenario: str,
        inputs: Dict[str, Any]
    ) -> ExecutionResult:
        """执行预案
        
        Args:
            plan: 预案JSON对象
            scenario: 场景描述
            inputs: 输入参数
            
        Returns:
            ExecutionResult: 执行结果
        """
        logger.info(f"Smolagents开始执行预案: {plan.plan_id}")
        start_time = time.time()
        
        try:
            # 验证输入
            self.validate_inputs(plan, inputs)
            
            # 创建执行状态
            state = self.create_execution_state(plan, scenario, inputs)
            state.status = "running"
            
            # 获取或创建Agent
            agent = await self.get_or_create_agent(plan)
            
            # 构建执行任务描述
            task_description = self.build_task_description(plan, scenario, inputs)
            
            # 执行任务
            result = await self.execute_with_agent(agent, task_description, plan, state)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Smolagents执行成功: {plan.plan_id}")
            return self.create_execution_result(
                state=state,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Smolagents执行异常: {plan.plan_id}, 错误: {str(e)}")
            
            state = self.create_execution_state(plan, scenario, inputs)
            return self.create_execution_result(
                state=state,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def get_or_create_agent(self, plan: PlanJSON) -> CodeAgent:
        """获取或创建Agent
        
        Args:
            plan: 预案对象
            
        Returns:
            CodeAgent: Smolagents代理
        """
        # 检查是否已缓存
        if plan.plan_id in self.agents:
            return self.agents[plan.plan_id]
        
        # 创建新的Agent
        agent = await self.create_agent(plan)
        self.agents[plan.plan_id] = agent
        
        return agent
    
    async def create_agent(self, plan: PlanJSON) -> CodeAgent:
        """创建Agent
        
        Args:
            plan: 预案对象
            
        Returns:
            CodeAgent: 新创建的代理
        """
        logger.info(f"创建Smolagents Agent: {plan.plan_id}")
        
        # 收集预案需要的工具
        tools = []
        required_tools = set()
        
        # 从步骤中提取工具需求
        for step in plan.steps:
            if step.type == StepType.RAG:
                required_tools.add("mock_rag_query")
            elif step.type == StepType.TOOL:
                if step.tool_name:
                    required_tools.add(step.tool_name)
            elif step.type == StepType.COMPUTE:
                required_tools.add("mock_calculator")
        
        # 创建工具实例
        for tool_name in required_tools:
            tool_instance = tool_registry.get_tool(tool_name)
            if tool_instance:
                grid_tool = GridTool(tool_name, tool_instance)
                tools.append(grid_tool)
            else:
                logger.warning(f"未找到工具: {tool_name}")
        
        # 创建模型
        model = LiteLLMModel(model_id=self.model_id)
        
        # 创建Agent
        agent = CodeAgent(
            tools=tools,
            model=model,
            additional_authorized_imports=["json", "math", "datetime"],
            max_steps=20
        )
        
        return agent
    
    def build_task_description(
        self, 
        plan: PlanJSON, 
        scenario: str,
        inputs: Dict[str, Any]
    ) -> str:
        """构建任务描述
        
        Args:
            plan: 预案对象
            scenario: 场景描述
            inputs: 输入参数
            
        Returns:
            str: 任务描述
        """
        task_lines = [
            f"执行电网调度预案: {plan.title}",
            f"场景描述: {scenario}",
            "",
            "输入参数:"
        ]
        
        for key, value in inputs.items():
            task_lines.append(f"- {key}: {value}")
        
        task_lines.extend([
            "",
            "执行步骤:"
        ])
        
        for i, step in enumerate(plan.steps, 1):
            task_lines.append(f"{i}. {step.description}")
            if step.type == StepType.RAG:
                task_lines.append(f"   - 查询内容: {step.query}")
            elif step.type == StepType.TOOL:
                task_lines.append(f"   - 调用工具: {step.tool_name}")
                task_lines.append(f"   - 工具输入: {step.inputs}")
            elif step.type == StepType.COMPUTE:
                task_lines.append(f"   - 计算公式: {step.formula}")
                task_lines.append(f"   - 计算输入: {step.inputs}")
            
            task_lines.append(f"   - 期望输出: {', '.join(step.outputs)}")
            task_lines.append("")
        
        task_lines.extend([
            "要求:",
            "1. 按照步骤顺序执行",
            "2. 使用提供的工具获取数据和执行计算",
            "3. 记录每个步骤的执行结果",
            "4. 最终返回所有计算结果，特别是:",
            f"   {', '.join(plan.plan_outputs)}",
            "",
            "请开始执行预案。"
        ])
        
        return "\n".join(task_lines)
    
    async def execute_with_agent(
        self,
        agent: CodeAgent,
        task_description: str,
        plan: PlanJSON,
        state: ExecutionState
    ) -> Dict[str, Any]:
        """使用Agent执行任务
        
        Args:
            agent: Smolagents代理
            task_description: 任务描述
            plan: 预案对象
            state: 执行状态
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            # 执行任务
            result = agent.run(task_description)
            
            # 解析结果（这里简化处理，实际应该解析Agent的详细输出）
            if isinstance(result, str):
                # 尝试从结果字符串中提取关键信息
                final_outputs = self.parse_agent_result(result, plan.plan_outputs)
            else:
                final_outputs = result
            
            # 更新状态
            state.final_outputs = final_outputs
            state.status = "completed"
            
            # 添加步骤历史（简化版）
            step_result = {
                "step_id": "smolagents_execution",
                "step_type": "agent_execution",
                "description": "Smolagents代理执行",
                "success": True,
                "outputs": final_outputs,
                "timestamp": datetime.now().isoformat()
            }
            state.step_history = [step_result]
            
            return final_outputs
            
        except Exception as e:
            logger.error(f"Agent执行失败: {str(e)}")
            state.status = "failed"
            state.error_message = str(e)
            raise ExecutorError(f"Agent执行失败: {str(e)}")
    
    def parse_agent_result(
        self, 
        result_text: str, 
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """解析Agent结果文本
        
        Args:
            result_text: Agent返回的结果文本
            expected_outputs: 期望的输出变量列表
            
        Returns:
            Dict[str, Any]: 解析后的输出字典
        """
        import re
        
        outputs = {}
        
        # 尝试提取数值结果
        for output_var in expected_outputs:
            # 查找形如 "P_max_device = 2800" 的模式
            pattern = rf"{re.escape(output_var)}\s*[=:]\s*([0-9]+\.?[0-9]*)"
            match = re.search(pattern, result_text)
            
            if match:
                try:
                    outputs[output_var] = float(match.group(1))
                except ValueError:
                    outputs[output_var] = match.group(1)
            else:
                # 如果找不到特定变量，尝试提取任何数值
                numbers = re.findall(r'\b\d+\.?\d*\b', result_text)
                if numbers:
                    outputs[output_var] = float(numbers[-1])  # 使用最后一个数值
                else:
                    outputs[output_var] = None
        
        # 如果没有找到任何输出，使用默认值
        if not any(v is not None for v in outputs.values()):
            for output_var in expected_outputs:
                outputs[output_var] = f"未找到{output_var}的值"
        
        return outputs
    
    def clear_agent_cache(self) -> None:
        """清空Agent缓存"""
        self.agents.clear()
        logger.info("Smolagents缓存已清空")
    
    async def list_available_tools(self) -> List[str]:
        """列出可用的工具
        
        Returns:
            List[str]: 工具名称列表
        """
        return list(tool_registry.list_tools().keys())
    
    def get_agent_info(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent信息
        
        Args:
            plan_id: 预案ID
            
        Returns:
            Dict[str, Any]: Agent信息
        """
        if plan_id not in self.agents:
            return None
        
        agent = self.agents[plan_id]
        return {
            "plan_id": plan_id,
            "model_id": self.model_id,
            "tools_count": len(agent.tools) if hasattr(agent, 'tools') else 0,
            "created_at": datetime.now().isoformat()
        }