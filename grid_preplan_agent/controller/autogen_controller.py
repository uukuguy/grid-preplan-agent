"""AutoGen控制器实现"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import json
from datetime import datetime
from pathlib import Path

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    from autogen.coding import LocalCommandLineCodeExecutor
except ImportError:
    # 如果autogen不可用，创建mock实现
    print("Warning: autogen not available, using mock implementation")
    
    class ConversableAgent:
        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs
        
        def generate_reply(self, messages=None, **kwargs):
            return f"Mock reply from {self.name}"
    
    class GroupChat:
        def __init__(self, agents, **kwargs):
            self.agents = agents
    
    class GroupChatManager:
        def __init__(self, groupchat, **kwargs):
            self.groupchat = groupchat

from ..core.models import PlanJSON, ExecutionResult, ExecutionState, DecisionReport
from ..core.plan_parser import PlanParser, create_parser
from ..executors.langgraph_executor import LangGraphExecutor
from ..executors.smolagents_executor import SmolagentsExecutor
from ..tools.api_registry import tool_registry
from .complexity_analyzer import ComplexityAnalyzer, ComplexityLevel
from ..utils.logger import logger


class AutoGenController:
    """AutoGen中控制器"""
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        plan_library_path: Optional[Path] = None
    ):
        """初始化控制器
        
        Args:
            llm_config: LLM配置
            plan_library_path: 预案库路径
        """
        self.llm_config = llm_config or self._default_llm_config()
        self.plan_library_path = plan_library_path or Path("plans")
        
        # 初始化组件
        self.plan_parser = create_parser()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.langgraph_executor = LangGraphExecutor()
        self.smolagents_executor = SmolagentsExecutor()
        
        # 缓存
        self.plan_cache: Dict[str, PlanJSON] = {}
        self.execution_history: Dict[str, ExecutionResult] = {}
        
        # 创建AutoGen Agents
        self._setup_agents()
    
    def _default_llm_config(self) -> Dict[str, Any]:
        """默认LLM配置"""
        return {
            "model": "gpt-4-turbo-preview",
            "api_type": "openai",
            "temperature": 0.1,
            "timeout": 60
        }
    
    def _setup_agents(self):
        """设置AutoGen智能体"""
        # 主控Agent
        self.controller_agent = ConversableAgent(
            name="GridController",
            system_message="""你是电网调度智能控制器。
            主要职责：
            1. 解析用户输入的场景和需求
            2. 选择合适的预案
            3. 分析预案复杂度
            4. 路由到合适的执行器
            5. 汇总执行结果
            
            请保持专业、准确、高效。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # 预案专家Agent
        self.plan_expert = ConversableAgent(
            name="PlanExpert", 
            system_message="""你是电网调度预案专家。
            专长：
            1. 预案选择和匹配
            2. 预案内容理解
            3. 预案参数提取
            4. 预案适用性判断
            
            请根据场景描述选择最合适的预案。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # 执行专家Agent
        self.execution_expert = ConversableAgent(
            name="ExecutionExpert",
            system_message="""你是预案执行专家。
            专长：
            1. 执行器选择
            2. 执行参数配置  
            3. 执行结果分析
            4. 异常处理建议
            
            请确保预案能够正确执行。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # 决策分析Agent
        self.decision_analyst = ConversableAgent(
            name="DecisionAnalyst",
            system_message="""你是电网调度决策分析专家。
            专长：
            1. 执行结果分析
            2. 决策报告生成
            3. 风险评估
            4. 建议措施制定
            
            请生成专业的决策分析报告。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
    
    async def process_scenario(
        self,
        scenario: str,
        inputs: Optional[Dict[str, Any]] = None,
        preferred_executor: Optional[str] = None
    ) -> ExecutionResult:
        """处理场景请求
        
        Args:
            scenario: 场景描述
            inputs: 输入参数
            preferred_executor: 首选执行器
            
        Returns:
            ExecutionResult: 执行结果
        """
        logger.info(f"处理场景: {scenario}")
        
        try:
            # 1. 选择预案
            plan = await self.select_plan(scenario)
            if not plan:
                raise ValueError(f"未找到适合的预案: {scenario}")
            
            # 2. 分析复杂度
            complexity, analysis = self.complexity_analyzer.analyze(plan)
            logger.info(f"预案复杂度: {complexity.value}")
            
            # 3. 选择执行器
            executor = self.route_to_executor(complexity, preferred_executor)
            
            # 4. 准备输入参数
            processed_inputs = await self.prepare_execution_inputs(plan, scenario, inputs or {})
            
            # 5. 执行预案
            result = await executor.execute(plan, scenario, processed_inputs)
            
            # 6. 记录执行历史
            self.execution_history[result.execution_id] = result
            
            logger.info(f"场景处理完成: {scenario}, 结果: {'成功' if result.success else '失败'}")
            return result
            
        except Exception as e:
            logger.error(f"场景处理失败: {scenario}, 错误: {str(e)}")
            
            # 创建失败结果
            dummy_state = ExecutionState(
                plan_id="unknown",
                execution_id=f"failed_{datetime.now().isoformat()}",
                scenario=scenario,
                inputs=inputs or {}
            )
            
            from ..executors.base_executor import BaseExecutor
            dummy_executor = BaseExecutor("dummy")
            return dummy_executor.create_execution_result(
                state=dummy_state,
                success=False,
                execution_time=0.0,
                error_message=str(e)
            )
    
    async def select_plan(self, scenario: str) -> Optional[PlanJSON]:
        """选择预案
        
        Args:
            scenario: 场景描述
            
        Returns:
            PlanJSON: 选择的预案，如果未找到返回None
        """
        # 简化的预案选择逻辑
        # 实际项目中应该使用更复杂的匹配算法
        
        available_plans = await self.load_available_plans()
        
        # 关键词匹配
        scenario_lower = scenario.lower()
        
        for plan in available_plans:
            # 检查标题和描述中的关键词
            plan_text = f"{plan.title} {plan.description}".lower()
            
            # 直流限额相关
            if any(keyword in scenario_lower for keyword in ["直流", "限额", "故障"]):
                if any(keyword in plan_text for keyword in ["直流", "限额"]):
                    logger.info(f"选择预案: {plan.title}")
                    return plan
            
            # 其他预案匹配规则可以在这里添加
        
        # 如果没有找到匹配的预案，返回默认预案或None
        if available_plans:
            logger.warning(f"未找到精确匹配的预案，使用默认预案: {available_plans[0].title}")
            return available_plans[0]
        
        return None
    
    async def load_available_plans(self) -> List[PlanJSON]:
        """加载可用的预案
        
        Returns:
            List[PlanJSON]: 预案列表
        """
        plans = []
        
        if not self.plan_library_path.exists():
            logger.warning(f"预案库路径不存在: {self.plan_library_path}")
            return plans
        
        # 扫描预案文件
        for plan_file in self.plan_library_path.glob("*.txt"):
            try:
                # 检查缓存
                if plan_file.stem in self.plan_cache:
                    plans.append(self.plan_cache[plan_file.stem])
                    continue
                
                # 解析预案文件
                plan = self.plan_parser.parse_file(plan_file)
                self.plan_cache[plan.plan_id] = plan
                plans.append(plan)
                
            except Exception as e:
                logger.error(f"加载预案失败: {plan_file}, 错误: {str(e)}")
                continue
        
        return plans
    
    def route_to_executor(
        self,
        complexity: ComplexityLevel,
        preferred_executor: Optional[str] = None
    ) -> Union[LangGraphExecutor, SmolagentsExecutor]:
        """路由到执行器
        
        Args:
            complexity: 复杂度级别
            preferred_executor: 首选执行器
            
        Returns:
            Union[LangGraphExecutor, SmolagentsExecutor]: 执行器实例
        """
        if preferred_executor:
            if preferred_executor.lower() == "langgraph":
                return self.langgraph_executor
            elif preferred_executor.lower() == "smolagents":
                return self.smolagents_executor
        
        # 根据复杂度自动选择
        if complexity == ComplexityLevel.LINEAR:
            return self.langgraph_executor  # 线性预案首选LangGraph
        elif complexity == ComplexityLevel.BRANCH:
            return self.langgraph_executor  # 分支预案使用LangGraph
        else:  # MULTI_AGENT
            return self.smolagents_executor  # 复杂预案使用Smolagents
    
    async def prepare_execution_inputs(
        self,
        plan: PlanJSON,
        scenario: str,
        user_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """准备执行输入参数
        
        Args:
            plan: 预案对象
            scenario: 场景描述
            user_inputs: 用户输入参数
            
        Returns:
            Dict[str, Any]: 处理后的输入参数
        """
        processed_inputs = user_inputs.copy()
        
        # 从场景中提取参数
        extracted_params = await self.extract_params_from_scenario(scenario, plan)
        processed_inputs.update(extracted_params)
        
        # 检查必需参数
        for input_name in plan.plan_inputs.keys():
            if input_name not in processed_inputs:
                # 尝试提供默认值或提示用户
                default_value = await self.get_default_input_value(input_name, scenario)
                if default_value is not None:
                    processed_inputs[input_name] = default_value
                else:
                    logger.warning(f"缺少必需参数: {input_name}")
        
        return processed_inputs
    
    async def extract_params_from_scenario(
        self,
        scenario: str, 
        plan: PlanJSON
    ) -> Dict[str, Any]:
        """从场景描述中提取参数
        
        Args:
            scenario: 场景描述
            plan: 预案对象
            
        Returns:
            Dict[str, Any]: 提取的参数
        """
        import re
        
        params = {}
        
        # 简单的实体提取
        # 设备名称提取
        device_patterns = [
            r"([^，。]+?一线)",
            r"([^，。]+?换流站)",
            r"([^，。]+?变电站)"
        ]
        
        for pattern in device_patterns:
            match = re.search(pattern, scenario)
            if match:
                device_name = match.group(1)
                params["device"] = device_name
                params["设备"] = device_name
                break
        
        # 直流线路提取
        dc_patterns = [
            r"([^，。]+?直流)",
            r"(天中直流|天哈直流)"
        ]
        
        for pattern in dc_patterns:
            match = re.search(pattern, scenario)
            if match:
                dc_line = match.group(1)
                params["dc_line"] = dc_line
                params["直流线路"] = dc_line
                break
        
        return params
    
    async def get_default_input_value(
        self,
        input_name: str,
        scenario: str
    ) -> Optional[Any]:
        """获取输入参数的默认值
        
        Args:
            input_name: 参数名称
            scenario: 场景描述
            
        Returns:
            Any: 默认值，如果无法确定返回None
        """
        # 常见参数的默认值
        defaults = {
            "device": "未知设备",
            "dc_line": "天中直流",  # 默认直流线路
            "设备": "未知设备",
            "直流线路": "天中直流"
        }
        
        return defaults.get(input_name)
    
    async def generate_decision_report(
        self,
        execution_result: ExecutionResult
    ) -> DecisionReport:
        """生成决策报告
        
        Args:
            execution_result: 执行结果
            
        Returns:
            DecisionReport: 决策报告
        """
        from ..core.models import DecisionReport
        import uuid
        
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        
        # 构建报告内容
        summary = self._build_summary(execution_result)
        background = self._build_background(execution_result)
        data_sources = self._extract_data_sources(execution_result)
        calculations = self._extract_calculations(execution_result)
        recommendations = self._generate_recommendations(execution_result)
        warnings = self._check_warnings(execution_result)
        
        report = DecisionReport(
            report_id=report_id,
            execution_id=execution_result.execution_id,
            plan_title=execution_result.plan_id,
            scenario=execution_result.scenario,
            summary=summary,
            background=background,
            data_sources=data_sources,
            calculations=calculations,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat(),
            confidence_level="高" if execution_result.success else "低",
            warnings=warnings
        )
        
        return report
    
    def _build_summary(self, result: ExecutionResult) -> str:
        """构建结论摘要"""
        if not result.success:
            return f"预案执行失败: {result.error_message}"
        
        # 提取主要结果
        key_outputs = []
        for key, value in result.final_outputs.items():
            if isinstance(value, (int, float)):
                key_outputs.append(f"{key}={value}")
            else:
                key_outputs.append(f"{key}={str(value)}")
        
        return f"预案执行成功，主要结果: {', '.join(key_outputs)}"
    
    def _build_background(self, result: ExecutionResult) -> str:
        """构建背景信息"""
        return f"""
场景描述: {result.scenario}
预案ID: {result.plan_id}
执行时间: {result.execution_time:.2f}秒
执行步骤数: {len(result.step_results)}
        """.strip()
    
    def _extract_data_sources(self, result: ExecutionResult) -> List[Dict[str, Any]]:
        """提取数据来源"""
        sources = []
        
        for step in result.step_results:
            if step.get("success"):
                sources.append({
                    "step": step.get("step_id"),
                    "type": step.get("step_type"),
                    "description": step.get("description"),
                    "timestamp": step.get("timestamp")
                })
        
        return sources
    
    def _extract_calculations(self, result: ExecutionResult) -> List[Dict[str, Any]]:
        """提取计算过程"""
        calculations = []
        
        for step in result.step_results:
            if step.get("step_type") == "compute" and step.get("success"):
                calculations.append({
                    "step": step.get("step_id"),
                    "description": step.get("description"),
                    "inputs": step.get("inputs", {}),
                    "outputs": step.get("outputs", {}),
                    "formula": step.get("formula", "")
                })
        
        return calculations
    
    def _generate_recommendations(self, result: ExecutionResult) -> List[str]:
        """生成建议措施"""
        recommendations = []
        
        if result.success:
            recommendations.append("建议按照计算结果执行相应的调度操作")
            recommendations.append("持续监控系统运行状态")
            
            # 根据结果值生成具体建议
            for key, value in result.final_outputs.items():
                if isinstance(value, (int, float)):
                    if "限额" in key:
                        recommendations.append(f"建议将{key}设置为{value}MW")
        else:
            recommendations.append("建议人工干预，检查系统状态")
            recommendations.append("核查输入参数是否正确")
            recommendations.append("联系技术支持团队")
        
        return recommendations
    
    def _check_warnings(self, result: ExecutionResult) -> List[str]:
        """检查警告信息"""
        warnings = []
        
        if not result.success:
            warnings.append(f"预案执行失败: {result.error_message}")
        
        if result.execution_time > 60:
            warnings.append("执行时间超过预期，建议优化预案")
        
        return warnings
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态
        
        Args:
            execution_id: 执行ID
            
        Returns:
            Dict[str, Any]: 执行状态信息
        """
        result = self.execution_history.get(execution_id)
        if not result:
            return None
        
        return {
            "execution_id": execution_id,
            "plan_id": result.plan_id,
            "scenario": result.scenario,
            "success": result.success,
            "execution_time": result.execution_time,
            "error_message": result.error_message,
            "step_count": len(result.step_results),
            "final_outputs": result.final_outputs
        }
    
    def list_available_plans(self) -> List[Dict[str, Any]]:
        """列出可用预案
        
        Returns:
            List[Dict[str, Any]]: 预案信息列表
        """
        plans_info = []
        
        for plan in self.plan_cache.values():
            plans_info.append({
                "plan_id": plan.plan_id,
                "title": plan.title,
                "description": plan.description,
                "step_count": len(plan.steps),
                "inputs": list(plan.plan_inputs.keys()),
                "outputs": plan.plan_outputs
            })
        
        return plans_info