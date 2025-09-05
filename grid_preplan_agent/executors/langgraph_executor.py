"""LangGraph执行器实现"""

from typing import Dict, Any, List, Optional, Callable
import asyncio
import time
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import CompiledStateGraph
    from typing_extensions import TypedDict
except ImportError:
    # 如果langgraph不可用，创建mock实现
    print("Warning: langgraph not available, using mock implementation")
    class StateGraph:
        def __init__(self, schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def set_entry_point(self, entry): pass
        def compile(self): return MockCompiledGraph()
    
    class MockCompiledGraph:
        async def ainvoke(self, state): return state
    
    END = "END"
    
    class TypedDict(dict):
        pass

from .base_executor import BaseExecutor, ExecutorError
from ..core.models import PlanJSON, PlanStep, ExecutionState, ExecutionResult, StepType
from ..tools.api_registry import tool_registry, call_tool
from ..utils.logger import logger


# 定义LangGraph状态Schema
class GraphState(TypedDict):
    """LangGraph状态定义"""
    execution_id: str
    plan_id: str
    scenario: str
    current_step: str
    variables: Dict[str, Any]
    step_results: List[Dict[str, Any]]
    error_message: Optional[str]
    status: str


class LangGraphExecutor(BaseExecutor):
    """LangGraph执行器"""
    
    def __init__(self):
        super().__init__("LangGraph")
        self.compiled_graphs: Dict[str, Any] = {}
    
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
        logger.info(f"LangGraph开始执行预案: {plan.plan_id}")
        start_time = time.time()
        
        try:
            # 验证输入
            self.validate_inputs(plan, inputs)
            
            # 创建执行状态
            state = self.create_execution_state(plan, scenario, inputs)
            state.status = "running"
            
            # 构建或获取StateGraph
            graph = await self.get_or_build_graph(plan)
            
            # 准备初始状态
            initial_state = GraphState(
                execution_id=state.execution_id,
                plan_id=plan.plan_id,
                scenario=scenario,
                current_step="",
                variables=state.variables,
                step_results=[],
                error_message=None,
                status="running"
            )
            
            # 执行图
            final_state = await graph.ainvoke(initial_state)
            
            # 更新执行状态
            state.variables = final_state["variables"]
            state.step_history = final_state["step_results"]
            state.status = final_state["status"]
            state.error_message = final_state.get("error_message")
            
            # 提取最终输出
            final_outputs = {}
            for output_var in plan.plan_outputs:
                if output_var in state.variables:
                    final_outputs[output_var] = state.variables[output_var]
            
            state.final_outputs = final_outputs
            
            execution_time = time.time() - start_time
            success = final_state["status"] == "completed"
            
            result = self.create_execution_result(
                state=state,
                success=success,
                execution_time=execution_time,
                error_message=state.error_message
            )
            
            logger.info(f"LangGraph执行{'成功' if success else '失败'}: {plan.plan_id}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"LangGraph执行异常: {plan.plan_id}, 错误: {str(e)}")
            
            state = self.create_execution_state(plan, scenario, inputs)
            return self.create_execution_result(
                state=state,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def get_or_build_graph(self, plan: PlanJSON) -> Any:
        """获取或构建StateGraph
        
        Args:
            plan: 预案对象
            
        Returns:
            CompiledStateGraph: 编译后的状态图
        """
        # 检查是否已缓存
        if plan.plan_id in self.compiled_graphs:
            return self.compiled_graphs[plan.plan_id]
        
        # 构建新的图
        graph = await self.build_graph(plan)
        self.compiled_graphs[plan.plan_id] = graph
        
        return graph
    
    async def build_graph(self, plan: PlanJSON) -> Any:
        """构建StateGraph
        
        Args:
            plan: 预案对象
            
        Returns:
            CompiledStateGraph: 编译后的状态图
        """
        logger.info(f"构建LangGraph: {plan.plan_id}")
        
        # 创建状态图
        graph = StateGraph(GraphState)
        
        # 添加步骤节点
        for step in plan.steps:
            node_func = self.create_step_node(step)
            graph.add_node(step.id, node_func)
        
        # 添加边（按步骤顺序连接）
        if plan.steps:
            # 设置入口点
            graph.set_entry_point(plan.steps[0].id)
            
            # 连接步骤
            for i in range(len(plan.steps) - 1):
                current_step = plan.steps[i]
                next_step = plan.steps[i + 1]
                graph.add_edge(current_step.id, next_step.id)
            
            # 最后一个步骤连接到END
            graph.add_edge(plan.steps[-1].id, END)
        
        return graph.compile()
    
    def create_step_node(self, step: PlanStep) -> Callable:
        """创建步骤节点函数
        
        Args:
            step: 预案步骤
            
        Returns:
            Callable: 节点执行函数
        """
        async def step_node(state: GraphState) -> GraphState:
            """步骤节点执行函数"""
            logger.info(f"执行步骤: {step.id} - {step.description}")
            
            try:
                # 更新当前步骤
                state["current_step"] = step.id
                
                # 根据步骤类型执行
                if step.type == StepType.RAG:
                    result = await self.execute_rag_step(step, state)
                elif step.type == StepType.TOOL:
                    result = await self.execute_tool_step(step, state)
                elif step.type == StepType.COMPUTE:
                    result = await self.execute_compute_step(step, state)
                else:
                    raise ExecutorError(f"不支持的步骤类型: {step.type}")
                
                # 更新变量
                for output_var in step.outputs:
                    if output_var in result:
                        state["variables"][output_var] = result[output_var]
                
                # 记录步骤结果
                step_result = {
                    "step_id": step.id,
                    "step_type": step.type.value,
                    "description": step.description,
                    "success": True,
                    "outputs": result,
                    "timestamp": datetime.now().isoformat()
                }
                state["step_results"].append(step_result)
                
                logger.info(f"步骤执行成功: {step.id}")
                
            except Exception as e:
                logger.error(f"步骤执行失败: {step.id}, 错误: {str(e)}")
                
                # 记录失败
                step_result = {
                    "step_id": step.id,
                    "step_type": step.type.value,
                    "description": step.description,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                state["step_results"].append(step_result)
                state["error_message"] = str(e)
                state["status"] = "failed"
                
                return state
            
            # 检查是否为最后一个步骤
            if step.id == state.get("last_step_id"):
                state["status"] = "completed"
            
            return state
        
        return step_node
    
    async def execute_rag_step(self, step: PlanStep, state: GraphState) -> Dict[str, Any]:
        """执行RAG步骤
        
        Args:
            step: RAG步骤
            state: 执行状态
            
        Returns:
            Dict[str, Any]: 步骤输出
        """
        query = step.query
        if not query:
            raise ExecutorError("RAG步骤缺少query字段")
        
        # 替换变量占位符
        query = self.substitute_variables(query, state["variables"])
        
        # 调用RAG工具（这里使用mock工具）
        try:
            rag_result = await call_tool("mock_rag_query", query=query)
            
            if not rag_result.success:
                raise ExecutorError(f"RAG查询失败: {rag_result.error_message}")
            
            # 解析结果
            result_data = rag_result.result
            outputs = {}
            
            # 根据输出变量分配结果
            if step.outputs:
                if isinstance(result_data, dict):
                    # 如果结果是字典，尝试映射到输出变量
                    for i, output_var in enumerate(step.outputs):
                        if i == 0:
                            outputs[output_var] = result_data.get("answer", str(result_data))
                        else:
                            outputs[output_var] = result_data
                else:
                    # 如果结果是简单值，分配给第一个输出变量
                    outputs[step.outputs[0]] = result_data
            
            return outputs
            
        except Exception as e:
            raise ExecutorError(f"RAG步骤执行失败: {str(e)}")
    
    async def execute_tool_step(self, step: PlanStep, state: GraphState) -> Dict[str, Any]:
        """执行Tool步骤
        
        Args:
            step: Tool步骤
            state: 执行状态
            
        Returns:
            Dict[str, Any]: 步骤输出
        """
        tool_name = step.tool_name
        if not tool_name:
            raise ExecutorError("Tool步骤缺少tool_name字段")
        
        # 准备工具输入参数
        tool_inputs = {}
        for key, value in step.inputs.items():
            if isinstance(value, str):
                # 替换变量占位符
                tool_inputs[key] = self.substitute_variables(value, state["variables"])
            else:
                tool_inputs[key] = value
        
        # 调用工具
        try:
            tool_result = await call_tool(tool_name, **tool_inputs)
            
            if not tool_result.success:
                raise ExecutorError(f"工具调用失败: {tool_result.error_message}")
            
            # 分配输出
            outputs = {}
            if step.outputs:
                outputs[step.outputs[0]] = tool_result.result
            
            return outputs
            
        except Exception as e:
            raise ExecutorError(f"Tool步骤执行失败: {str(e)}")
    
    async def execute_compute_step(self, step: PlanStep, state: GraphState) -> Dict[str, Any]:
        """执行Compute步骤
        
        Args:
            step: Compute步骤  
            state: 执行状态
            
        Returns:
            Dict[str, Any]: 步骤输出
        """
        formula = step.formula
        if not formula:
            raise ExecutorError("Compute步骤缺少formula字段")
        
        try:
            # 准备计算输入
            compute_inputs = {}
            for key, value in step.inputs.items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    # 变量引用
                    var_name = value[1:-1]
                    if var_name in state["variables"]:
                        compute_inputs[key] = state["variables"][var_name]
                    else:
                        raise ExecutorError(f"未找到变量: {var_name}")
                else:
                    compute_inputs[key] = value
            
            # 简单的公式计算（这里可以扩展为更复杂的表达式解析）
            result = await self.evaluate_formula(formula, compute_inputs)
            
            # 分配输出
            outputs = {}
            if step.outputs:
                outputs[step.outputs[0]] = result
            
            return outputs
            
        except Exception as e:
            raise ExecutorError(f"Compute步骤执行失败: {str(e)}")
    
    async def evaluate_formula(self, formula: str, inputs: Dict[str, Any]) -> Any:
        """评估计算公式
        
        Args:
            formula: 计算公式
            inputs: 输入变量
            
        Returns:
            Any: 计算结果
        """
        # 简化的公式评估（实际项目中应使用更安全的表达式解析器）
        if formula.startswith("min(") and formula.endswith(")"):
            # 处理min函数
            vars_str = formula[4:-1]  # 去掉"min("和")"
            var_names = [v.strip() for v in vars_str.split(",")]
            
            values = []
            for var_name in var_names:
                if var_name in inputs:
                    values.append(float(inputs[var_name]))
                else:
                    raise ValueError(f"公式中的变量未找到: {var_name}")
            
            return min(values)
        
        elif formula.startswith("max(") and formula.endswith(")"):
            # 处理max函数
            vars_str = formula[4:-1]
            var_names = [v.strip() for v in vars_str.split(",")]
            
            values = []
            for var_name in var_names:
                if var_name in inputs:
                    values.append(float(inputs[var_name]))
                else:
                    raise ValueError(f"公式中的变量未找到: {var_name}")
            
            return max(values)
        
        else:
            # 其他公式类型可以在这里扩展
            raise ValueError(f"不支持的公式格式: {formula}")
    
    def substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """替换文本中的变量占位符
        
        Args:
            text: 包含占位符的文本
            variables: 变量字典
            
        Returns:
            str: 替换后的文本
        """
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            else:
                logger.warning(f"未找到变量: {var_name}")
                return match.group(0)  # 保留原占位符
        
        # 替换{variable_name}格式的占位符
        return re.sub(r'\{([^}]+)\}', replace_var, text)
    
    def clear_graph_cache(self) -> None:
        """清空图缓存"""
        self.compiled_graphs.clear()
        logger.info("LangGraph缓存已清空")