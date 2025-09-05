"""执行器基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from ..core.models import PlanJSON, ExecutionState, ExecutionResult, StepType
from ..utils.logger import logger


class BaseExecutor(ABC):
    """执行器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_history: Dict[str, ExecutionResult] = {}
    
    @abstractmethod
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
        pass
    
    def create_execution_state(
        self, 
        plan: PlanJSON, 
        scenario: str,
        inputs: Dict[str, Any]
    ) -> ExecutionState:
        """创建执行状态
        
        Args:
            plan: 预案对象
            scenario: 场景描述
            inputs: 输入参数
            
        Returns:
            ExecutionState: 执行状态对象
        """
        execution_id = f"{plan.plan_id}_{uuid.uuid4().hex[:8]}"
        
        return ExecutionState(
            plan_id=plan.plan_id,
            execution_id=execution_id,
            scenario=scenario,
            inputs=inputs,
            variables=inputs.copy(),  # 初始化变量表
            status="pending"
        )
    
    def create_execution_result(
        self,
        state: ExecutionState,
        success: bool,
        execution_time: float,
        error_message: Optional[str] = None,
        failed_step: Optional[str] = None
    ) -> ExecutionResult:
        """创建执行结果
        
        Args:
            state: 执行状态
            success: 是否成功
            execution_time: 执行时间
            error_message: 错误信息
            failed_step: 失败步骤
            
        Returns:
            ExecutionResult: 执行结果对象
        """
        # 提取最终输出
        final_outputs = {}
        if hasattr(state, 'final_outputs') and state.final_outputs:
            final_outputs = state.final_outputs
        
        result = ExecutionResult(
            execution_id=state.execution_id,
            plan_id=state.plan_id,
            success=success,
            scenario=state.scenario,
            final_outputs=final_outputs,
            variables=state.variables,
            step_results=state.step_history,
            execution_time=execution_time,
            error_message=error_message,
            failed_step=failed_step
        )
        
        # 保存到历史记录
        self.execution_history[state.execution_id] = result
        
        return result
    
    def validate_inputs(self, plan: PlanJSON, inputs: Dict[str, Any]) -> bool:
        """验证输入参数
        
        Args:
            plan: 预案对象
            inputs: 输入参数
            
        Returns:
            bool: 验证结果
            
        Raises:
            ValueError: 验证失败时抛出
        """
        # 检查必需的预案输入
        for input_name, input_desc in plan.plan_inputs.items():
            if input_name not in inputs:
                raise ValueError(f"缺少必需的输入参数: {input_name} ({input_desc})")
        
        return True
    
    def get_execution_history(self, execution_id: str) -> Optional[ExecutionResult]:
        """获取执行历史
        
        Args:
            execution_id: 执行ID
            
        Returns:
            ExecutionResult: 执行结果，如果不存在返回None
        """
        return self.execution_history.get(execution_id)
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()
        logger.info(f"{self.name}执行器历史已清空")


class ExecutorError(Exception):
    """执行器异常"""
    
    def __init__(self, message: str, step_id: Optional[str] = None):
        self.message = message
        self.step_id = step_id
        super().__init__(message)