from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator


class StepType(str, Enum):
    """步骤类型枚举"""
    RAG = "rag"
    TOOL = "tool" 
    COMPUTE = "compute"


class Variable(BaseModel):
    """变量定义模型"""
    name: str = Field(..., description="变量名称")
    symbol: str = Field(..., description="变量符号")
    unit: str = Field(..., description="变量单位")
    description: Optional[str] = Field(None, description="变量描述")
    formula: Optional[str] = Field(None, description="计算公式")


class PlanStep(BaseModel):
    """预案步骤模型"""
    id: str = Field(..., description="步骤ID")
    type: StepType = Field(..., description="步骤类型")
    description: str = Field(..., description="步骤描述")
    inputs: Dict[str, Union[str, Dict[str, Any]]] = Field(
        default_factory=dict, description="输入参数"
    )
    outputs: List[str] = Field(default_factory=list, description="输出变量")
    
    # RAG步骤特有字段
    query: Optional[str] = Field(None, description="RAG查询内容")
    
    # Tool步骤特有字段
    tool_name: Optional[str] = Field(None, description="工具名称")
    
    # Compute步骤特有字段
    formula: Optional[str] = Field(None, description="计算公式")
    
    @validator('query')
    def validate_rag_query(cls, v, values):
        if values.get('type') == StepType.RAG and not v:
            raise ValueError("RAG步骤必须提供query字段")
        return v
    
    @validator('tool_name')
    def validate_tool_name(cls, v, values):
        if values.get('type') == StepType.TOOL and not v:
            raise ValueError("Tool步骤必须提供tool_name字段")
        return v
    
    @validator('formula')
    def validate_formula(cls, v, values):
        if values.get('type') == StepType.COMPUTE and not v:
            raise ValueError("Compute步骤必须提供formula字段")
        return v


class PlanJSON(BaseModel):
    """预案JSON模型"""
    plan_id: str = Field(..., description="预案ID")
    title: str = Field(..., description="预案标题")
    description: str = Field(..., description="预案描述")
    version: str = Field(default="1.0", description="预案版本")
    
    variables: List[Variable] = Field(default_factory=list, description="变量定义")
    steps: List[PlanStep] = Field(..., description="执行步骤")
    
    # 预案输入输出定义
    plan_inputs: Dict[str, str] = Field(default_factory=dict, description="预案输入参数")
    plan_outputs: List[str] = Field(default_factory=list, description="预案输出")
    
    # 元数据
    author: Optional[str] = Field(None, description="创建者")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")
    tags: List[str] = Field(default_factory=list, description="标签")


class ExecutionState(BaseModel):
    """执行状态模型"""
    plan_id: str = Field(..., description="预案ID")
    execution_id: str = Field(..., description="执行ID")
    
    # 执行输入
    scenario: str = Field(..., description="场景描述")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="输入参数")
    
    # 执行状态
    current_step: Optional[str] = Field(None, description="当前步骤")
    variables: Dict[str, Any] = Field(default_factory=dict, description="变量值")
    
    # 执行历史
    step_history: List[Dict[str, Any]] = Field(default_factory=list, description="步骤执行历史")
    
    # 执行结果
    status: str = Field(default="pending", description="执行状态: pending/running/completed/failed")
    error_message: Optional[str] = Field(None, description="错误信息")
    final_outputs: Dict[str, Any] = Field(default_factory=dict, description="最终输出")


class ToolResult(BaseModel):
    """工具调用结果模型"""
    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否成功")
    result: Any = Field(None, description="结果值")
    unit: Optional[str] = Field(None, description="结果单位")
    source: Optional[str] = Field(None, description="数据来源")
    timestamp: Optional[str] = Field(None, description="时间戳")
    error_message: Optional[str] = Field(None, description="错误信息")


class ComputeResult(BaseModel):
    """计算结果模型"""
    formula: str = Field(..., description="计算公式")
    inputs: Dict[str, Any] = Field(..., description="输入值")
    result: Any = Field(..., description="计算结果")
    unit: Optional[str] = Field(None, description="结果单位")
    step_by_step: Optional[List[str]] = Field(None, description="逐步计算过程")


class RAGResult(BaseModel):
    """RAG查询结果模型"""
    query: str = Field(..., description="查询内容")
    results: List[str] = Field(..., description="查询结果")
    sources: List[str] = Field(default_factory=list, description="来源文档")
    confidence: Optional[float] = Field(None, description="置信度")


class ExecutionResult(BaseModel):
    """执行结果模型"""
    execution_id: str = Field(..., description="执行ID")
    plan_id: str = Field(..., description="预案ID")
    success: bool = Field(..., description="是否成功")
    
    # 执行详情
    scenario: str = Field(..., description="场景描述")
    final_outputs: Dict[str, Any] = Field(..., description="最终输出")
    variables: Dict[str, Any] = Field(..., description="所有变量值")
    
    # 执行过程
    step_results: List[Dict[str, Any]] = Field(..., description="步骤执行结果")
    execution_time: float = Field(..., description="执行时间(秒)")
    
    # 错误信息
    error_message: Optional[str] = Field(None, description="错误信息")
    failed_step: Optional[str] = Field(None, description="失败步骤")


class DecisionReport(BaseModel):
    """决策报告模型"""
    report_id: str = Field(..., description="报告ID")
    execution_id: str = Field(..., description="执行ID")
    plan_title: str = Field(..., description="预案标题")
    
    # 报告内容
    scenario: str = Field(..., description="场景描述")
    summary: str = Field(..., description="结论摘要")
    
    # 详细内容
    background: str = Field(..., description="背景信息")
    data_sources: List[Dict[str, Any]] = Field(..., description="数据来源")
    calculations: List[Dict[str, Any]] = Field(..., description="计算过程")
    recommendations: List[str] = Field(..., description="建议措施")
    
    # 元数据
    generated_at: str = Field(..., description="生成时间")
    confidence_level: Optional[str] = Field(None, description="置信等级")
    warnings: List[str] = Field(default_factory=list, description="警告信息")