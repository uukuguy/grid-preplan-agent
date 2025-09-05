"""Mock工具，用于测试和演示"""

from typing import Dict, Any
import asyncio
import random
from datetime import datetime

from .api_registry import BaseTool, register_tool, ToolResult
from ..utils.logger import logger


@register_tool("mock_rag_query", "模拟RAG查询工具")
class MockRAGQueryTool(BaseTool):
    """模拟RAG查询工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.knowledge_base = {
            "送端判定": "根据电网拓扑结构，该设备位于送端电网",
            "受端判定": "根据电网拓扑结构，该设备位于受端电网",
            "直流限额规程": "根据《电网调度管理条例》，直流输电限额应考虑送受端能力",
            "故障处理原则": "设备故障时应立即评估对直流输电的影响"
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行模拟RAG查询"""
        query = kwargs.get("query", "")
        
        try:
            # 模拟查询延迟
            await asyncio.sleep(0.1)
            
            # 简单的关键词匹配
            result_text = ""
            for key, value in self.knowledge_base.items():
                if any(keyword in query for keyword in key.split()):
                    result_text = value
                    break
            
            if not result_text:
                result_text = "未找到相关信息"
            
            return self._create_result(
                success=True,
                result={
                    "answer": result_text,
                    "source": "模拟知识库",
                    "confidence": 0.8
                },
                source="Mock RAG系统"
            )
            
        except Exception as e:
            return self._create_result(False, error_message=f"RAG查询失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询内容"
                }
            },
            "required": ["query"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "查询结果"
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度"
                }
            }
        }


@register_tool("mock_api_call", "模拟API调用工具")
class MockAPICallTool(BaseTool):
    """模拟外部API调用工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.api_responses = {
            "weather": {"temperature": 25, "humidity": 60},
            "power_load": {"current_load": 85000, "max_capacity": 100000},
            "grid_status": {"voltage": 500, "frequency": 50.0}
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行模拟API调用"""
        api_name = kwargs.get("api_name", "default")
        
        try:
            # 模拟网络延迟
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # 模拟偶发失败
            if random.random() < 0.1:  # 10%失败率
                return self._create_result(
                    False, 
                    error_message="模拟网络错误"
                )
            
            # 返回模拟数据
            response_data = self.api_responses.get(
                api_name, 
                {"status": "ok", "data": f"mock_data_for_{api_name}"}
            )
            
            return self._create_result(
                success=True,
                result=response_data,
                source=f"Mock API - {api_name}"
            )
            
        except Exception as e:
            return self._create_result(False, error_message=f"API调用失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "api_name": {
                    "type": "string",
                    "description": "API名称"
                },
                "params": {
                    "type": "object",
                    "description": "API参数"
                }
            },
            "required": ["api_name"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "API返回数据"
                }
            }
        }


@register_tool("mock_calculator", "模拟计算器工具")
class MockCalculatorTool(BaseTool):
    """模拟计算器工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.supported_operations = ["add", "subtract", "multiply", "divide", "min", "max"]
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行模拟计算"""
        operation = kwargs.get("operation", "add")
        operands = kwargs.get("operands", [])
        
        if operation not in self.supported_operations:
            return self._create_result(
                False, 
                error_message=f"不支持的操作: {operation}"
            )
        
        if not operands or len(operands) < 2:
            return self._create_result(
                False, 
                error_message="至少需要两个操作数"
            )
        
        try:
            # 确保操作数都是数值
            numbers = [float(x) for x in operands]
            
            # 执行计算
            if operation == "add":
                result = sum(numbers)
            elif operation == "subtract":
                result = numbers[0] - sum(numbers[1:])
            elif operation == "multiply":
                result = 1
                for num in numbers:
                    result *= num
            elif operation == "divide":
                result = numbers[0]
                for num in numbers[1:]:
                    if num == 0:
                        return self._create_result(
                            False, 
                            error_message="除零错误"
                        )
                    result /= num
            elif operation == "min":
                result = min(numbers)
            elif operation == "max":
                result = max(numbers)
            
            return self._create_result(
                success=True,
                result=result,
                source="Mock计算器"
            )
            
        except ValueError as e:
            return self._create_result(False, error_message=f"数值转换错误: {str(e)}")
        except Exception as e:
            return self._create_result(False, error_message=f"计算错误: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.supported_operations,
                    "description": "计算操作类型"
                },
                "operands": {
                    "type": "array",
                    "items": {"type": ["number", "string"]},
                    "description": "操作数列表"
                }
            },
            "required": ["operation", "operands"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "number",
                    "description": "计算结果"
                }
            }
        }


def initialize_mock_tools():
    """初始化所有Mock工具"""
    logger.info("Mock工具已初始化")