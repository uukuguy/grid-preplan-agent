"""电网专用工具集合"""

from typing import Dict, Any, List, Optional
import asyncio
import httpx
from datetime import datetime

from .api_registry import BaseTool, register_tool, ToolResult
from ..utils.logger import logger


@register_tool("query_send_limit", "查询直流线路送端限额")
class QuerySendLimitTool(BaseTool):
    """查询送端限额工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.base_url = "http://localhost:8000"  # 模拟API地址
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行查询送端限额"""
        line = kwargs.get("line")
        if not line:
            return self._create_result(False, error_message="缺少line参数")
        
        try:
            # 模拟API调用
            async with httpx.AsyncClient() as client:
                # 这里应该调用真实的电网API
                # response = await client.get(f"{self.base_url}/send_limit/{line}")
                
                # 模拟数据返回
                mock_data = {
                    "天中直流": 3200.0,
                    "天哈直流": 2800.0,
                    "default": 2500.0
                }
                
                result_value = mock_data.get(line, mock_data["default"])
                
                return self._create_result(
                    success=True,
                    result=result_value,
                    unit="MW",
                    source=f"送端限额数据库-{line}"
                )
                
        except Exception as e:
            return self._create_result(False, error_message=f"查询送端限额失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "line": {
                    "type": "string",
                    "description": "直流线路名称"
                }
            },
            "required": ["line"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "送端限额值"
                },
                "unit": {
                    "type": "string",
                    "description": "单位(MW)"
                }
            }
        }


@register_tool("query_recv_limit", "查询直流线路受端限额")
class QueryRecvLimitTool(BaseTool):
    """查询受端限额工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.base_url = "http://localhost:8000"
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行查询受端限额"""
        line = kwargs.get("line")
        if not line:
            return self._create_result(False, error_message="缺少line参数")
        
        try:
            # 模拟数据返回
            mock_data = {
                "天中直流": 3000.0,
                "天哈直流": 2600.0,
                "default": 2400.0
            }
            
            result_value = mock_data.get(line, mock_data["default"])
            
            return self._create_result(
                success=True,
                result=result_value,
                unit="MW",
                source=f"受端限额数据库-{line}"
            )
            
        except Exception as e:
            return self._create_result(False, error_message=f"查询受端限额失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "line": {
                    "type": "string",
                    "description": "直流线路名称"
                }
            },
            "required": ["line"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "受端限额值"
                },
                "unit": {
                    "type": "string",
                    "description": "单位(MW)"
                }
            }
        }


@register_tool("query_converter_capacity", "查询换流器运行容量")
class QueryConverterCapacityTool(BaseTool):
    """查询换流器运行容量工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.base_url = "http://localhost:8000"
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行查询换流器容量"""
        line = kwargs.get("line")
        if not line:
            return self._create_result(False, error_message="缺少line参数")
        
        try:
            # 模拟设备数据
            mock_data = {
                "天中直流": {
                    "P_max_convert": 1600.0,  # MW
                    "F_current": 2.5,        # kA
                    "N_convert": 2           # 个数
                },
                "天哈直流": {
                    "P_max_convert": 1400.0,
                    "F_current": 2.0,
                    "N_convert": 2
                }
            }
            
            line_data = mock_data.get(line)
            if not line_data:
                return self._create_result(False, error_message=f"未找到线路数据: {line}")
            
            # 计算系统传输能力: P_dcsystem = P_max_convert × F_current × N_convert
            p_dcsystem = (
                line_data["P_max_convert"] * 
                line_data["F_current"] * 
                line_data["N_convert"]
            )
            
            return self._create_result(
                success=True,
                result=p_dcsystem,
                unit="MW",
                source=f"换流器参数数据库-{line}"
            )
            
        except Exception as e:
            return self._create_result(False, error_message=f"查询换流器容量失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "line": {
                    "type": "string",
                    "description": "直流线路名称"
                }
            },
            "required": ["line"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "系统传输能力值"
                },
                "unit": {
                    "type": "string",
                    "description": "单位(MW)"
                }
            }
        }


@register_tool("query_device_impact", "查询设备影响的直流线路")
class QueryDeviceImpactTool(BaseTool):
    """查询设备影响工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.knowledge_base = {
            "天哈一线": {
                "affected_lines": ["天中直流"],
                "side": "送端",
                "description": "天哈一线停运影响天中直流送端"
            },
            "华中换流站": {
                "affected_lines": ["天中直流", "天哈直流"],
                "side": "受端",
                "description": "华中换流站影响多条直流受端"
            }
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行查询设备影响"""
        device = kwargs.get("device")
        if not device:
            return self._create_result(False, error_message="缺少device参数")
        
        try:
            device_info = self.knowledge_base.get(device)
            if not device_info:
                return self._create_result(False, error_message=f"未找到设备信息: {device}")
            
            result = {
                "dc_line": device_info["affected_lines"][0] if device_info["affected_lines"] else "",
                "side_info": device_info["side"],
                "affected_lines": device_info["affected_lines"],
                "description": device_info["description"]
            }
            
            return self._create_result(
                success=True,
                result=result,
                source=f"设备影响知识库-{device}"
            )
            
        except Exception as e:
            return self._create_result(False, error_message=f"查询设备影响失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "设备名称"
                }
            },
            "required": ["device"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dc_line": {
                    "type": "string",
                    "description": "主要影响的直流线路"
                },
                "side_info": {
                    "type": "string",
                    "description": "影响端（送端/受端）"
                }
            }
        }


@register_tool("compute_min_value", "计算最小值")
class ComputeMinValueTool(BaseTool):
    """计算最小值工具"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行最小值计算"""
        try:
            values = []
            
            # 收集所有数值参数
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, str):
                    try:
                        values.append(float(value))
                    except ValueError:
                        continue
            
            if not values:
                return self._create_result(False, error_message="没有有效的数值参数")
            
            min_value = min(values)
            
            return self._create_result(
                success=True,
                result=min_value,
                source="数学计算"
            )
            
        except Exception as e:
            return self._create_result(False, error_message=f"计算失败: {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": {
                "type": ["number", "string"]
            },
            "description": "接受任意数量的数值参数"
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "最小值"
                }
            }
        }


# 初始化所有工具
def initialize_grid_tools():
    """初始化所有电网工具"""
    logger.info("电网工具已初始化")