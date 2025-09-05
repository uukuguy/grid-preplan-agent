from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Type
from datetime import datetime
from dataclasses import dataclass
import inspect

from ..core.models import ToolResult
from ..utils.logger import logger


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tool_class: Type["BaseTool"]
    version: str = "1.0"


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具调用
        
        Args:
            **kwargs: 工具输入参数
            
        Returns:
            ToolResult: 工具执行结果
        """
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """获取输入参数Schema"""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """获取输出结果Schema"""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """验证输入参数
        
        Args:
            inputs: 输入参数
            
        Returns:
            bool: 验证结果
        """
        schema = self.get_input_schema()
        required_fields = schema.get("required", [])
        
        # 检查必需字段
        for field in required_fields:
            if field not in inputs:
                raise ValueError(f"缺少必需参数: {field}")
        
        return True
    
    def _create_result(self, success: bool, result: Any = None, 
                      unit: str = None, source: str = None,
                      error_message: str = None) -> ToolResult:
        """创建工具结果对象
        
        Args:
            success: 是否成功
            result: 结果值
            unit: 单位
            source: 数据来源
            error_message: 错误信息
            
        Returns:
            ToolResult: 工具结果
        """
        return ToolResult(
            tool_name=self.name,
            success=success,
            result=result,
            unit=unit,
            source=source or f"{self.name}_api",
            timestamp=datetime.now().isoformat(),
            error_message=error_message
        )


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, ToolInfo] = {}
        self._instances: Dict[str, BaseTool] = {}
    
    def register(self, tool_class: Type[BaseTool], **kwargs) -> None:
        """注册工具
        
        Args:
            tool_class: 工具类
            **kwargs: 初始化参数
        """
        # 创建工具实例获取信息
        temp_instance = tool_class(**kwargs)
        
        tool_info = ToolInfo(
            name=temp_instance.name,
            description=temp_instance.description,
            input_schema=temp_instance.get_input_schema(),
            output_schema=temp_instance.get_output_schema(),
            tool_class=tool_class
        )
        
        self._tools[temp_instance.name] = tool_info
        self._instances[temp_instance.name] = temp_instance
        
        logger.info(f"工具已注册: {temp_instance.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例
        
        Args:
            name: 工具名称
            
        Returns:
            BaseTool: 工具实例，如果不存在返回None
        """
        return self._instances.get(name)
    
    def get_tool_info(self, name: str) -> Optional[ToolInfo]:
        """获取工具信息
        
        Args:
            name: 工具名称
            
        Returns:
            ToolInfo: 工具信息，如果不存在返回None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> Dict[str, ToolInfo]:
        """列出所有已注册的工具
        
        Returns:
            Dict[str, ToolInfo]: 工具信息字典
        """
        return self._tools.copy()
    
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在
        
        Args:
            name: 工具名称
            
        Returns:
            bool: 是否存在
        """
        return name in self._tools
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """执行工具调用
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            ToolResult: 执行结果
            
        Raises:
            ValueError: 工具不存在时抛出
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"工具不存在: {name}")
        
        try:
            # 验证输入
            tool.validate_inputs(kwargs)
            
            # 执行工具
            result = await tool.execute(**kwargs)
            
            logger.info(f"工具执行成功: {name}, 结果: {result.result}")
            return result
            
        except Exception as e:
            logger.error(f"工具执行失败: {name}, 错误: {str(e)}")
            return ToolResult(
                tool_name=name,
                success=False,
                result=None,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def unregister(self, name: str) -> bool:
        """注销工具
        
        Args:
            name: 工具名称
            
        Returns:
            bool: 是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            del self._instances[name]
            logger.info(f"工具已注销: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()
        self._instances.clear()
        logger.info("所有工具已清空")


# 全局工具注册中心实例
tool_registry = ToolRegistry()


def register_tool(name: str = None, description: str = None):
    """工具注册装饰器
    
    Args:
        name: 工具名称（可选）
        description: 工具描述（可选）
        
    Returns:
        装饰器函数
    """
    def decorator(tool_class: Type[BaseTool]):
        # 从类中获取名称和描述
        tool_name = name or getattr(tool_class, '_name', tool_class.__name__)
        tool_desc = description or getattr(tool_class, '_description', tool_class.__doc__ or "")
        
        # 注册工具
        tool_registry.register(tool_class, name=tool_name, description=tool_desc)
        return tool_class
    
    return decorator


def get_tool(name: str) -> Optional[BaseTool]:
    """获取工具实例（便捷函数）
    
    Args:
        name: 工具名称
        
    Returns:
        BaseTool: 工具实例
    """
    return tool_registry.get_tool(name)


async def call_tool(name: str, **kwargs) -> ToolResult:
    """调用工具（便捷函数）
    
    Args:
        name: 工具名称
        **kwargs: 工具参数
        
    Returns:
        ToolResult: 执行结果
    """
    return await tool_registry.execute_tool(name, **kwargs)