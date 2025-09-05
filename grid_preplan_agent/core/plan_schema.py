import json
from pathlib import Path
from typing import Dict, Any
import jsonschema
from jsonschema import validate, ValidationError

# Plan JSON Schema定义
PLAN_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://grid-ai.com/schemas/plan.json",
    "title": "Grid Plan JSON Schema",
    "description": "电网调度预案JSON格式Schema",
    "type": "object",
    "properties": {
        "plan_id": {
            "type": "string",
            "description": "预案唯一标识符",
            "pattern": "^[a-zA-Z0-9_-]+$"
        },
        "title": {
            "type": "string",
            "description": "预案标题",
            "minLength": 1,
            "maxLength": 200
        },
        "description": {
            "type": "string",
            "description": "预案描述",
            "maxLength": 1000
        },
        "version": {
            "type": "string",
            "description": "预案版本号",
            "pattern": "^\\d+\\.\\d+(\\.\\d+)?$",
            "default": "1.0"
        },
        "variables": {
            "type": "array",
            "description": "变量定义列表",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "变量名称"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "变量符号"
                    },
                    "unit": {
                        "type": "string",
                        "description": "变量单位"
                    },
                    "description": {
                        "type": "string",
                        "description": "变量描述"
                    },
                    "formula": {
                        "type": "string",
                        "description": "计算公式"
                    }
                },
                "required": ["name", "symbol", "unit"],
                "additionalProperties": false
            }
        },
        "steps": {
            "type": "array",
            "description": "执行步骤列表",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "步骤ID",
                        "pattern": "^[a-zA-Z0-9_-]+$"
                    },
                    "type": {
                        "type": "string",
                        "description": "步骤类型",
                        "enum": ["rag", "tool", "compute"]
                    },
                    "description": {
                        "type": "string",
                        "description": "步骤描述",
                        "minLength": 1
                    },
                    "inputs": {
                        "type": "object",
                        "description": "输入参数",
                        "additionalProperties": true
                    },
                    "outputs": {
                        "type": "array",
                        "description": "输出变量列表",
                        "items": {
                            "type": "string"
                        }
                    },
                    "query": {
                        "type": "string",
                        "description": "RAG查询内容(仅RAG类型步骤)"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "工具名称(仅Tool类型步骤)"
                    },
                    "formula": {
                        "type": "string",
                        "description": "计算公式(仅Compute类型步骤)"
                    }
                },
                "required": ["id", "type", "description", "outputs"],
                "allOf": [
                    {
                        "if": {"properties": {"type": {"const": "rag"}}},
                        "then": {"required": ["query"]}
                    },
                    {
                        "if": {"properties": {"type": {"const": "tool"}}},
                        "then": {"required": ["tool_name"]}
                    },
                    {
                        "if": {"properties": {"type": {"const": "compute"}}},
                        "then": {"required": ["formula"]}
                    }
                ],
                "additionalProperties": false
            }
        },
        "plan_inputs": {
            "type": "object",
            "description": "预案输入参数定义",
            "additionalProperties": {
                "type": "string"
            }
        },
        "plan_outputs": {
            "type": "array",
            "description": "预案输出变量列表",
            "items": {
                "type": "string"
            }
        },
        "author": {
            "type": "string",
            "description": "创建者"
        },
        "created_at": {
            "type": "string",
            "description": "创建时间",
            "format": "date-time"
        },
        "updated_at": {
            "type": "string",
            "description": "更新时间",
            "format": "date-time"
        },
        "tags": {
            "type": "array",
            "description": "标签列表",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["plan_id", "title", "description", "steps"],
    "additionalProperties": false
}


class PlanSchemaValidator:
    """预案JSON Schema验证器"""
    
    def __init__(self):
        self.schema = PLAN_JSON_SCHEMA
    
    def validate(self, plan_data: Dict[str, Any]) -> bool:
        """
        验证预案数据是否符合Schema
        
        Args:
            plan_data: 预案数据字典
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValidationError: 验证失败时抛出
        """
        try:
            validate(instance=plan_data, schema=self.schema)
            return True
        except ValidationError as e:
            raise ValidationError(f"预案JSON格式验证失败: {e.message}")
    
    def validate_json_string(self, plan_json: str) -> bool:
        """
        验证JSON字符串格式
        
        Args:
            plan_json: JSON字符串
            
        Returns:
            bool: 验证是否通过
        """
        try:
            plan_data = json.loads(plan_json)
            return self.validate(plan_data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"JSON格式错误: {e}")
    
    def validate_file(self, file_path: Path) -> bool:
        """
        验证JSON文件格式
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            bool: 验证是否通过
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        return self.validate(plan_data)
    
    def get_validation_errors(self, plan_data: Dict[str, Any]) -> list:
        """
        获取详细的验证错误信息
        
        Args:
            plan_data: 预案数据字典
            
        Returns:
            list: 错误信息列表
        """
        errors = []
        try:
            validate(instance=plan_data, schema=self.schema)
        except ValidationError:
            validator = jsonschema.Draft202012Validator(self.schema)
            for error in validator.iter_errors(plan_data):
                errors.append({
                    "path": list(error.path),
                    "message": error.message,
                    "failed_value": error.instance
                })
        return errors


def load_schema() -> Dict[str, Any]:
    """加载预案Schema"""
    return PLAN_JSON_SCHEMA


def save_schema(file_path: Path):
    """保存Schema到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(PLAN_JSON_SCHEMA, f, indent=2, ensure_ascii=False)


# 示例预案JSON数据
EXAMPLE_PLAN_JSON = {
    "plan_id": "dc_limit_fault",
    "title": "设备故障直流限额计算预案",
    "description": "计算设备故障情况下的直流传输限额",
    "version": "1.0",
    "variables": [
        {
            "name": "送端电网限额",
            "symbol": "P_max_send",
            "unit": "MW",
            "description": "送端电网的传输限额"
        },
        {
            "name": "受端电网限额",
            "symbol": "P_max_receive", 
            "unit": "MW",
            "description": "受端电网的传输限额"
        },
        {
            "name": "电网运行方式传输限额",
            "symbol": "P_max_net",
            "unit": "MW",
            "formula": "min(P_max_send, P_max_receive)"
        },
        {
            "name": "设备故障直流限额",
            "symbol": "P_max_device",
            "unit": "MW",
            "formula": "min(P_max_net, P_dcsystem)"
        }
    ],
    "steps": [
        {
            "id": "step1",
            "type": "rag",
            "description": "查询停运设备影响的直流线路，并判定其位于送端还是受端",
            "query": "判定停运设备影响直流线路送/受端",
            "inputs": {
                "device": "{设备}",
                "line": "{直流线路}"
            },
            "outputs": ["dc_line", "side_info"]
        },
        {
            "id": "tool_send",
            "type": "tool",
            "description": "查询送端限额",
            "tool_name": "query_send_limit",
            "inputs": {
                "line": "{dc_line}"
            },
            "outputs": ["P_max_send"]
        },
        {
            "id": "tool_recv",
            "type": "tool",
            "description": "查询受端限额",
            "tool_name": "query_recv_limit",
            "inputs": {
                "line": "{dc_line}"
            },
            "outputs": ["P_max_receive"]
        },
        {
            "id": "compute_net",
            "type": "compute",
            "description": "计算电网运行方式传输限额",
            "formula": "min(P_max_send, P_max_receive)",
            "inputs": {
                "P_max_send": "{P_max_send}",
                "P_max_receive": "{P_max_receive}"
            },
            "outputs": ["P_max_net"]
        },
        {
            "id": "compute_final",
            "type": "compute",
            "description": "计算设备故障直流限额",
            "formula": "min(P_max_net, P_dcsystem)",
            "inputs": {
                "P_max_net": "{P_max_net}",
                "P_dcsystem": "{P_dcsystem}"
            },
            "outputs": ["P_max_device"]
        }
    ],
    "plan_inputs": {
        "device": "故障设备名称",
        "dc_line": "直流线路名称"
    },
    "plan_outputs": ["P_max_device"],
    "tags": ["直流限额", "故障处理", "传输限额"]
}