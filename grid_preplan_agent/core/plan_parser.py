import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .models import PlanJSON, PlanStep, Variable, StepType
from .plan_schema import PlanSchemaValidator, EXAMPLE_PLAN_JSON
from ..utils.logger import logger


class PlanParser:
    """预案解析器：将自然语言预案文本转换为结构化的Plan JSON"""
    
    def __init__(self, llm_model: str = "gpt-4-turbo-preview"):
        """初始化解析器
        
        Args:
            llm_model: 使用的LLM模型
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.validator = PlanSchemaValidator()
        self.few_shot_examples = self._load_few_shot_examples()
        
    def parse(self, plan_text: str, plan_id: Optional[str] = None) -> PlanJSON:
        """解析预案文本为Plan JSON
        
        Args:
            plan_text: 预案文本内容
            plan_id: 可选的预案ID，如未提供则自动生成
            
        Returns:
            PlanJSON: 解析后的预案JSON对象
            
        Raises:
            ValueError: 解析失败时抛出
        """
        if not plan_id:
            plan_id = self._generate_plan_id(plan_text)
            
        logger.info(f"开始解析预案: {plan_id}")
        
        try:
            # 1. 预处理文本
            cleaned_text = self._preprocess_text(plan_text)
            
            # 2. 使用LLM进行结构化提取
            structured_data = self._llm_extract(cleaned_text, plan_id)
            
            # 3. 后处理和验证
            plan_json = self._post_process(structured_data, plan_id)
            
            # 4. Schema验证
            self.validator.validate(plan_json.dict())
            
            logger.info(f"预案解析成功: {plan_id}")
            return plan_json
            
        except Exception as e:
            logger.error(f"预案解析失败: {plan_id}, 错误: {str(e)}")
            raise ValueError(f"预案解析失败: {str(e)}")
    
    def parse_file(self, file_path: Path, plan_id: Optional[str] = None) -> PlanJSON:
        """从文件解析预案
        
        Args:
            file_path: 预案文件路径
            plan_id: 可选的预案ID
            
        Returns:
            PlanJSON: 解析后的预案JSON对象
        """
        if not file_path.exists():
            raise FileNotFoundError(f"预案文件不存在: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            plan_text = f.read()
            
        if not plan_id:
            plan_id = file_path.stem
            
        return self.parse(plan_text, plan_id)
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除注释区
        lines = text.split('\n')
        cleaned_lines = []
        in_comment_block = False
        
        for line in lines:
            if line.strip().startswith('# ========='):
                in_comment_block = not in_comment_block
                continue
            if not in_comment_block:
                cleaned_lines.append(line)
        
        # 清理空行和多余空格
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # 合并多个空行
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _llm_extract(self, plan_text: str, plan_id: str) -> Dict[str, Any]:
        """使用LLM提取结构化信息
        
        Args:
            plan_text: 预案文本
            plan_id: 预案ID
            
        Returns:
            Dict[str, Any]: 结构化数据
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(plan_text, plan_id)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # 尝试提取JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块标记，尝试提取完整的JSON
            json_str = response_text.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"LLM输出不是有效的JSON: {json_str}")
            raise ValueError(f"LLM返回的JSON格式不正确: {str(e)}")
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return f"""你是一个专门解析电网调度预案的AI助手。你的任务是将自然语言描述的预案转换为结构化的JSON格式。

**解析规则：**
1. 提取预案标题和描述
2. 识别步骤序号和描述，确定步骤类型（rag/tool/compute）
3. 提取输入输出变量
4. 解析变量定义和公式
5. 生成符合Schema的JSON格式

**步骤类型判断：**
- **rag**: 查询、判定、检索类步骤（如"查询停运设备"、"判定送/受端"）
- **tool**: 调用工具获取数据（如"查询送端限额"、"获取换流器参数"）
- **compute**: 计算类步骤（如"计算最小值"、"计算传输限额"）

**示例输出格式：**
```json
{EXAMPLE_PLAN_JSON}
```

**重要说明：**
- 必须严格按照JSON Schema格式输出
- 每个步骤必须包含id, type, description, outputs
- 根据步骤类型添加相应字段(query/tool_name/formula)
- 变量定义要包含name, symbol, unit
- 计算公式使用LaTeX格式

请只返回JSON格式的结果，不要包含其他解释文本。"""
    
    def _build_user_prompt(self, plan_text: str, plan_id: str) -> str:
        """构建用户提示"""
        return f"""请解析以下预案文本，生成对应的Plan JSON：

预案ID: {plan_id}
预案内容：
```
{plan_text}
```

请严格按照系统提示中的格式要求，将上述预案转换为JSON格式。特别注意：
1. 正确识别步骤类型（rag/tool/compute）
2. 提取所有变量定义和公式
3. 确保输入输出变量对应关系正确
4. 生成合法的JSON格式"""
    
    def _post_process(self, data: Dict[str, Any], plan_id: str) -> PlanJSON:
        """后处理和数据清理
        
        Args:
            data: LLM输出的结构化数据
            plan_id: 预案ID
            
        Returns:
            PlanJSON: 清理后的预案对象
        """
        # 确保必要字段存在
        data.setdefault("plan_id", plan_id)
        data.setdefault("version", "1.0")
        data.setdefault("variables", [])
        data.setdefault("plan_inputs", {})
        data.setdefault("plan_outputs", [])
        data.setdefault("tags", [])
        
        # 添加时间戳
        current_time = datetime.now().isoformat()
        data.setdefault("created_at", current_time)
        data.setdefault("updated_at", current_time)
        
        # 验证步骤ID唯一性
        step_ids = [step.get("id") for step in data.get("steps", [])]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("步骤ID必须唯一")
        
        # 清理空字符串
        for step in data.get("steps", []):
            for key, value in step.items():
                if isinstance(value, str) and not value.strip():
                    step[key] = None
        
        try:
            return PlanJSON(**data)
        except ValidationError as e:
            logger.error(f"Pydantic验证失败: {e}")
            raise ValueError(f"数据格式验证失败: {str(e)}")
    
    def _generate_plan_id(self, plan_text: str) -> str:
        """从预案文本生成ID
        
        Args:
            plan_text: 预案文本
            
        Returns:
            str: 生成的预案ID
        """
        # 尝试从标题提取ID
        lines = plan_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # 第一个非注释行通常是标题
                # 简化标题作为ID
                title = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '_', line)
                title = re.sub(r'_+', '_', title).strip('_')
                if title:
                    return title[:20]  # 限制长度
                break
        
        # 如果无法从标题提取，使用时间戳
        return f"plan_{uuid.uuid4().hex[:8]}"
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """加载Few-shot示例"""
        # 这里可以从文件或数据库加载更多示例
        return [EXAMPLE_PLAN_JSON]
    
    def validate_plan(self, plan_json: PlanJSON) -> bool:
        """验证预案JSON
        
        Args:
            plan_json: 预案JSON对象
            
        Returns:
            bool: 验证是否通过
        """
        try:
            self.validator.validate(plan_json.dict())
            return True
        except Exception as e:
            logger.error(f"预案验证失败: {str(e)}")
            return False
    
    def get_validation_errors(self, plan_json: PlanJSON) -> List[Dict[str, Any]]:
        """获取验证错误详情
        
        Args:
            plan_json: 预案JSON对象
            
        Returns:
            List[Dict[str, Any]]: 错误详情列表
        """
        return self.validator.get_validation_errors(plan_json.dict())


def create_parser(model: str = "gpt-4-turbo-preview") -> PlanParser:
    """创建预案解析器实例
    
    Args:
        model: LLM模型名称
        
    Returns:
        PlanParser: 解析器实例
    """
    return PlanParser(model)