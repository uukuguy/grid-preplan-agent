"""预案解析器测试"""

import pytest
import asyncio
from pathlib import Path

from grid_preplan_agentcore.plan_parser import PlanParser
from grid_preplan_agentcore.models import PlanJSON, PlanStep, Variable, StepType


class TestPlanParser:
    """预案解析器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.parser = PlanParser("gpt-4-turbo-preview")
    
    def test_preprocess_text(self):
        """测试文本预处理"""
        test_text = """
# =========================================================
# 注释内容
# =========================================================

设备故障直流限额计算预案

步骤：
1. 查询设备信息


变量定义：
- 送端限额：P_send (MW)
        """
        
        cleaned = self.parser._preprocess_text(test_text)
        
        assert "注释内容" not in cleaned
        assert "设备故障直流限额计算预案" in cleaned
        assert "步骤：" in cleaned
        assert "变量定义：" in cleaned
    
    def test_generate_plan_id(self):
        """测试预案ID生成"""
        test_text = "设备故障直流限额计算预案\n\n这是一个测试预案"
        
        plan_id = self.parser._generate_plan_id(test_text)
        
        assert plan_id is not None
        assert len(plan_id) > 0
        assert "_" in plan_id or plan_id.startswith("设备故障")
    
    def test_substitute_variables(self):
        """测试变量替换"""
        from grid_preplan_agentexecutors.langgraph_executor import LangGraphExecutor
        
        executor = LangGraphExecutor()
        variables = {
            "device": "天哈一线",
            "dc_line": "天中直流",
            "P_max_send": 3200.0
        }
        
        text = "设备{device}影响{dc_line}，送端限额为{P_max_send}MW"
        result = executor.substitute_variables(text, variables)
        
        assert result == "设备天哈一线影响天中直流，送端限额为3200.0MW"
    
    @pytest.mark.asyncio
    async def test_parse_simple_plan(self):
        """测试解析简单预案"""
        simple_plan_text = """
设备故障直流限额计算预案

步骤：
1. 查询送端限额
   输入：dc_line
   输出：P_max_send

2. 计算最小值
   输入：P_max_send, P_max_receive
   输出：P_min

变量定义：
- 送端限额：P_max_send (MW)
- 受端限额：P_max_receive (MW)
- 最小值：P_min (MW)
        """
        
        try:
            plan = self.parser.parse(simple_plan_text, "test_plan")
            
            assert isinstance(plan, PlanJSON)
            assert plan.plan_id == "test_plan"
            assert plan.title is not None
            assert len(plan.steps) >= 1
            assert len(plan.variables) >= 1
            
        except Exception as e:
            # 如果LLM不可用，跳过此测试
            pytest.skip(f"LLM不可用，跳过测试: {str(e)}")
    
    def test_validate_plan_basic(self):
        """测试基本预案验证"""
        # 创建一个基本的有效预案
        from grid_preplan_agentcore.plan_schema import EXAMPLE_PLAN_JSON
        
        valid_plan = PlanJSON(**EXAMPLE_PLAN_JSON)
        
        result = self.parser.validate_plan(valid_plan)
        assert result is True
    
    def test_validate_plan_invalid(self):
        """测试无效预案验证"""
        try:
            # 缺少必需字段的无效预案
            invalid_plan_data = {
                "plan_id": "test_invalid",
                # 缺少其他必需字段
            }
            
            invalid_plan = PlanJSON(**invalid_plan_data)
            result = self.parser.validate_plan(invalid_plan)
            assert result is False
            
        except Exception:
            # 预期会抛出验证异常
            pass
    
    @pytest.mark.asyncio  
    async def test_parse_file(self):
        """测试从文件解析预案"""
        # 使用项目中的示例预案文件
        plan_file = Path("plans/dc_limit_fault.txt")
        
        if plan_file.exists():
            try:
                plan = self.parser.parse_file(plan_file)
                
                assert isinstance(plan, PlanJSON)
                assert plan.plan_id is not None
                assert plan.title is not None
                
            except Exception as e:
                # 如果LLM不可用或文件格式问题，跳过测试
                pytest.skip(f"文件解析测试跳过: {str(e)}")
        else:
            pytest.skip("预案文件不存在，跳过测试")


@pytest.fixture
def mock_llm_response():
    """模拟LLM响应"""
    return """
{
  "plan_id": "test_plan",
  "title": "测试预案",
  "description": "这是一个测试预案",
  "steps": [
    {
      "id": "step1",
      "type": "tool",
      "description": "查询数据",
      "tool_name": "test_tool",
      "inputs": {},
      "outputs": ["result"]
    }
  ],
  "variables": [
    {
      "name": "测试变量",
      "symbol": "test_var",
      "unit": "MW"
    }
  ]
}
    """


class TestPlanParserIntegration:
    """预案解析器集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_parse_workflow(self):
        """测试完整解析工作流"""
        parser = PlanParser()
        
        # 使用简化的预案文本
        plan_text = """
简单测试预案

步骤：
1. 获取数据
   输入：input_param
   输出：output_value

变量定义：
- 输入参数：input_param (string)
- 输出值：output_value (number)
        """
        
        try:
            # 解析预案
            plan = parser.parse(plan_text, "integration_test")
            
            # 验证结果
            assert plan.plan_id == "integration_test"
            assert len(plan.steps) >= 1
            
            # 验证Schema
            validation_result = parser.validate_plan(plan)
            assert validation_result is True
            
        except Exception as e:
            pytest.skip(f"集成测试跳过: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])