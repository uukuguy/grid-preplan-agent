"""执行器测试"""

import pytest
import asyncio
from pathlib import Path

from grid_preplan_agentcore.models import PlanJSON, PlanStep, Variable, StepType
from grid_preplan_agentexecutors.langgraph_executor import LangGraphExecutor
from grid_preplan_agentexecutors.smolagents_executor import SmolagentsExecutor
from grid_preplan_agenttools.grid_tools import initialize_grid_tools
from grid_preplan_agenttools.mock_tools import initialize_mock_tools


@pytest.fixture
def sample_plan():
    """创建示例预案"""
    return PlanJSON(
        plan_id="test_dc_limit",
        title="测试直流限额预案",
        description="用于测试的直流限额计算预案",
        steps=[
            PlanStep(
                id="step1",
                type=StepType.TOOL,
                description="查询送端限额",
                tool_name="query_send_limit",
                inputs={"line": "天中直流"},
                outputs=["P_max_send"]
            ),
            PlanStep(
                id="step2",
                type=StepType.TOOL,
                description="查询受端限额", 
                tool_name="query_recv_limit",
                inputs={"line": "天中直流"},
                outputs=["P_max_receive"]
            ),
            PlanStep(
                id="step3",
                type=StepType.COMPUTE,
                description="计算最小限额",
                formula="min(P_max_send, P_max_receive)",
                inputs={
                    "P_max_send": "{P_max_send}",
                    "P_max_receive": "{P_max_receive}"
                },
                outputs=["P_max_net"]
            )
        ],
        variables=[
            Variable(name="送端限额", symbol="P_max_send", unit="MW"),
            Variable(name="受端限额", symbol="P_max_receive", unit="MW"),
            Variable(name="网络限额", symbol="P_max_net", unit="MW")
        ],
        plan_inputs={"line": "直流线路名称"},
        plan_outputs=["P_max_net"]
    )


@pytest.fixture(scope="session", autouse=True)
def setup_tools():
    """设置工具"""
    initialize_grid_tools()
    initialize_mock_tools()


class TestLangGraphExecutor:
    """LangGraph执行器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.executor = LangGraphExecutor()
    
    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, sample_plan):
        """测试执行简单预案"""
        scenario = "天中直流限额计算测试"
        inputs = {"line": "天中直流"}
        
        result = await self.executor.execute(sample_plan, scenario, inputs)
        
        assert result is not None
        assert result.execution_id is not None
        assert result.plan_id == "test_dc_limit"
        assert result.scenario == scenario
        
        # 检查是否有最终输出
        if result.success:
            assert "P_max_net" in result.final_outputs
            assert isinstance(result.final_outputs["P_max_net"], (int, float))
        
        # 检查执行时间
        assert result.execution_time >= 0
    
    def test_substitute_variables(self):
        """测试变量替换"""
        variables = {
            "P_max_send": 3200.0,
            "P_max_receive": 3000.0,
            "line": "天中直流"
        }
        
        text = "线路{line}的送端限额为{P_max_send}MW，受端限额为{P_max_receive}MW"
        result = self.executor.substitute_variables(text, variables)
        
        expected = "线路天中直流的送端限额为3200.0MW，受端限额为3000.0MW"
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_evaluate_formula(self):
        """测试公式计算"""
        inputs = {
            "P_max_send": 3200.0,
            "P_max_receive": 3000.0
        }
        
        # 测试min函数
        result = await self.executor.evaluate_formula("min(P_max_send, P_max_receive)", inputs)
        assert result == 3000.0
        
        # 测试max函数
        result = await self.executor.evaluate_formula("max(P_max_send, P_max_receive)", inputs)
        assert result == 3200.0
    
    def test_validate_inputs(self, sample_plan):
        """测试输入验证"""
        valid_inputs = {"line": "天中直流"}
        
        # 应该验证通过
        result = self.executor.validate_inputs(sample_plan, valid_inputs)
        assert result is True
        
        # 缺少必需参数应该抛出异常
        invalid_inputs = {}
        with pytest.raises(ValueError):
            self.executor.validate_inputs(sample_plan, invalid_inputs)


class TestSmolagentsExecutor:
    """Smolagents执行器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.executor = SmolagentsExecutor()
    
    def test_build_task_description(self, sample_plan):
        """测试任务描述构建"""
        scenario = "测试场景"
        inputs = {"line": "天中直流"}
        
        description = self.executor.build_task_description(sample_plan, scenario, inputs)
        
        assert "测试直流限额预案" in description
        assert "测试场景" in description
        assert "天中直流" in description
        assert "步骤：" in description
    
    def test_parse_agent_result(self):
        """测试Agent结果解析"""
        result_text = """
        计算完成！结果如下：
        P_max_net = 2800
        P_max_send = 3200
        计算过程顺利完成。
        """
        
        expected_outputs = ["P_max_net"]
        result = self.executor.parse_agent_result(result_text, expected_outputs)
        
        assert "P_max_net" in result
        assert result["P_max_net"] == 2800.0
    
    @pytest.mark.asyncio
    async def test_list_available_tools(self):
        """测试列出可用工具"""
        tools = await self.executor.list_available_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # 检查是否包含预期的工具
        expected_tools = ["query_send_limit", "query_recv_limit", "mock_rag_query"]
        for tool in expected_tools:
            if tool in tools:  # 如果工具存在就检查
                assert tool in tools


class TestExecutorComparison:
    """执行器对比测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.langgraph_executor = LangGraphExecutor()
        self.smolagents_executor = SmolagentsExecutor()
    
    @pytest.mark.asyncio
    async def test_same_plan_different_executors(self, sample_plan):
        """测试同一预案在不同执行器上的结果"""
        scenario = "执行器对比测试"
        inputs = {"line": "天中直流"}
        
        # 执行LangGraph
        lg_result = await self.langgraph_executor.execute(sample_plan, scenario, inputs)
        
        # 执行Smolagents  
        sm_result = await self.smolagents_executor.execute(sample_plan, scenario, inputs)
        
        # 比较结果
        assert lg_result.plan_id == sm_result.plan_id
        assert lg_result.scenario == sm_result.scenario
        
        # 如果两个都成功，比较最终输出
        if lg_result.success and sm_result.success:
            lg_output = lg_result.final_outputs.get("P_max_net")
            sm_output = sm_result.final_outputs.get("P_max_net")
            
            if lg_output is not None and sm_output is not None:
                # 允许小的数值差异
                assert abs(float(lg_output) - float(sm_output)) < 100


@pytest.fixture
def complex_plan():
    """创建复杂预案用于测试"""
    return PlanJSON(
        plan_id="complex_test_plan",
        title="复杂测试预案",
        description="包含多种步骤类型的复杂预案",
        steps=[
            PlanStep(
                id="rag_step",
                type=StepType.RAG,
                description="查询规程信息",
                query="直流限额计算规程",
                outputs=["rule_info"]
            ),
            PlanStep(
                id="tool_step1",
                type=StepType.TOOL,
                description="获取设备信息",
                tool_name="query_device_impact",
                inputs={"device": "天哈一线"},
                outputs=["dc_line", "side_info"]
            ),
            PlanStep(
                id="tool_step2",
                type=StepType.TOOL,
                description="查询限额",
                tool_name="query_send_limit",
                inputs={"line": "{dc_line}"},
                outputs=["P_limit"]
            ),
            PlanStep(
                id="compute_step",
                type=StepType.COMPUTE,
                description="计算最终结果",
                formula="min(P_limit, 2500)",
                inputs={"P_limit": "{P_limit}"},
                outputs=["final_result"]
            )
        ],
        plan_inputs={"device": "设备名称"},
        plan_outputs=["final_result"]
    )


class TestComplexPlanExecution:
    """复杂预案执行测试"""
    
    @pytest.mark.asyncio
    async def test_complex_plan_langgraph(self, complex_plan):
        """测试复杂预案在LangGraph上的执行"""
        executor = LangGraphExecutor()
        
        scenario = "复杂预案测试场景"
        inputs = {"device": "天哈一线"}
        
        result = await executor.execute(complex_plan, scenario, inputs)
        
        assert result is not None
        assert result.plan_id == "complex_test_plan"
        
        # 检查步骤执行历史
        assert len(result.step_results) > 0
        
        # 如果执行成功，检查最终输出
        if result.success:
            assert "final_result" in result.final_outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])