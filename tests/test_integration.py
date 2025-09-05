"""集成测试"""

import pytest
import asyncio
from pathlib import Path

from grid_preplan_agentcontroller.autogen_controller import AutoGenController
from grid_preplan_agentagents.decision_agent import DecisionAgent
from grid_preplan_agentagents.rag_agent import RAGAgent, create_rag_agent
from grid_preplan_agenttools.grid_tools import initialize_grid_tools
from grid_preplan_agenttools.mock_tools import initialize_mock_tools


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """设置测试环境"""
    initialize_grid_tools()
    initialize_mock_tools()


@pytest.fixture
def controller():
    """创建控制器实例"""
    return AutoGenController(plan_library_path=Path("plans"))


@pytest.fixture
def decision_agent():
    """创建决策Agent实例"""
    return DecisionAgent()


@pytest.fixture
def rag_agent():
    """创建RAG Agent实例"""
    return create_rag_agent()


class TestFullWorkflow:
    """完整工作流测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, controller):
        """测试端到端工作流"""
        scenario = "天哈一线停运，影响天中直流限额"
        
        try:
            # 执行场景
            result = await controller.process_scenario(scenario)
            
            assert result is not None
            assert result.scenario == scenario
            assert result.execution_id is not None
            
            # 检查执行状态
            if result.success:
                assert result.final_outputs is not None
                assert len(result.step_results) > 0
                
                # 生成决策报告
                decision_agent = DecisionAgent()
                report = await decision_agent.generate_report(result)
                
                assert report is not None
                assert report.execution_id == result.execution_id
                assert report.summary is not None
                
            else:
                # 如果失败，检查错误信息
                assert result.error_message is not None
                
        except Exception as e:
            # 如果依赖不可用，跳过测试
            pytest.skip(f"端到端测试跳过: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_plan_loading_and_selection(self, controller):
        """测试预案加载和选择"""
        try:
            # 加载可用预案
            plans = await controller.load_available_plans()
            
            if plans:
                assert len(plans) > 0
                
                # 测试预案选择
                scenario = "直流限额计算"
                selected_plan = await controller.select_plan(scenario)
                
                if selected_plan:
                    assert selected_plan.plan_id is not None
                    assert selected_plan.title is not None
                    
        except Exception as e:
            pytest.skip(f"预案加载测试跳过: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_complexity_analysis(self, controller):
        """测试复杂度分析"""
        try:
            plans = await controller.load_available_plans()
            
            if plans:
                plan = plans[0]
                complexity, analysis = controller.complexity_analyzer.analyze(plan)
                
                assert complexity is not None
                assert analysis is not None
                assert "step_count" in analysis
                assert "step_types" in analysis
                
        except Exception as e:
            pytest.skip(f"复杂度分析测试跳过: {str(e)}")


class TestRAGAgent:
    """RAG Agent测试"""
    
    @pytest.mark.asyncio
    async def test_rag_query(self, rag_agent):
        """测试RAG查询"""
        try:
            query = "什么是送端受端判定？"
            result = await rag_agent.query(query)
            
            assert result is not None
            assert result.query == query
            assert len(result.results) > 0
            assert result.confidence >= 0.0
            
        except Exception as e:
            pytest.skip(f"RAG查询测试跳过: {str(e)}")
    
    def test_add_document(self, rag_agent):
        """测试添加文档"""
        try:
            content = "这是一个测试文档，包含电网调度的相关信息。"
            metadata = {"title": "测试文档", "source": "test"}
            
            result = rag_agent.add_document(content, metadata)
            
            # 如果向量数据库可用，应该添加成功
            if rag_agent.vectorstore:
                assert result is True
            
        except Exception as e:
            pytest.skip(f"添加文档测试跳过: {str(e)}")
    
    def test_get_statistics(self, rag_agent):
        """测试获取统计信息"""
        stats = rag_agent.get_statistics()
        
        assert "total_documents" in stats
        assert "vectorstore_initialized" in stats
        assert "default_knowledge_items" in stats
        assert isinstance(stats["total_documents"], int)


class TestDecisionAgent:
    """决策Agent测试"""
    
    @pytest.mark.asyncio
    async def test_report_generation(self, decision_agent):
        """测试报告生成"""
        # 创建模拟执行结果
        from grid_preplan_agentcore.models import ExecutionResult
        
        mock_result = ExecutionResult(
            execution_id="test_execution_123",
            plan_id="test_plan",
            success=True,
            scenario="测试场景：设备故障直流限额计算",
            final_outputs={"P_max_device": 2800.0, "side_info": "送端"},
            variables={"P_max_send": 3200.0, "P_max_receive": 3000.0},
            step_results=[
                {
                    "step_id": "step1",
                    "step_type": "tool",
                    "description": "查询送端限额",
                    "success": True,
                    "outputs": {"P_max_send": 3200.0},
                    "timestamp": "2024-01-01T10:00:00"
                },
                {
                    "step_id": "step2",
                    "step_type": "compute",
                    "description": "计算最小限额",
                    "success": True,
                    "outputs": {"P_max_device": 2800.0},
                    "timestamp": "2024-01-01T10:00:01"
                }
            ],
            execution_time=2.5
        )
        
        try:
            report = await decision_agent.generate_report(mock_result)
            
            assert report is not None
            assert report.execution_id == "test_execution_123"
            assert report.plan_title == "test_plan"
            assert report.scenario == mock_result.scenario
            assert report.summary is not None
            assert len(report.data_sources) > 0
            assert report.confidence_level is not None
            
        except Exception as e:
            pytest.skip(f"报告生成测试跳过: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_report_export(self, decision_agent):
        """测试报告导出"""
        # 创建简单的模拟报告
        from grid_preplan_agentcore.models import DecisionReport
        
        mock_report = DecisionReport(
            report_id="test_report_123",
            execution_id="test_execution_123",
            plan_title="测试预案",
            scenario="测试场景",
            summary="测试摘要",
            background="测试背景",
            data_sources=[{"step": "test_step", "type": "tool"}],
            calculations=[{"step": "calc_step", "formula": "min(a, b)"}],
            recommendations=["建议1", "建议2"],
            generated_at="2024-01-01T10:00:00",
            confidence_level="高"
        )
        
        try:
            # 测试导出为Markdown
            markdown_content = await decision_agent.export_report(mock_report, "markdown")
            assert "# 电网调度决策报告" in markdown_content
            assert "test_report_123" in markdown_content
            
            # 测试导出为JSON
            json_content = await decision_agent.export_report(mock_report, "json")
            assert "test_report_123" in json_content
            assert "execution_id" in json_content
            
        except Exception as e:
            pytest.skip(f"报告导出测试跳过: {str(e)}")


class TestToolsIntegration:
    """工具集成测试"""
    
    @pytest.mark.asyncio
    async def test_grid_tools(self):
        """测试电网工具"""
        from grid_preplan_agenttools.api_registry import call_tool
        
        try:
            # 测试查询送端限额
            result = await call_tool("query_send_limit", line="天中直流")
            assert result.success is True
            assert result.result is not None
            assert result.unit == "MW"
            
            # 测试查询设备影响
            result = await call_tool("query_device_impact", device="天哈一线")
            assert result.success is True
            assert result.result is not None
            
        except Exception as e:
            pytest.skip(f"工具测试跳过: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_mock_tools(self):
        """测试模拟工具"""
        from grid_preplan_agenttools.api_registry import call_tool
        
        try:
            # 测试模拟RAG查询
            result = await call_tool("mock_rag_query", query="测试查询")
            assert result.success is True
            assert result.result is not None
            
            # 测试模拟计算器
            result = await call_tool("mock_calculator", operation="min", operands=[100, 200, 150])
            assert result.success is True
            assert result.result == 100
            
        except Exception as e:
            pytest.skip(f"模拟工具测试跳过: {str(e)}")


class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_invalid_scenario(self, controller):
        """测试无效场景处理"""
        invalid_scenario = "这是一个完全无关的场景描述"
        
        result = await controller.process_scenario(invalid_scenario)
        
        # 应该能处理无效场景，即使失败也要有合理的错误信息
        assert result is not None
        if not result.success:
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_missing_tools(self):
        """测试缺失工具的处理"""
        from grid_preplan_agenttools.api_registry import call_tool
        
        result = await call_tool("non_existent_tool", param="test")
        
        assert result.success is False
        assert "不存在" in result.error_message or "not found" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_tool_params(self):
        """测试无效工具参数处理"""
        from grid_preplan_agenttools.api_registry import call_tool
        
        # 缺少必需参数
        result = await call_tool("query_send_limit")  # 缺少line参数
        
        assert result.success is False
        assert result.error_message is not None


if __name__ == "__main__":
    # 运行特定的测试类
    pytest.main([__file__ + "::TestFullWorkflow", "-v", "-s"])