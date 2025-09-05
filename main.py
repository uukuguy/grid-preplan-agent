"""电网调度辅助决策智能体主程序"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from grid_preplan_agentcontroller.autogen_controller import AutoGenController
from grid_preplan_agentagents.decision_agent import DecisionAgent
from grid_preplan_agenttools.grid_tools import initialize_grid_tools
from grid_preplan_agenttools.mock_tools import initialize_mock_tools
from grid_preplan_agentutils.logger import setup_logger, logger


async def main():
    """主程序入口"""
    print("🚀 启动电网调度辅助决策智能体系统")
    
    # 设置日志
    logger = setup_logger("main", "INFO")
    logger.info("系统初始化开始")
    
    # 初始化工具
    print("🔧 初始化工具系统...")
    initialize_grid_tools()
    initialize_mock_tools()
    logger.info("工具系统初始化完成")
    
    # 创建控制器
    print("🎛️ 创建AutoGen控制器...")
    controller = AutoGenController(
        plan_library_path=Path("plans")
    )
    
    # 创建决策Agent
    decision_agent = DecisionAgent()
    
    print("\n✅ 系统初始化完成！\n")
    
    # 示例场景测试
    scenarios = [
        {
            "scenario": "天哈一线停运，需要计算天中直流限额",
            "inputs": {"device": "天哈一线"}
        },
        {
            "scenario": "华中换流站故障，影响直流传输",
            "inputs": {"device": "华中换流站"}
        }
    ]
    
    for i, test_case in enumerate(scenarios, 1):
        print(f"📋 执行测试场景 {i}: {test_case['scenario']}")
        print("-" * 50)
        
        try:
            # 处理场景
            result = await controller.process_scenario(
                test_case["scenario"],
                test_case["inputs"]
            )
            
            # 显示执行结果
            if result.success:
                print(f"✅ 执行成功 (耗时: {result.execution_time:.2f}秒)")
                print(f"📊 最终结果: {json.dumps(result.final_outputs, ensure_ascii=False, indent=2)}")
                
                # 生成决策报告
                print("📝 生成决策报告...")
                report = await decision_agent.generate_report(result)
                
                # 导出报告为Markdown
                report_path = Path(f"output/report_{result.execution_id}.md")
                report_path.parent.mkdir(exist_ok=True)
                
                await decision_agent.export_report(
                    report, 
                    format_type="markdown",
                    output_path=report_path
                )
                
                print(f"📄 报告已保存至: {report_path}")
                
            else:
                print(f"❌ 执行失败: {result.error_message}")
                if result.failed_step:
                    print(f"🚫 失败步骤: {result.failed_step}")
            
            print(f"📈 执行历史: {len(result.step_results)} 个步骤")
            
        except Exception as e:
            print(f"💥 场景执行异常: {str(e)}")
            logger.error(f"场景执行异常: {str(e)}", exc_info=True)
        
        print("\n")
    
    # 显示系统统计
    print("📊 系统统计信息:")
    print(f"   - 可用预案数量: {len(controller.list_available_plans())}")
    print(f"   - 执行历史记录: {len(controller.execution_history)}")
    
    from grid_preplan_agenttools.api_registry import tool_registry
    print(f"   - 注册工具数量: {len(tool_registry.list_tools())}")
    
    print("\n🎉 演示完成！")


async def interactive_mode():
    """交互式模式"""
    print("🔄 进入交互式模式")
    print("输入 'exit' 退出，输入 'help' 查看帮助")
    
    # 初始化系统
    initialize_grid_tools()
    initialize_mock_tools()
    
    controller = AutoGenController(plan_library_path=Path("plans"))
    decision_agent = DecisionAgent()
    
    while True:
        try:
            scenario = input("\n请输入场景描述: ").strip()
            
            if scenario.lower() == 'exit':
                print("👋 再见！")
                break
            
            if scenario.lower() == 'help':
                print_help()
                continue
            
            if not scenario:
                print("❌ 场景描述不能为空")
                continue
            
            print(f"🔄 处理场景: {scenario}")
            
            # 处理场景
            result = await controller.process_scenario(scenario)
            
            # 显示结果
            if result.success:
                print(f"✅ 执行成功!")
                print(f"📊 结果: {json.dumps(result.final_outputs, ensure_ascii=False)}")
                
                # 询问是否生成报告
                generate_report = input("是否生成详细报告? (y/N): ").strip().lower()
                if generate_report in ['y', 'yes']:
                    report = await decision_agent.generate_report(result)
                    print("\n📋 决策报告:")
                    print("=" * 50)
                    print(f"执行摘要: {report.summary}")
                    print(f"置信等级: {report.confidence_level}")
                    if report.recommendations:
                        print("建议措施:")
                        for i, rec in enumerate(report.recommendations, 1):
                            print(f"  {i}. {rec}")
                    print("=" * 50)
            
            else:
                print(f"❌ 执行失败: {result.error_message}")
        
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"💥 处理异常: {str(e)}")


def print_help():
    """打印帮助信息"""
    print("""
📚 使用帮助:

支持的场景类型:
- 设备故障直流限额计算
- 断面功率校核
- 备用容量分析

示例场景:
- "天哈一线停运，需要计算天中直流限额"
- "华中换流站故障，影响直流传输"
- "设备检修期间的功率限制"

命令:
- exit: 退出程序
- help: 显示此帮助信息

🔧 系统功能:
- 自动解析预案文本
- 智能选择执行器(LangGraph/Smolagents)  
- 生成专业决策报告
- 支持多种导出格式
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="电网调度辅助决策智能体")
    parser.add_argument(
        "--mode", 
        choices=["demo", "interactive"], 
        default="demo",
        help="运行模式: demo(演示模式) 或 interactive(交互模式)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        asyncio.run(main())
    elif args.mode == "interactive":
        asyncio.run(interactive_mode())