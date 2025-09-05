"""决策报告生成Agent"""

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.models import ExecutionResult, DecisionReport
from ..utils.logger import logger


class DecisionAgent:
    """决策报告生成Agent"""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        """初始化决策Agent
        
        Args:
            model: 使用的LLM模型
        """
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.report_templates = self._load_report_templates()
    
    async def generate_report(
        self,
        execution_result: ExecutionResult,
        report_type: str = "standard"
    ) -> DecisionReport:
        """生成决策报告
        
        Args:
            execution_result: 执行结果
            report_type: 报告类型
            
        Returns:
            DecisionReport: 决策报告
        """
        logger.info(f"生成决策报告: {execution_result.execution_id}")
        
        try:
            # 分析执行结果
            analysis = await self.analyze_execution_result(execution_result)
            
            # 生成报告内容
            content = await self.generate_report_content(execution_result, analysis)
            
            # 构建报告对象
            report = self._build_decision_report(
                execution_result, 
                analysis, 
                content
            )
            
            logger.info(f"决策报告生成完成: {report.report_id}")
            return report
            
        except Exception as e:
            logger.error(f"决策报告生成失败: {str(e)}")
            raise
    
    async def analyze_execution_result(
        self, 
        execution_result: ExecutionResult
    ) -> Dict[str, Any]:
        """分析执行结果
        
        Args:
            execution_result: 执行结果
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        analysis = {
            "success_rate": 1.0 if execution_result.success else 0.0,
            "execution_efficiency": self._analyze_efficiency(execution_result),
            "data_reliability": self._analyze_data_reliability(execution_result),
            "result_confidence": self._analyze_confidence(execution_result),
            "risk_factors": self._identify_risk_factors(execution_result),
            "key_findings": self._extract_key_findings(execution_result),
            "performance_metrics": self._calculate_performance_metrics(execution_result)
        }
        
        return analysis
    
    async def generate_report_content(
        self,
        execution_result: ExecutionResult,
        analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """生成报告内容
        
        Args:
            execution_result: 执行结果
            analysis: 分析结果
            
        Returns:
            Dict[str, str]: 报告内容各部分
        """
        # 构建LLM提示
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(execution_result, analysis)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 调用LLM生成内容
        response = await self.llm.ainvoke(messages)
        response_text = response.content
        
        # 解析生成的内容
        content = self._parse_generated_content(response_text)
        
        return content
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个专业的电网调度决策分析专家。你的任务是根据预案执行结果生成专业的决策分析报告。

报告要求：
1. 客观、准确地分析执行结果
2. 提供具体的数值和技术细节
3. 识别潜在风险和问题
4. 给出可操作的建议措施
5. 使用专业的电力系统术语

报告结构：
- 执行摘要：简明扼要的结论和关键数据
- 技术分析：详细的技术分析和计算过程
- 风险评估：识别的风险因素和影响评估
- 操作建议：具体的操作建议和注意事项
- 监控要点：需要持续监控的关键指标

请确保报告内容专业、准确、实用。"""
    
    def _build_user_prompt(
        self,
        execution_result: ExecutionResult,
        analysis: Dict[str, Any]
    ) -> str:
        """构建用户提示"""
        prompt_lines = [
            "请根据以下预案执行结果生成决策分析报告：",
            "",
            f"场景描述: {execution_result.scenario}",
            f"预案ID: {execution_result.plan_id}",
            f"执行状态: {'成功' if execution_result.success else '失败'}",
            f"执行时间: {execution_result.execution_time:.2f}秒",
            ""
        ]
        
        if execution_result.success:
            prompt_lines.extend([
                "执行结果:",
                json.dumps(execution_result.final_outputs, ensure_ascii=False, indent=2),
                ""
            ])
        else:
            prompt_lines.extend([
                f"失败原因: {execution_result.error_message}",
                f"失败步骤: {execution_result.failed_step}",
                ""
            ])
        
        prompt_lines.extend([
            "执行步骤详情:",
            json.dumps(execution_result.step_results, ensure_ascii=False, indent=2),
            "",
            "分析数据:",
            json.dumps(analysis, ensure_ascii=False, indent=2),
            "",
            "请生成包含以下部分的报告:",
            "1. 执行摘要",
            "2. 技术分析", 
            "3. 风险评估",
            "4. 操作建议",
            "5. 监控要点",
            "",
            "每个部分用明确的标题分隔，内容要专业、详细。"
        ])
        
        return "\n".join(prompt_lines)
    
    def _parse_generated_content(self, response_text: str) -> Dict[str, str]:
        """解析LLM生成的内容
        
        Args:
            response_text: LLM返回的文本
            
        Returns:
            Dict[str, str]: 解析后的内容各部分
        """
        import re
        
        content = {}
        
        # 定义各部分的标题模式
        section_patterns = {
            "summary": r"(?:执行摘要|摘要|概述)[:：]\s*(.*?)(?=(?:技术分析|风险评估|操作建议|监控要点|$))",
            "technical_analysis": r"(?:技术分析|分析)[:：]\s*(.*?)(?=(?:风险评估|操作建议|监控要点|$))",
            "risk_assessment": r"(?:风险评估|风险分析)[:：]\s*(.*?)(?=(?:操作建议|监控要点|$))",
            "recommendations": r"(?:操作建议|建议措施|建议)[:：]\s*(.*?)(?=(?:监控要点|$))",
            "monitoring": r"(?:监控要点|监控|要点)[:：]\s*(.*?)$"
        }
        
        for key, pattern in section_patterns.items():
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                content[key] = match.group(1).strip()
            else:
                content[key] = "未提供相关内容"
        
        return content
    
    def _build_decision_report(
        self,
        execution_result: ExecutionResult,
        analysis: Dict[str, Any],
        content: Dict[str, str]
    ) -> DecisionReport:
        """构建决策报告对象
        
        Args:
            execution_result: 执行结果
            analysis: 分析结果
            content: 报告内容
            
        Returns:
            DecisionReport: 决策报告对象
        """
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        
        # 提取数据来源
        data_sources = []
        for step in execution_result.step_results:
            if step.get("success"):
                data_sources.append({
                    "step_id": step.get("step_id"),
                    "type": step.get("step_type"),
                    "description": step.get("description"),
                    "timestamp": step.get("timestamp"),
                    "source": "系统执行"
                })
        
        # 提取计算过程
        calculations = []
        for step in execution_result.step_results:
            if (step.get("step_type") == "compute" and 
                step.get("success")):
                calculations.append({
                    "step": step.get("step_id"),
                    "description": step.get("description"),
                    "inputs": step.get("inputs", {}),
                    "outputs": step.get("outputs", {}),
                    "formula": step.get("formula", "")
                })
        
        # 生成建议列表
        recommendations = self._extract_recommendations(content.get("recommendations", ""))
        
        # 检查警告
        warnings = analysis.get("risk_factors", [])
        if not execution_result.success:
            warnings.append(f"执行失败: {execution_result.error_message}")
        
        return DecisionReport(
            report_id=report_id,
            execution_id=execution_result.execution_id,
            plan_title=execution_result.plan_id,
            scenario=execution_result.scenario,
            summary=content.get("summary", ""),
            background=self._build_background_section(execution_result),
            data_sources=data_sources,
            calculations=calculations,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat(),
            confidence_level=self._determine_confidence_level(analysis),
            warnings=warnings
        )
    
    def _analyze_efficiency(self, execution_result: ExecutionResult) -> float:
        """分析执行效率
        
        Args:
            execution_result: 执行结果
            
        Returns:
            float: 效率评分 (0-1)
        """
        if not execution_result.success:
            return 0.0
        
        # 基于执行时间和步骤数量评估效率
        step_count = len(execution_result.step_results)
        execution_time = execution_result.execution_time
        
        # 理想情况：每步骤3秒
        ideal_time = step_count * 3
        
        if execution_time <= ideal_time:
            return 1.0
        else:
            return max(0.0, ideal_time / execution_time)
    
    def _analyze_data_reliability(self, execution_result: ExecutionResult) -> float:
        """分析数据可靠性
        
        Args:
            execution_result: 执行结果
            
        Returns:
            float: 可靠性评分 (0-1)
        """
        if not execution_result.success:
            return 0.0
        
        successful_steps = sum(1 for step in execution_result.step_results if step.get("success"))
        total_steps = len(execution_result.step_results)
        
        return successful_steps / total_steps if total_steps > 0 else 0.0
    
    def _analyze_confidence(self, execution_result: ExecutionResult) -> float:
        """分析结果置信度
        
        Args:
            execution_result: 执行结果
            
        Returns:
            float: 置信度评分 (0-1)
        """
        if not execution_result.success:
            return 0.0
        
        # 基于多个因素计算置信度
        factors = [
            self._analyze_efficiency(execution_result),
            self._analyze_data_reliability(execution_result),
            1.0 if execution_result.final_outputs else 0.5
        ]
        
        return sum(factors) / len(factors)
    
    def _identify_risk_factors(self, execution_result: ExecutionResult) -> List[str]:
        """识别风险因素
        
        Args:
            execution_result: 执行结果
            
        Returns:
            List[str]: 风险因素列表
        """
        risks = []
        
        if not execution_result.success:
            risks.append(f"执行失败: {execution_result.error_message}")
        
        if execution_result.execution_time > 60:
            risks.append("执行时间过长，可能存在性能问题")
        
        failed_steps = [step for step in execution_result.step_results if not step.get("success")]
        if failed_steps:
            risks.append(f"部分步骤执行失败: {len(failed_steps)}个步骤")
        
        return risks
    
    def _extract_key_findings(self, execution_result: ExecutionResult) -> List[str]:
        """提取关键发现
        
        Args:
            execution_result: 执行结果
            
        Returns:
            List[str]: 关键发现列表
        """
        findings = []
        
        if execution_result.success:
            findings.append(f"预案执行成功，耗时{execution_result.execution_time:.2f}秒")
            
            for key, value in execution_result.final_outputs.items():
                if isinstance(value, (int, float)):
                    findings.append(f"{key}: {value}")
        else:
            findings.append(f"预案执行失败: {execution_result.error_message}")
        
        return findings
    
    def _calculate_performance_metrics(self, execution_result: ExecutionResult) -> Dict[str, Any]:
        """计算性能指标
        
        Args:
            execution_result: 执行结果
            
        Returns:
            Dict[str, Any]: 性能指标
        """
        return {
            "total_execution_time": execution_result.execution_time,
            "step_count": len(execution_result.step_results),
            "success_rate": self._analyze_data_reliability(execution_result),
            "efficiency_score": self._analyze_efficiency(execution_result),
            "average_step_time": (
                execution_result.execution_time / len(execution_result.step_results)
                if execution_result.step_results else 0
            )
        }
    
    def _extract_recommendations(self, recommendations_text: str) -> List[str]:
        """从文本中提取建议列表
        
        Args:
            recommendations_text: 建议文本
            
        Returns:
            List[str]: 建议列表
        """
        import re
        
        # 按行分割并清理
        lines = recommendations_text.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if line:
                # 移除序号和项目符号
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-•*]\s*', '', line)
                if line:
                    recommendations.append(line)
        
        return recommendations
    
    def _build_background_section(self, execution_result: ExecutionResult) -> str:
        """构建背景信息部分
        
        Args:
            execution_result: 执行结果
            
        Returns:
            str: 背景信息文本
        """
        return f"""场景描述: {execution_result.scenario}
预案标识: {execution_result.plan_id}
执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
执行状态: {'成功' if execution_result.success else '失败'}
执行耗时: {execution_result.execution_time:.2f}秒
处理步骤: {len(execution_result.step_results)}个步骤"""
    
    def _determine_confidence_level(self, analysis: Dict[str, Any]) -> str:
        """确定置信等级
        
        Args:
            analysis: 分析结果
            
        Returns:
            str: 置信等级
        """
        confidence_score = analysis.get("result_confidence", 0.0)
        
        if confidence_score >= 0.9:
            return "很高"
        elif confidence_score >= 0.7:
            return "高"
        elif confidence_score >= 0.5:
            return "中等"
        elif confidence_score >= 0.3:
            return "低"
        else:
            return "很低"
    
    def _load_report_templates(self) -> Dict[str, str]:
        """加载报告模板
        
        Returns:
            Dict[str, str]: 报告模板字典
        """
        # 这里可以从文件加载模板
        return {
            "standard": "标准决策报告模板",
            "emergency": "应急决策报告模板",
            "detailed": "详细分析报告模板"
        }
    
    async def export_report(
        self,
        report: DecisionReport,
        format_type: str = "markdown",
        output_path: Optional[Path] = None
    ) -> str:
        """导出报告
        
        Args:
            report: 决策报告
            format_type: 导出格式 (markdown, json, html)
            output_path: 输出路径
            
        Returns:
            str: 导出的文件路径或内容
        """
        if format_type == "markdown":
            content = self._export_to_markdown(report)
        elif format_type == "json":
            content = self._export_to_json(report)
        elif format_type == "html":
            content = self._export_to_html(report)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return str(output_path)
        
        return content
    
    def _export_to_markdown(self, report: DecisionReport) -> str:
        """导出为Markdown格式
        
        Args:
            report: 决策报告
            
        Returns:
            str: Markdown内容
        """
        import json
        
        md_lines = [
            f"# 电网调度决策报告",
            f"",
            f"**报告ID**: {report.report_id}",
            f"**执行ID**: {report.execution_id}",
            f"**预案标题**: {report.plan_title}",
            f"**生成时间**: {report.generated_at}",
            f"**置信等级**: {report.confidence_level}",
            f"",
            f"## 场景描述",
            f"{report.scenario}",
            f"",
            f"## 执行摘要",
            f"{report.summary}",
            f"",
            f"## 背景信息",
            f"```",
            f"{report.background}",
            f"```",
            f"",
            f"## 数据来源",
            f"```json",
            json.dumps(report.data_sources, ensure_ascii=False, indent=2),
            f"```",
            f"",
            f"## 计算过程",
            f"```json",
            json.dumps(report.calculations, ensure_ascii=False, indent=2),
            f"```",
            f"",
            f"## 建议措施",
        ]
        
        for i, rec in enumerate(report.recommendations, 1):
            md_lines.append(f"{i}. {rec}")
        
        if report.warnings:
            md_lines.extend([
                f"",
                f"## 警告信息",
            ])
            for warning in report.warnings:
                md_lines.append(f"⚠️ {warning}")
        
        return "\n".join(md_lines)
    
    def _export_to_json(self, report: DecisionReport) -> str:
        """导出为JSON格式
        
        Args:
            report: 决策报告
            
        Returns:
            str: JSON内容
        """
        import json
        return json.dumps(report.dict(), ensure_ascii=False, indent=2)
    
    def _export_to_html(self, report: DecisionReport) -> str:
        """导出为HTML格式
        
        Args:
            report: 决策报告
            
        Returns:
            str: HTML内容
        """
        import json
        
        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电网调度决策报告 - {report.report_id}</title>
    <style>
        body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #2c5282; }}
        .metadata {{ background-color: #f7fafc; padding: 15px; border-radius: 5px; }}
        .warning {{ color: #e53e3e; background-color: #fed7d7; padding: 10px; border-radius: 5px; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .recommendation {{ margin: 10px 0; padding: 10px; background-color: #e6fffa; border-left: 4px solid #38b2ac; }}
    </style>
</head>
<body>
    <h1>电网调度决策报告</h1>
    
    <div class="metadata">
        <p><strong>报告ID</strong>: {report.report_id}</p>
        <p><strong>执行ID</strong>: {report.execution_id}</p>
        <p><strong>预案标题</strong>: {report.plan_title}</p>
        <p><strong>生成时间</strong>: {report.generated_at}</p>
        <p><strong>置信等级</strong>: {report.confidence_level}</p>
    </div>
    
    <h2>场景描述</h2>
    <p>{report.scenario}</p>
    
    <h2>执行摘要</h2>
    <p>{report.summary}</p>
    
    <h2>背景信息</h2>
    <pre>{report.background}</pre>
    
    <h2>建议措施</h2>"""
        
        for i, rec in enumerate(report.recommendations, 1):
            html_template += f'<div class="recommendation">{i}. {rec}</div>'
        
        if report.warnings:
            html_template += '<h2>警告信息</h2>'
            for warning in report.warnings:
                html_template += f'<div class="warning">⚠️ {warning}</div>'
        
        html_template += """
</body>
</html>"""
        
        return html_template


# 导入json模块
import json