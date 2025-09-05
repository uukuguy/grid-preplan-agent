"""预案复杂度分析器"""

from typing import Dict, Any, List, Tuple
from enum import Enum

from ..core.models import PlanJSON, PlanStep, StepType
from ..utils.logger import logger


class ComplexityLevel(str, Enum):
    """复杂度级别"""
    LINEAR = "linear"          # 线性简单预案
    BRANCH = "branch"          # 分支决策预案
    MULTI_AGENT = "multi_agent"  # 多Agent协作预案


class ComplexityAnalyzer:
    """预案复杂度分析器"""
    
    def __init__(self):
        self.analysis_rules = [
            self._check_linear_pattern,
            self._check_branch_pattern,
            self._check_multi_agent_pattern
        ]
    
    def analyze(self, plan: PlanJSON) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        """分析预案复杂度
        
        Args:
            plan: 预案JSON对象
            
        Returns:
            Tuple[ComplexityLevel, Dict[str, Any]]: 复杂度级别和分析详情
        """
        logger.info(f"分析预案复杂度: {plan.plan_id}")
        
        analysis_result = {
            "plan_id": plan.plan_id,
            "step_count": len(plan.steps),
            "step_types": self._count_step_types(plan.steps),
            "has_dependencies": self._has_step_dependencies(plan.steps),
            "has_conditions": self._has_conditional_logic(plan.steps),
            "variable_complexity": self._analyze_variable_complexity(plan.variables),
            "execution_pattern": "sequential"  # 默认为顺序执行
        }
        
        # 应用分析规则
        for rule in self.analysis_rules:
            complexity, details = rule(plan, analysis_result)
            if complexity:
                analysis_result.update(details)
                logger.info(f"预案 {plan.plan_id} 复杂度级别: {complexity.value}")
                return complexity, analysis_result
        
        # 默认为线性复杂度
        logger.info(f"预案 {plan.plan_id} 复杂度级别: {ComplexityLevel.LINEAR.value}")
        return ComplexityLevel.LINEAR, analysis_result
    
    def _count_step_types(self, steps: List[PlanStep]) -> Dict[str, int]:
        """统计步骤类型"""
        counts = {"rag": 0, "tool": 0, "compute": 0}
        for step in steps:
            counts[step.type.value] += 1
        return counts
    
    def _has_step_dependencies(self, steps: List[PlanStep]) -> bool:
        """检查是否存在步骤依赖"""
        # 简化检查：如果后面的步骤输入依赖前面步骤的输出
        all_outputs = set()
        
        for step in steps:
            # 检查当前步骤的输入是否依赖之前的输出
            for input_key, input_value in step.inputs.items():
                if isinstance(input_value, str) and input_value.startswith("{") and input_value.endswith("}"):
                    var_name = input_value[1:-1]
                    if var_name in all_outputs:
                        return True
            
            # 添加当前步骤的输出
            all_outputs.update(step.outputs)
        
        return False
    
    def _has_conditional_logic(self, steps: List[PlanStep]) -> bool:
        """检查是否存在条件逻辑"""
        # 检查步骤描述中是否包含条件关键词
        condition_keywords = ["如果", "当", "则", "否则", "满足", "条件", "判断", "选择"]
        
        for step in steps:
            step_text = step.description.lower()
            if any(keyword in step_text for keyword in condition_keywords):
                return True
            
            # 检查公式中的条件逻辑
            if step.formula and any(op in step.formula for op in ["if", "case", "when"]):
                return True
        
        return False
    
    def _analyze_variable_complexity(self, variables: List) -> Dict[str, Any]:
        """分析变量复杂度"""
        if not variables:
            return {"level": "simple", "formula_count": 0, "complex_formulas": []}
        
        formula_count = sum(1 for var in variables if hasattr(var, 'formula') and var.formula)
        
        complex_formulas = []
        for var in variables:
            if hasattr(var, 'formula') and var.formula:
                # 检查公式复杂度
                formula = var.formula
                if any(op in formula for op in ["max", "min", "sum", "avg", "sqrt", "pow"]):
                    complex_formulas.append(var.symbol if hasattr(var, 'symbol') else str(var))
        
        level = "complex" if len(complex_formulas) > 2 else "moderate" if formula_count > 0 else "simple"
        
        return {
            "level": level,
            "formula_count": formula_count,
            "complex_formulas": complex_formulas
        }
    
    def _check_linear_pattern(
        self, 
        plan: PlanJSON, 
        analysis: Dict[str, Any]
    ) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        """检查线性模式"""
        # 线性模式特征：
        # 1. 步骤数量适中（< 15）
        # 2. 无复杂条件逻辑
        # 3. 变量复杂度简单或适中
        # 4. 顺序执行
        
        if (analysis["step_count"] <= 15 and
            not analysis["has_conditions"] and
            analysis["variable_complexity"]["level"] in ["simple", "moderate"]):
            
            return ComplexityLevel.LINEAR, {
                "complexity_reason": "线性顺序执行，无复杂条件逻辑",
                "recommended_executor": "LangGraph"
            }
        
        return None, {}
    
    def _check_branch_pattern(
        self, 
        plan: PlanJSON, 
        analysis: Dict[str, Any]
    ) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        """检查分支模式"""
        # 分支模式特征：
        # 1. 包含条件逻辑
        # 2. 变量复杂度适中到复杂
        # 3. 步骤数量中等
        
        if (analysis["has_conditions"] or 
            analysis["variable_complexity"]["level"] == "complex" or
            analysis["step_count"] > 15):
            
            return ComplexityLevel.BRANCH, {
                "complexity_reason": "包含条件分支或复杂计算逻辑",
                "recommended_executor": "AutoGen"
            }
        
        return None, {}
    
    def _check_multi_agent_pattern(
        self, 
        plan: PlanJSON, 
        analysis: Dict[str, Any]
    ) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        """检查多Agent模式"""
        # 多Agent模式特征：
        # 1. 步骤数量很多（> 20）
        # 2. 需要多种专业知识
        # 3. 复杂的依赖关系
        
        rag_steps = analysis["step_types"]["rag"]
        tool_steps = analysis["step_types"]["tool"]
        
        if (analysis["step_count"] > 20 or
            (rag_steps > 5 and tool_steps > 5) or
            self._requires_domain_expertise(plan)):
            
            return ComplexityLevel.MULTI_AGENT, {
                "complexity_reason": "需要多种专业知识和复杂协作",
                "recommended_executor": "AutoGen Multi-Agent"
            }
        
        return None, {}
    
    def _requires_domain_expertise(self, plan: PlanJSON) -> bool:
        """检查是否需要领域专业知识"""
        # 检查预案是否涉及多个专业领域
        domain_keywords = {
            "电气": ["电压", "电流", "功率", "阻抗", "电气"],
            "机械": ["机械", "振动", "转速", "扭矩"],
            "热力": ["温度", "压力", "热量", "冷却"],
            "控制": ["控制", "调节", "自动", "保护"],
            "通信": ["通信", "信号", "数据", "网络"]
        }
        
        found_domains = set()
        plan_text = f"{plan.title} {plan.description}"
        
        for step in plan.steps:
            plan_text += f" {step.description}"
        
        plan_text = plan_text.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in plan_text for keyword in keywords):
                found_domains.add(domain)
        
        # 如果涉及3个以上专业领域，认为需要多Agent协作
        return len(found_domains) >= 3
    
    def get_executor_recommendation(
        self, 
        complexity: ComplexityLevel
    ) -> Dict[str, Any]:
        """获取执行器推荐
        
        Args:
            complexity: 复杂度级别
            
        Returns:
            Dict[str, Any]: 执行器推荐信息
        """
        recommendations = {
            ComplexityLevel.LINEAR: {
                "primary": "LangGraph",
                "alternative": "Smolagents", 
                "reason": "线性流程适合StateGraph顺序执行",
                "estimated_time": "快速（< 30秒）"
            },
            ComplexityLevel.BRANCH: {
                "primary": "AutoGen",
                "alternative": "LangGraph",
                "reason": "分支逻辑需要智能决策能力",
                "estimated_time": "中等（30-120秒）"
            },
            ComplexityLevel.MULTI_AGENT: {
                "primary": "AutoGen Multi-Agent",
                "alternative": "AutoGen",
                "reason": "复杂协作需要多智能体配合",
                "estimated_time": "较长（120-300秒）"
            }
        }
        
        return recommendations.get(complexity, recommendations[ComplexityLevel.LINEAR])


def analyze_plan_complexity(plan: PlanJSON) -> Tuple[ComplexityLevel, Dict[str, Any]]:
    """分析预案复杂度（便捷函数）
    
    Args:
        plan: 预案JSON对象
        
    Returns:
        Tuple[ComplexityLevel, Dict[str, Any]]: 复杂度级别和分析详情
    """
    analyzer = ComplexityAnalyzer()
    return analyzer.analyze(plan)