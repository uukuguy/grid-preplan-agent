"""RAG查询Agent"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.models import RAGResult
from ..utils.logger import logger


class RAGAgent:
    """RAG查询Agent"""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-small",
        knowledge_base_path: Optional[Path] = None
    ):
        """初始化RAG Agent
        
        Args:
            model: 使用的LLM模型
            embedding_model: 嵌入模型
            knowledge_base_path: 知识库路径
        """
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.knowledge_base_path = knowledge_base_path or Path("knowledge_base")
        
        # 向量数据库
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []
        
        # 知识库内容
        self.grid_knowledge = self._load_default_knowledge()
        
        # 初始化向量数据库
        self._initialize_vectorstore()
    
    def _load_default_knowledge(self) -> Dict[str, str]:
        """加载默认知识库内容
        
        Returns:
            Dict[str, str]: 知识库内容
        """
        return {
            "送端受端判定": """
            根据电网拓扑结构和潮流方向判定设备位于送端还是受端：
            
            1. 送端特征：
               - 位于发电侧或电力输出端
               - 潮流方向为输出
               - 承担电力输送任务
               
            2. 受端特征：
               - 位于负荷侧或电力输入端
               - 潮流方向为输入
               - 承担电力接收任务
               
            3. 判定方法：
               - 查看设备在电网中的位置
               - 分析正常运行时的潮流方向
               - 考虑设备的功能定位
            """,
            
            "直流输电限额": """
            直流输电限额的确定原则：
            
            1. 系统限额 = min(送端限额, 受端限额)
            2. 设备限额 = 换流器额定容量 × 运行系数
            3. 实际限额 = min(系统限额, 设备限额)
            
            影响因素：
            - 送端电网承受能力
            - 受端电网接纳能力
            - 换流器设备容量
            - 系统运行方式
            - 安全稳定约束
            """,
            
            "故障处理原则": """
            电网设备故障处理基本原则：
            
            1. 安全第一原则
            2. 快速响应原则
            3. 影响最小化原则
            4. 系统稳定原则
            
            处理流程：
            1. 故障识别和定位
            2. 影响范围评估
            3. 应急措施制定
            4. 操作执行
            5. 效果验证
            6. 持续监控
            """,
            
            "电网调度规程": """
            《电网调度管理条例》主要规定：
            
            1. 调度权限和责任
            2. 调度操作规范
            3. 安全技术要求
            4. 应急处置程序
            
            调度原则：
            - 统一调度、分级管理
            - 安全第一、兼顾效益
            - 实时平衡、动态优化
            """,
            
            "换流器参数": """
            换流器关键参数：
            
            1. 额定容量：换流器额定功率
            2. 额定电压：换流器额定直流电压
            3. 额定电流：换流器额定直流电流
            4. 过载能力：短时过载倍数
            5. 调制比：PWM调制深度
            
            运行特性：
            - P = U × I （功率 = 电压 × 电流）
            - 功率因数调节能力
            - 无功调节范围
            """
        }
    
    def _initialize_vectorstore(self):
        """初始化向量数据库"""
        try:
            # 创建文档
            documents = []
            for title, content in self.grid_knowledge.items():
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "source": "default_knowledge",
                        "type": "regulation"
                    }
                )
                documents.append(doc)
            
            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            self.documents = split_docs
            
            # 创建向量数据库
            if split_docs:
                self.vectorstore = FAISS.from_documents(
                    split_docs,
                    self.embeddings
                )
                logger.info(f"向量数据库初始化完成，包含{len(split_docs)}个文档片段")
            else:
                logger.warning("未找到文档，向量数据库为空")
                
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
    
    async def query(
        self,
        question: str,
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> RAGResult:
        """执行RAG查询
        
        Args:
            question: 查询问题
            top_k: 返回文档数量
            similarity_threshold: 相似度阈值
            
        Returns:
            RAGResult: 查询结果
        """
        logger.info(f"执行RAG查询: {question}")
        
        try:
            if not self.vectorstore:
                return RAGResult(
                    query=question,
                    results=["知识库未初始化"],
                    sources=["system"],
                    confidence=0.0
                )
            
            # 检索相关文档
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                question, k=top_k
            )
            
            # 过滤低相似度文档
            relevant_docs = []
            sources = []
            
            for doc, score in docs_with_scores:
                # FAISS返回的是距离，需要转换为相似度
                similarity = 1 / (1 + score)
                if similarity >= similarity_threshold:
                    relevant_docs.append(doc)
                    sources.append(doc.metadata.get("title", "unknown"))
            
            if not relevant_docs:
                return RAGResult(
                    query=question,
                    results=["未找到相关信息"],
                    sources=["system"],
                    confidence=0.0
                )
            
            # 生成回答
            answer = await self._generate_answer(question, relevant_docs)
            
            # 计算置信度
            confidence = min(1.0, len(relevant_docs) / top_k)
            
            result = RAGResult(
                query=question,
                results=[answer],
                sources=sources,
                confidence=confidence
            )
            
            logger.info(f"RAG查询完成，置信度: {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"RAG查询失败: {str(e)}")
            return RAGResult(
                query=question,
                results=[f"查询失败: {str(e)}"],
                sources=["system"],
                confidence=0.0
            )
    
    async def _generate_answer(
        self,
        question: str,
        documents: List[Document]
    ) -> str:
        """基于检索文档生成答案
        
        Args:
            question: 用户问题
            documents: 检索到的相关文档
            
        Returns:
            str: 生成的答案
        """
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", f"文档{i}")
            content = doc.page_content
            context_parts.append(f"【{title}】\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # 构建提示
        system_prompt = """你是一个专业的电网调度技术专家。请根据提供的知识库内容回答用户问题。

要求：
1. 基于提供的知识内容回答
2. 回答要准确、专业
3. 如果知识库中没有相关信息，请明确说明
4. 使用专业的电力系统术语
5. 保持简洁明了"""
        
        user_prompt = f"""知识库内容：
{context}

用户问题：{question}

请基于上述知识库内容回答问题。"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 调用LLM生成答案
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """添加文档到知识库
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            
        Returns:
            bool: 是否添加成功
        """
        try:
            doc = Document(
                page_content=content,
                metadata=metadata or {}
            )
            
            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            split_docs = text_splitter.split_documents([doc])
            
            # 添加到向量数据库
            if self.vectorstore and split_docs:
                self.vectorstore.add_documents(split_docs)
                self.documents.extend(split_docs)
                logger.info(f"成功添加{len(split_docs)}个文档片段")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    def load_knowledge_from_file(self, file_path: Path) -> bool:
        """从文件加载知识
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                    # 假设JSON格式为 {"title": "content", ...}
                    for title, content in data.items():
                        self.add_document(content, {"title": title, "source": str(file_path)})
                else:
                    # 文本文件
                    content = f.read()
                    self.add_document(content, {"title": file_path.stem, "source": str(file_path)})
            
            logger.info(f"成功从文件加载知识: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"从文件加载知识失败: {str(e)}")
            return False
    
    def save_vectorstore(self, save_path: Path) -> bool:
        """保存向量数据库
        
        Args:
            save_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            if self.vectorstore:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(save_path))
                logger.info(f"向量数据库已保存到: {save_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"保存向量数据库失败: {str(e)}")
            return False
    
    def load_vectorstore(self, load_path: Path) -> bool:
        """加载向量数据库
        
        Args:
            load_path: 加载路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if load_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(load_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"向量数据库已从{load_path}加载")
                return True
            return False
            
        except Exception as e:
            logger.error(f"加载向量数据库失败: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_documents": len(self.documents),
            "vectorstore_initialized": self.vectorstore is not None,
            "default_knowledge_items": len(self.grid_knowledge),
            "embedding_model": "text-embedding-3-small"
        }
    
    async def batch_query(
        self,
        questions: List[str],
        top_k: int = 3
    ) -> List[RAGResult]:
        """批量查询
        
        Args:
            questions: 问题列表
            top_k: 每个问题返回的文档数量
            
        Returns:
            List[RAGResult]: 查询结果列表
        """
        results = []
        
        for question in questions:
            result = await self.query(question, top_k)
            results.append(result)
        
        return results


# 便捷函数
def create_rag_agent(knowledge_base_path: Optional[Path] = None) -> RAGAgent:
    """创建RAG Agent实例
    
    Args:
        knowledge_base_path: 知识库路径
        
    Returns:
        RAGAgent: RAG Agent实例
    """
    return RAGAgent(knowledge_base_path=knowledge_base_path)