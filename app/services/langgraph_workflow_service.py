"""
LangGraph Workflow Service for Multi-Source Summarization
Intelligent orchestration for links + PDFs + web search with ThinkingConfig
"""

import asyncio
import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph not available. Install with: pip install langgraph")
    StateGraph = None
    END = None
    ToolNode = None
    MemorySaver = None
    LANGGRAPH_AVAILABLE = False

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# Google GenAI imports for ThinkingConfig
try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ Google GenAI not available. Install with: pip install google-generativeai")

# Local imports
from app.config.langchain_config import StudyGuruConfig
from app.services.langchain_service import langchain_service
from app.services.context_service import context_service
from app.core.config import settings


class WorkflowState(Enum):
    """Workflow execution states"""

    ANALYZING_INPUT = "analyzing_input"
    PROCESSING_PDFS = "processing_pdfs"
    SEARCHING_WEB = "searching_web"
    INTEGRATING_SOURCES = "integrating_sources"
    GENERATING_SUMMARY = "generating_summary"
    COMPLETED = "completed"
    ERROR = "error"


class ComplexityLevel(Enum):
    """Task complexity levels for ThinkingConfig"""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"


@dataclass
class InputAnalysis:
    """Analysis of user input to determine processing strategy"""

    has_text: bool = False
    has_pdfs: bool = False
    has_links: bool = False
    has_images: bool = False
    complexity_level: ComplexityLevel = ComplexityLevel.SIMPLE
    requires_web_search: bool = False
    requires_analytical_thinking: bool = False
    estimated_processing_time: int = 0  # seconds
    suggested_workflow: str = ""


@dataclass
class ProcessingResult:
    """Result from each processing step"""

    success: bool = True
    content: str = ""
    metadata: Dict[str, Any] = None
    tokens_used: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class WorkflowContext:
    """Context for the entire workflow execution"""

    user_id: str
    interaction_id: str
    input_analysis: InputAnalysis
    pdf_results: List[ProcessingResult] = None
    web_search_results: ProcessingResult = None
    integration_result: ProcessingResult = None
    final_summary: ProcessingResult = None
    thinking_steps: List[str] = None
    total_tokens: int = 0
    total_processing_time: float = 0.0


class ThinkingConfigManager:
    """Manages ThinkingConfig for both GPT and Gemini models"""

    @staticmethod
    def should_use_thinking(complexity: ComplexityLevel, task_type: str) -> bool:
        """Determine if thinking should be enabled based on complexity and task type"""
        thinking_tasks = [
            "analytical_reasoning",
            "complex_analysis",
            "multi_source_integration",
            "comprehensive_summarization",
            "cross_reference_verification",
        ]

        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ANALYTICAL]:
            return True

        if task_type in thinking_tasks:
            return True

        return False

    @staticmethod
    def get_thinking_config(
        complexity: ComplexityLevel, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get appropriate thinking configuration"""
        if not ThinkingConfigManager.should_use_thinking(complexity, task_type):
            return None

        if StudyGuruConfig.MODELS._is_gemini_model() and GENAI_AVAILABLE:
            return {
                "thinking_config": {
                    "include_thoughts": True,
                    "thinking_budget": (
                        2048 if complexity == ComplexityLevel.ANALYTICAL else 1024
                    ),
                }
            }
        else:
            # For GPT models, we'll use reasoning effort parameter
            return {
                "reasoning_effort": (
                    "high" if complexity == ComplexityLevel.ANALYTICAL else "medium"
                )
            }


class InputAnalyzer:
    """Analyzes user input to determine processing strategy"""

    @staticmethod
    async def analyze_input(
        message: str, media_files: List[Dict[str, str]], user_id: str
    ) -> InputAnalysis:
        """Analyze input to determine workflow strategy"""

        # Basic content detection
        has_text = bool(message and message.strip())
        has_pdfs = any(
            file.get("type", "").lower() == "application/pdf" for file in media_files
        )
        has_links = bool(re.search(r"https?://[^\s]+", message or ""))
        has_images = any(
            file.get("type", "").startswith("image/") for file in media_files
        )

        # Determine complexity level
        complexity = InputAnalyzer._determine_complexity(
            has_text, has_pdfs, has_links, has_images, message
        )

        # Determine if web search is needed
        requires_web_search = InputAnalyzer._needs_web_search(message, has_links)

        # Determine if analytical thinking is needed
        requires_analytical_thinking = InputAnalyzer._needs_analytical_thinking(
            message, has_pdfs, has_links
        )

        # Estimate processing time
        estimated_time = InputAnalyzer._estimate_processing_time(
            has_pdfs, has_links, has_images, complexity
        )

        # Suggest workflow
        suggested_workflow = InputAnalyzer._suggest_workflow(
            has_text, has_pdfs, has_links, has_images, complexity
        )

        return InputAnalysis(
            has_text=has_text,
            has_pdfs=has_pdfs,
            has_links=has_links,
            has_images=has_images,
            complexity_level=complexity,
            requires_web_search=requires_web_search,
            requires_analytical_thinking=requires_analytical_thinking,
            estimated_processing_time=estimated_time,
            suggested_workflow=suggested_workflow,
        )

    @staticmethod
    def _determine_complexity(
        has_text: bool, has_pdfs: bool, has_links: bool, has_images: bool, message: str
    ) -> ComplexityLevel:
        """Determine task complexity level"""

        # Count complexity factors
        factors = 0
        if has_pdfs:
            factors += 2
        if has_links:
            factors += 1
        if has_images:
            factors += 1
        if has_text and len(message) > 500:
            factors += 1

        # Check for analytical keywords
        analytical_keywords = [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "summarize",
            "explain",
            "discuss",
            "research",
            "investigate",
            "examine",
        ]

        if any(keyword in message.lower() for keyword in analytical_keywords):
            factors += 2

        # Determine complexity level
        if factors >= 4:
            return ComplexityLevel.ANALYTICAL
        elif factors >= 2:
            return ComplexityLevel.COMPLEX
        elif factors >= 1:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    @staticmethod
    def _needs_web_search(message: str, has_links: bool) -> bool:
        """Determine if web search is needed"""
        if has_links:
            return True

        # Check for keywords that suggest current information is needed
        web_search_keywords = [
            "latest",
            "current",
            "recent",
            "new",
            "update",
            "news",
            "today",
            "2024",
            "recently",
            "now",
            "current status",
        ]

        return any(keyword in message.lower() for keyword in web_search_keywords)

    @staticmethod
    def _needs_analytical_thinking(
        message: str, has_pdfs: bool, has_links: bool
    ) -> bool:
        """Determine if analytical thinking is needed"""
        if has_pdfs and has_links:
            return True

        analytical_keywords = [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "research",
            "investigate",
            "examine",
            "assess",
            "critique",
            "review",
        ]

        return any(keyword in message.lower() for keyword in analytical_keywords)

    @staticmethod
    def _estimate_processing_time(
        has_pdfs: bool, has_links: bool, has_images: bool, complexity: ComplexityLevel
    ) -> int:
        """Estimate processing time in seconds"""
        base_time = 5

        if has_pdfs:
            base_time += 15  # PDF processing time
        if has_links:
            base_time += 10  # Web search time
        if has_images:
            base_time += 8  # Image processing time

        # Complexity multiplier
        complexity_multiplier = {
            ComplexityLevel.SIMPLE: 1.0,
            ComplexityLevel.MODERATE: 1.5,
            ComplexityLevel.COMPLEX: 2.0,
            ComplexityLevel.ANALYTICAL: 2.5,
        }

        return int(base_time * complexity_multiplier[complexity])

    @staticmethod
    def _suggest_workflow(
        has_text: bool,
        has_pdfs: bool,
        has_links: bool,
        has_images: bool,
        complexity: ComplexityLevel,
    ) -> str:
        """Suggest appropriate workflow"""
        if has_pdfs and has_links:
            return "hybrid_processing"
        elif has_pdfs:
            return "document_processing"
        elif has_links:
            return "web_search_processing"
        elif has_images:
            return "multimodal_processing"
        else:
            return "text_processing"


class LangGraphWorkflowService:
    """Main service for LangGraph-based multi-source summarization"""

    def __init__(self):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required but not installed. Install with: pip install langgraph"
            )

        self.memory = MemorySaver()
        self.workflow_graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Define the workflow graph
        workflow = StateGraph(Dict[str, Any])

        # Add nodes
        workflow.add_node("analyze_input", self._analyze_input_node)
        workflow.add_node("process_pdfs", self._process_pdfs_node)
        workflow.add_node("search_web", self._search_web_node)
        workflow.add_node("integrate_sources", self._integrate_sources_node)
        workflow.add_node("generate_summary", self._generate_summary_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Set entrypoint - this is required for LangGraph
        workflow.set_entry_point("analyze_input")

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "analyze_input",
            self._should_continue_after_analysis,
            {"continue": "process_pdfs", "error": "handle_error"},
        )

        # Add edges for normal flow
        workflow.add_edge("process_pdfs", "search_web")
        workflow.add_edge("search_web", "integrate_sources")
        workflow.add_edge("integrate_sources", "generate_summary")
        workflow.add_edge("generate_summary", END)

        # Add error handling edge
        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=self.memory)

    async def _analyze_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input to determine processing strategy"""
        try:
            message = state.get("message", "")
            media_files = state.get("media_files", [])
            user_id = state.get("user_id", "")

            # Analyze input
            input_analysis = await InputAnalyzer.analyze_input(
                message, media_files, user_id
            )

            # Create workflow context
            context = WorkflowContext(
                user_id=user_id,
                interaction_id=state.get("interaction_id", ""),
                input_analysis=input_analysis,
                thinking_steps=[],
            )

            # Add thinking step
            thinking_step = f"🔍 Analyzing input: {input_analysis.suggested_workflow} workflow detected"
            context.thinking_steps.append(thinking_step)

            return {
                "context": context,
                "state": WorkflowState.ANALYZING_INPUT,
                "thinking_steps": context.thinking_steps,
            }

        except Exception as e:
            return {
                "error": f"Input analysis failed: {str(e)}",
                "state": WorkflowState.ERROR,
            }

    async def _process_pdfs_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF documents if present"""
        try:
            context = state.get("context")
            media_files = state.get("media_files", [])

            if not context.input_analysis.has_pdfs:
                return {
                    "context": context,
                    "state": WorkflowState.PROCESSING_PDFS,
                    "thinking_steps": context.thinking_steps,
                }

            # Add thinking step
            thinking_step = f"📄 Processing {len(media_files)} PDF document(s)..."
            context.thinking_steps.append(thinking_step)

            # Process PDFs in parallel
            pdf_tasks = []
            for media_file in media_files:
                if media_file.get("type", "").lower() == "application/pdf":
                    task = self._process_single_pdf(media_file, context)
                    pdf_tasks.append(task)

            if pdf_tasks:
                pdf_results = await asyncio.gather(*pdf_tasks, return_exceptions=True)
                context.pdf_results = [
                    result
                    for result in pdf_results
                    if not isinstance(result, Exception)
                ]

                # Add thinking step
                thinking_step = (
                    f"✅ Processed {len(context.pdf_results)} PDF(s) successfully"
                )
                context.thinking_steps.append(thinking_step)

            return {
                "context": context,
                "state": WorkflowState.PROCESSING_PDFS,
                "thinking_steps": context.thinking_steps,
            }

        except Exception as e:
            return {
                "error": f"PDF processing failed: {str(e)}",
                "state": WorkflowState.ERROR,
            }

    async def _search_web_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search web if needed"""
        try:
            context = state.get("context")
            message = state.get("message", "")

            if not context.input_analysis.requires_web_search:
                return {
                    "context": context,
                    "state": WorkflowState.SEARCHING_WEB,
                    "thinking_steps": context.thinking_steps,
                }

            # Add thinking step
            thinking_step = "🌐 Searching web for current information..."
            context.thinking_steps.append(thinking_step)

            # Extract search queries from PDF content if available
            search_queries = []
            if context.pdf_results:
                for pdf_result in context.pdf_results:
                    if pdf_result.success:
                        # Extract key topics from PDF content
                        topics = await self._extract_topics_from_content(
                            pdf_result.content
                        )
                        search_queries.extend(topics)

            # Add original message as search query
            search_queries.append(message)

            # Perform web search
            web_search_result = await self._perform_web_search(search_queries, context)
            context.web_search_results = web_search_result

            # Add thinking step
            thinking_step = f"✅ Found {len(web_search_result.metadata.get('sources', []))} relevant sources"
            context.thinking_steps.append(thinking_step)

            return {
                "context": context,
                "state": WorkflowState.SEARCHING_WEB,
                "thinking_steps": context.thinking_steps,
            }

        except Exception as e:
            return {
                "error": f"Web search failed: {str(e)}",
                "state": WorkflowState.ERROR,
            }

    async def _integrate_sources_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate all sources intelligently"""
        try:
            context = state.get("context")

            # Add thinking step
            thinking_step = "🔄 Integrating information from all sources..."
            context.thinking_steps.append(thinking_step)

            # Prepare integration content
            integration_content = await self._prepare_integration_content(context)

            # Generate integrated analysis
            integration_result = await self._generate_integrated_analysis(
                integration_content, context
            )
            context.integration_result = integration_result

            # Add thinking step
            thinking_step = "✅ Successfully integrated all sources"
            context.thinking_steps.append(thinking_step)

            return {
                "context": context,
                "state": WorkflowState.INTEGRATING_SOURCES,
                "thinking_steps": context.thinking_steps,
            }

        except Exception as e:
            return {
                "error": f"Source integration failed: {str(e)}",
                "state": WorkflowState.ERROR,
            }

    async def _generate_summary_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive summary"""
        try:
            context = state.get("context")

            # Add thinking step
            thinking_step = "📝 Generating comprehensive summary..."
            context.thinking_steps.append(thinking_step)

            # Generate final summary
            summary_result = await self._generate_final_summary(context)
            context.final_summary = summary_result

            # Calculate totals
            context.total_tokens = sum(
                [
                    result.tokens_used
                    for result in [
                        context.pdf_results or [],
                        (
                            [context.web_search_results]
                            if context.web_search_results
                            else []
                        ),
                        (
                            [context.integration_result]
                            if context.integration_result
                            else []
                        ),
                        [context.final_summary] if context.final_summary else [],
                    ]
                ]
            )

            # Add thinking step
            thinking_step = f"✅ Summary generated successfully ({context.total_tokens} tokens used)"
            context.thinking_steps.append(thinking_step)

            return {
                "context": context,
                "state": WorkflowState.COMPLETED,
                "thinking_steps": context.thinking_steps,
                "final_result": summary_result.content,
                "total_tokens": context.total_tokens,
            }

        except Exception as e:
            return {
                "error": f"Summary generation failed: {str(e)}",
                "state": WorkflowState.ERROR,
            }

    async def _handle_error_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors gracefully"""
        error_message = state.get("error", "Unknown error occurred")

        return {
            "error": error_message,
            "state": WorkflowState.ERROR,
            "final_result": f"Error: {error_message}",
        }

    def _should_continue_after_analysis(self, state: Dict[str, Any]) -> str:
        """Determine if workflow should continue after analysis"""
        if "error" in state:
            return "error"
        return "continue"

    # Helper methods
    async def _process_single_pdf(
        self, media_file: Dict[str, str], context: WorkflowContext
    ) -> ProcessingResult:
        """Process a single PDF file"""
        start_time = datetime.now()

        try:
            # Use existing document integration service
            from app.services.document_integration_service import (
                document_integration_service,
            )

            document_analysis = (
                await document_integration_service.process_document_comprehensive(
                    media_id=media_file.get("id", ""),
                    interaction_id=context.interaction_id,
                    user_id=context.user_id,
                    file_url=media_file.get("url", ""),
                    max_tokens=2000,
                )
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Validate that we got meaningful content
            if not document_analysis.content or not document_analysis.content.strip():
                return ProcessingResult(
                    success=False,
                    error="PDF processing returned empty content",
                    processing_time=processing_time,
                )

            return ProcessingResult(
                success=True,
                content=document_analysis.content,
                metadata=document_analysis.metadata,
                tokens_used=document_analysis.metadata.get("tokens_used", 0),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                success=False, error=str(e), processing_time=processing_time
            )

    async def _extract_topics_from_content(self, content: str) -> List[str]:
        """Extract key topics from content for web search"""
        # Simple topic extraction - can be enhanced with NLP
        words = content.lower().split()
        # Filter out common words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        topics = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(topics))[:5]  # Return top 5 unique topics

    async def _perform_web_search(
        self, queries: List[str], context: WorkflowContext
    ) -> ProcessingResult:
        """Perform web search using Gemini's native search"""
        start_time = datetime.now()

        try:
            # Use web search model with thinking config
            thinking_config = ThinkingConfigManager.get_thinking_config(
                context.input_analysis.complexity_level, "web_search"
            )

            # Get web search model
            if StudyGuruConfig.MODELS._is_gemini_model():
                model = StudyGuruConfig.MODELS.get_web_search_model()
            else:
                # Fallback to regular chat model
                model = StudyGuruConfig.MODELS.get_chat_model(web_search=True)

            # Validate queries
            if not queries or not any(query.strip() for query in queries):
                return ProcessingResult(
                    success=False,
                    error="No valid search queries provided",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Create search prompt
            search_prompt = f"""
            Search for current information related to: {', '.join(queries)}
            
            Provide comprehensive, up-to-date information with source citations.
            Focus on educational content and recent developments.
            """

            # Validate prompt is not empty
            if not search_prompt.strip():
                return ProcessingResult(
                    success=False,
                    error="Empty search prompt generated",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Perform search
            response = await model.ainvoke([HumanMessage(content=search_prompt)])

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                success=True,
                content=response.content,
                metadata={"sources": queries, "search_type": "web"},
                tokens_used=len(response.content.split()),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                success=False, error=str(e), processing_time=processing_time
            )

    async def _prepare_integration_content(self, context: WorkflowContext) -> str:
        """Prepare content for integration"""
        content_parts = []

        # Add PDF content
        if context.pdf_results:
            for pdf_result in context.pdf_results:
                if pdf_result.success:
                    content_parts.append(f"PDF Content:\n{pdf_result.content}")

        # Add web search content
        if context.web_search_results and context.web_search_results.success:
            content_parts.append(
                f"Web Search Results:\n{context.web_search_results.content}"
            )

        return "\n\n".join(content_parts)

    async def _generate_integrated_analysis(
        self, content: str, context: WorkflowContext
    ) -> ProcessingResult:
        """Generate integrated analysis of all sources"""
        start_time = datetime.now()

        try:
            # Use thinking config if needed
            thinking_config = ThinkingConfigManager.get_thinking_config(
                context.input_analysis.complexity_level, "multi_source_integration"
            )

            # Get appropriate model
            model = StudyGuruConfig.MODELS.get_chat_model()

            # Create integration prompt
            integration_prompt = f"""
            Analyze and integrate the following information from multiple sources:
            
            {content}
            
            Provide a comprehensive analysis that:
            1. Identifies key themes and connections
            2. Highlights important insights
            3. Notes any contradictions or gaps
            4. Synthesizes information coherently
            """

            # Generate analysis
            response = await model.ainvoke([HumanMessage(content=integration_prompt)])

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                success=True,
                content=response.content,
                metadata={"integration_type": "multi_source"},
                tokens_used=len(response.content.split()),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                success=False, error=str(e), processing_time=processing_time
            )

    async def _generate_final_summary(
        self, context: WorkflowContext
    ) -> ProcessingResult:
        """Generate final comprehensive summary"""
        start_time = datetime.now()

        try:
            # Use thinking config for complex summaries
            thinking_config = ThinkingConfigManager.get_thinking_config(
                context.input_analysis.complexity_level, "comprehensive_summarization"
            )

            # Get appropriate model
            model = StudyGuruConfig.MODELS.get_chat_model()

            # Prepare summary content
            summary_content = ""
            if context.integration_result and context.integration_result.success:
                summary_content = context.integration_result.content
            else:
                # Fallback to individual sources
                content_parts = []
                if context.pdf_results:
                    for pdf_result in context.pdf_results:
                        if pdf_result.success and pdf_result.content:
                            content_parts.append(pdf_result.content)
                if (
                    context.web_search_results
                    and context.web_search_results.success
                    and context.web_search_results.content
                ):
                    content_parts.append(context.web_search_results.content)
                summary_content = "\n\n".join(content_parts)

            # Validate that we have content to summarize
            if not summary_content or not summary_content.strip():
                return ProcessingResult(
                    success=False,
                    error="No content available for summarization",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Create summary prompt
            summary_prompt = f"""
            Create a comprehensive summary of the following information:
            
            {summary_content}
            
            The summary should be:
            1. Well-structured and easy to understand
            2. Include key insights and findings
            3. Highlight important connections between sources
            4. Be educational and informative
            5. Include relevant citations where appropriate
            """

            # Generate summary
            response = await model.ainvoke([HumanMessage(content=summary_prompt)])

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                success=True,
                content=response.content,
                metadata={"summary_type": "comprehensive"},
                tokens_used=len(response.content.split()),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                success=False, error=str(e), processing_time=processing_time
            )

    async def execute_workflow(
        self,
        message: str,
        media_files: List[Dict[str, str]],
        user_id: str,
        interaction_id: str,
    ) -> Dict[str, Any]:
        """Execute the complete workflow"""

        # Initial state
        initial_state = {
            "message": message,
            "media_files": media_files,
            "user_id": user_id,
            "interaction_id": interaction_id,
            "thinking_steps": [],
        }

        try:
            # Execute workflow with checkpointer config
            # LangGraph requires thread_id or other checkpoint keys when using a checkpointer
            # Generate a stable thread_id from interaction_id or create one from user_id and message
            if interaction_id:
                thread_id = f"thread_{interaction_id}"
            else:
                # Create a hash-based thread_id if no interaction_id
                message_hash = hashlib.md5(f"{user_id}_{message}".encode()).hexdigest()[
                    :12
                ]
                thread_id = f"thread_{user_id}_{message_hash}"

            config = {"configurable": {"thread_id": thread_id}}
            result = await self.workflow_graph.ainvoke(initial_state, config=config)

            return {
                "success": True,
                "result": result.get("final_result", ""),
                "thinking_steps": result.get("thinking_steps", []),
                "total_tokens": result.get("total_tokens", 0),
                "workflow_state": result.get("state", WorkflowState.COMPLETED),
                "context": result.get("context"),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "thinking_steps": ["❌ Workflow execution failed"],
                "result": f"Error: {str(e)}",
            }


# Global instance
langgraph_workflow_service = LangGraphWorkflowService()
