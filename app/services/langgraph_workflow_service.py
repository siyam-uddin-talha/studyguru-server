"""
LangGraph Workflow Service for Multi-Source Summarization
Intelligent orchestration for links + PDFs + web search with ThinkingConfig
"""

import asyncio
import json
import re
import hashlib
import requests
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
from langchain_community.utilities import GoogleSerperAPIWrapper

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
    original_message: str = ""  # Store the original user message
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
            print(
                f"🔍 [ANALYZE INPUT] Message received: {message[:200]}... (length: {len(message)})"
            )
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
                original_message=message,  # Store original message for response generation
                thinking_steps=[],
            )

            # Add thinking step
            thinking_step = f"🔍 Analyzing input: {input_analysis.suggested_workflow} workflow detected"
            context.thinking_steps.append(thinking_step)

            return {
                "message": message,  # Preserve message in state
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

            message = state.get("message", "")  # Preserve message

            if not context.input_analysis.has_pdfs:
                return {
                    "message": message,  # Preserve message in state
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
                "message": message,  # Preserve message in state
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
        """Search web and scrape URLs if needed"""
        try:
            context = state.get("context")
            message = state.get("message", "")

            print(f"🌐 [WEB SEARCH NODE] Called with message: {message[:100]}...")
            print(
                f"🌐 [WEB SEARCH NODE] requires_web_search: {context.input_analysis.requires_web_search}"
            )

            if not context.input_analysis.requires_web_search:
                print(f"⚠️ [WEB SEARCH NODE] Skipping - web search not required")
                return {
                    "context": context,
                    "state": WorkflowState.SEARCHING_WEB,
                    "thinking_steps": context.thinking_steps,
                }

            # Extract URLs from message - improved regex to handle all URL formats
            # Match http:// or https:// followed by any non-whitespace characters
            url_pattern = r"https?://[^\s]+"
            urls_in_message = re.findall(url_pattern, message)

            # Also try a more lenient pattern if first one fails
            if not urls_in_message:
                url_pattern_alt = r'https?://[^\s\n\r<>"]+'
                urls_in_message = re.findall(url_pattern_alt, message)

            print(f"🔗 [WEB SEARCH NODE] Message length: {len(message)} chars")
            print(f"🔗 [WEB SEARCH NODE] Message preview: {message[:200]}")
            print(
                f"🔗 [WEB SEARCH NODE] Extracted {len(urls_in_message)} URL(s): {urls_in_message}"
            )

            # Add thinking step
            if urls_in_message:
                thinking_step = f"🔗 Found {len(urls_in_message)} URL(s) to analyze..."
                context.thinking_steps.append(thinking_step)
            else:
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

            # Clean message to get search query (remove URLs)
            clean_message = re.sub(url_pattern, "", message).strip()
            if clean_message and len(clean_message) > 5:  # Relaxed length check
                search_queries.append(clean_message)

            # Add specific query for the URL content if no other queries
            if urls_in_message and not search_queries:
                search_queries.append("summary of " + urls_in_message[0])

            # Perform web search with URLs
            web_search_result = await self._perform_web_search(
                queries=search_queries, context=context, urls=urls_in_message
            )
            context.web_search_results = web_search_result

            # Add thinking step
            if web_search_result.success:
                urls_scraped = web_search_result.metadata.get("urls_scraped", 0)
                searches_done = web_search_result.metadata.get("searches_performed", 0)
                thinking_step = f"✅ Retrieved content: {urls_scraped} URL(s) scraped, {searches_done} search(es) performed"
            else:
                thinking_step = f"⚠️ Web search had issues: {web_search_result.error}"
            context.thinking_steps.append(thinking_step)

            return {
                "message": message,  # Preserve message in state
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
            # Check for existing errors in state
            if "error" in state:
                print(
                    f"⚠️ [GENERATE SUMMARY NODE] Found existing error in state: {state.get('error')}"
                )

            context = state.get("context")

            # Add thinking step
            thinking_step = "📝 Generating comprehensive summary..."
            context.thinking_steps.append(thinking_step)

            # Generate final summary
            summary_result = await self._generate_final_summary(context)
            context.final_summary = summary_result

            # Debug logging
            print(f"🔍 [GENERATE SUMMARY NODE] summary_result received:")
            print(f"   Type: {type(summary_result)}")
            print(f"   Success: {summary_result.success if summary_result else 'None'}")
            print(
                f"   Content length: {len(summary_result.content) if summary_result and summary_result.content else 0}"
            )
            if summary_result and summary_result.content:
                print(f"   Content preview: {summary_result.content[:200]}")

            # Calculate totals - properly flatten the list of results
            all_results = []

            # Add PDF results (already a list)
            if context.pdf_results:
                all_results.extend(context.pdf_results)

            # Add web search result (single object)
            if context.web_search_results:
                all_results.append(context.web_search_results)

            # Add integration result (single object)
            if context.integration_result:
                all_results.append(context.integration_result)

            # Add final summary (single object)
            if context.final_summary:
                all_results.append(context.final_summary)

            # Calculate total tokens from all ProcessingResult objects
            context.total_tokens = sum(
                result.tokens_used
                for result in all_results
                if result and hasattr(result, "tokens_used")
            )

            # Add thinking step
            thinking_step = f"✅ Summary generated successfully ({context.total_tokens} tokens used)"
            context.thinking_steps.append(thinking_step)

            # Debug logging
            final_content = (
                summary_result.content
                if summary_result and summary_result.content
                else ""
            )
            print(
                f"🔍 [GENERATE SUMMARY NODE] summary_result type: {type(summary_result)}"
            )
            print(
                f"🔍 [GENERATE SUMMARY NODE] summary_result.success: {summary_result.success if summary_result else 'N/A'}"
            )
            print(
                f"🔍 [GENERATE SUMMARY NODE] final_content length: {len(final_content)}"
            )
            print(
                f"🔍 [GENERATE SUMMARY NODE] final_content preview: {final_content[:200]}"
            )

            # Return successful result - don't include error key to avoid state merge issues
            message = state.get("message", "")  # Preserve message
            return_dict = {
                "message": message,  # Preserve message in state
                "context": context,
                "state": WorkflowState.COMPLETED,
                "thinking_steps": context.thinking_steps,
                "final_result": final_content,
                "total_tokens": context.total_tokens,
            }
            print(
                f"🔍 [GENERATE SUMMARY NODE] About to return with keys: {list(return_dict.keys())}"
            )
            print(
                f"🔍 [GENERATE SUMMARY NODE] final_result in return: {len(return_dict.get('final_result', ''))} chars"
            )
            print(
                f"🔍 [GENERATE SUMMARY NODE] final_result preview: {return_dict.get('final_result', '')[:200]}"
            )
            return return_dict

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

    async def _scrape_url_with_serper(self, url: str) -> Optional[str]:
        """Scrape a URL using Serper's scrape API"""
        try:
            if not settings.SERPER_API_KEY:
                print("⚠️ SERPER_API_KEY not set, cannot scrape URL")
                return None

            print(f"🔗 [SCRAPE] Attempting to scrape: {url}")
            scrape_url = "https://scrape.serper.dev"
            payload = json.dumps({"url": url})
            headers = {
                "X-API-KEY": settings.SERPER_API_KEY,
                "Content-Type": "application/json",
            }

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    scrape_url, headers=headers, data=payload, timeout=30
                ),
            )

            print(f"🔗 [SCRAPE] Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                # Extract text content from the response
                text_content = result.get("text", "")
                title = result.get("title", "")

                print(f"🔗 [SCRAPE] Title: {title[:100] if title else 'None'}")
                print(f"🔗 [SCRAPE] Content length: {len(text_content)} chars")

                if text_content:
                    return f"Title: {title}\n\nContent:\n{text_content[:10000]}"  # Limit content length
                else:
                    print(
                        f"⚠️ [SCRAPE] No text content in response. Keys: {list(result.keys())}"
                    )

            print(
                f"⚠️ Serper scrape failed with status {response.status_code}: {response.text[:200]}"
            )
            return None

        except Exception as e:
            print(f"⚠️ Error scraping URL {url}: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def _search_with_serper(self, query: str) -> Optional[str]:
        """Search using Serper's search API"""
        try:
            if not settings.SERPER_API_KEY:
                print("⚠️ SERPER_API_KEY not set, cannot search")
                return None

            search_url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query, "num": 3})
            headers = {
                "X-API-KEY": settings.SERPER_API_KEY,
                "Content-Type": "application/json",
            }

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    search_url, headers=headers, data=payload, timeout=30
                ),
            )

            if response.status_code == 200:
                result = response.json()

                # Debug logging
                # print(f"🔍 [SEARCH] Raw result keys: {list(result.keys())}")

                # Format organic results
                formatted_results = []

                # Add answer box if available
                if "answerBox" in result:
                    answer_box = result["answerBox"]
                    if "answer" in answer_box:
                        formatted_results.append(f"Answer: {answer_box['answer']}")
                    elif "snippet" in answer_box:
                        formatted_results.append(f"Answer: {answer_box['snippet']}")

                # Add knowledge graph if available
                if "knowledgeGraph" in result:
                    kg = result["knowledgeGraph"]
                    if "description" in kg:
                        formatted_results.append(f"Overview: {kg['description']}")

                # Add organic results
                organic = result.get("organic", [])
                if not organic:
                    print(f"⚠️ [SEARCH] No organic results found for query: {query}")

                for item in organic[:5]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")

                    # Include result if it has title and link, even without snippet
                    if title and link:
                        formatted_results.append(
                            f"• {title}\n  Snippet: {snippet or 'No snippet available'}\n  Source: {link}"
                        )

                if formatted_results:
                    return "\n\n".join(formatted_results)
                else:
                    print(
                        f"⚠️ [SEARCH] No formatted results extracted for query: {query}"
                    )

            print(f"⚠️ Serper search failed with status {response.status_code}")
            if response.status_code != 200:
                print(f"⚠️ Response text: {response.text[:200]}")
            return None

        except Exception as e:
            print(f"⚠️ Error searching with Serper: {e}")
            return None

    async def _perform_web_search(
        self, queries: List[str], context: WorkflowContext, urls: List[str] = None
    ) -> ProcessingResult:
        """Perform web search and URL scraping using Serper API"""
        start_time = datetime.now()

        print(f"🔍 [PERFORM WEB SEARCH] Called with:")
        print(f"   URLs: {urls}")
        print(f"   Queries: {queries}")

        try:
            all_results = []

            # First, scrape any URLs provided
            if urls:
                print(f"🔗 [PERFORM WEB SEARCH] Scraping {len(urls)} URLs...")
                for url in urls[:3]:  # Limit to 3 URLs
                    scraped_content = await self._scrape_url_with_serper(url)
                    if scraped_content:
                        # Format content without URL in header to avoid confusion
                        # The content is already retrieved, so we don't need to mention the URL
                        all_results.append(f"📄 Retrieved Content:\n{scraped_content}")
                        print(f"✅ Successfully scraped: {url}")
                    else:
                        print(
                            f"⚠️ Failed to scrape: {url}. Attempting to search for it instead..."
                        )
                        # Fallback: Search for the URL to get snippet/title
                        search_result = await self._search_with_serper(url)
                        if search_result:
                            all_results.append(
                                f"🔍 Retrieved Information:\n{search_result}"
                            )
                            print(f"✅ Found search results for URL: {url}")
                        else:
                            print(f"❌ Failed to search for URL: {url}")

            # Then, perform searches for queries (if no URLs or need more context)
            if queries and (not urls or not all_results):
                print(f"🔍 Searching for {len(queries)} queries...")
                for query in queries[:3]:  # Limit to 3 queries
                    if query.strip() and not query.startswith("http"):
                        search_result = await self._search_with_serper(query)
                        if search_result:
                            all_results.append(
                                f"🔍 Additional Information:\n{search_result}"
                            )

            if not all_results:
                print(f"❌ [PERFORM WEB SEARCH] No results retrieved")
                return ProcessingResult(
                    success=False,
                    error="No content could be retrieved from URLs or search",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            combined_results = "\n\n---\n\n".join(all_results)
            processing_time = (datetime.now() - start_time).total_seconds()

            print(f"✅ [PERFORM WEB SEARCH] Completed:")
            print(f"   Success: True")
            print(f"   Content length: {len(combined_results)} chars")
            print(f"   Results count: {len(all_results)}")
            print(f"   Processing time: {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                content=combined_results,
                metadata={
                    "sources": (urls or [])
                    + [q for q in (queries or []) if not q.startswith("http")],
                    "search_type": "serper",
                    "urls_scraped": len([r for r in all_results if r.startswith("📄")]),
                    "searches_performed": len(
                        [r for r in all_results if r.startswith("🔍")]
                    ),
                },
                tokens_used=len(combined_results.split()),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"❌ Web search/scrape error: {e}")
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
        """Generate final comprehensive summary based on user query and gathered content"""
        start_time = datetime.now()

        try:
            # Get appropriate model
            model = StudyGuruConfig.MODELS.get_chat_model()

            # Prepare summary content from all sources
            content_parts = []

            # Add integration result if available
            if context.integration_result and context.integration_result.success:
                content_parts.append(
                    f"Integrated Analysis:\n{context.integration_result.content}"
                )

            # Add PDF results
            if context.pdf_results:
                for pdf_result in context.pdf_results:
                    if pdf_result.success and pdf_result.content:
                        content_parts.append(f"Document Content:\n{pdf_result.content}")

            # Add web search/scrape results (already retrieved - no URL needed)
            if (
                context.web_search_results
                and context.web_search_results.success
                and context.web_search_results.content
            ):
                # Remove any URL references from the content header
                content_text = context.web_search_results.content
                # Clean up any "Content from URL:" headers that might be in the content
                content_text = re.sub(
                    r"📄 Content from https?://[^\n]+\n",
                    "📄 Retrieved Content:\n",
                    content_text,
                )
                content_parts.append(f"Retrieved Content:\n{content_text}")

            summary_content = "\n\n---\n\n".join(content_parts)

            # Debug logging
            print(f"🔍 [SUMMARY] Content parts: {len(content_parts)}")
            print(f"🔍 [SUMMARY] Total content length: {len(summary_content)} chars")

            # If no content was gathered, provide a helpful message
            if not summary_content or not summary_content.strip():
                # Still try to respond based on the original message
                original_message = (
                    context.original_message
                    if hasattr(context, "original_message")
                    else ""
                )

                # Check if there are URLs in the message that couldn't be scraped
                if context.input_analysis.has_links:
                    error_response = (
                        "I apologize, but I was unable to retrieve content from the provided URL(s). "
                        "This could be due to:\n"
                        "1. The website blocking automated access\n"
                        "2. The page requiring authentication\n"
                        "3. Network connectivity issues\n\n"
                        "Please try:\n"
                        "- Copying and pasting the relevant text directly\n"
                        "- Providing a different link to the same content\n"
                        "- Describing the topic you'd like to learn about"
                    )
                else:
                    error_response = "I couldn't find any content to analyze. Please provide more details about what you'd like to learn."

                return ProcessingResult(
                    success=True,  # Return success=True so the response is shown
                    content=error_response,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Get the user's original question/request
            user_query = ""
            if hasattr(context, "original_message") and context.original_message:
                # Extract the question part (remove URLs)
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                user_query = re.sub(url_pattern, "", context.original_message).strip()

            # Create summary prompt that addresses the user's question
            # IMPORTANT: The content below is already retrieved/scraped - do NOT mention inability to access URLs
            if user_query:
                summary_prompt = f"""You have been provided with content that has already been retrieved for you. Use this content to answer the user's question.

User's Question: {user_query}

Retrieved Content (already available - use this directly):
{summary_content}

Please provide a comprehensive, educational response that:
1. Directly answers the user's question using the provided content
2. Is well-structured and easy to understand
3. Includes key insights and findings from the content
4. Cites relevant sources when appropriate
5. Is informative and helpful for learning

IMPORTANT: The content above is already retrieved and available to you. Do NOT mention that you cannot access URLs or websites - you already have the content."""
            else:
                summary_prompt = f"""You have been provided with content that has already been retrieved for you. Please summarize and explain it in an educational way.

Retrieved Content (already available - use this directly):
{summary_content}

The response should be:
1. Well-structured and easy to understand
2. Include key insights and findings from the provided content
3. Be educational and informative
4. Cite sources where appropriate

IMPORTANT: The content above is already retrieved and available to you. Do NOT mention that you cannot access URLs or websites - you already have the content."""

            # Generate summary
            print(f"🚀 [SUMMARY] Generating response...")
            response = await model.ainvoke([HumanMessage(content=summary_prompt)])

            processing_time = (datetime.now() - start_time).total_seconds()
            print(
                f"✅ [SUMMARY] Response generated: {len(response.content)} chars in {processing_time:.2f}s"
            )

            return ProcessingResult(
                success=True,
                content=response.content,
                metadata={"summary_type": "comprehensive"},
                tokens_used=len(response.content.split()),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"❌ [SUMMARY] Error: {e}")
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

            # Debug logging
            print(f"🔍 [EXECUTE WORKFLOW] Result keys: {list(result.keys())}")
            if "error" in result:
                print(f"❌ [EXECUTE WORKFLOW] ERROR DETECTED: {result.get('error')}")
                print(f"❌ [EXECUTE WORKFLOW] Error state: {result.get('state')}")
            print(
                f"🔍 [EXECUTE WORKFLOW] final_result type: {type(result.get('final_result'))}"
            )
            print(
                f"🔍 [EXECUTE WORKFLOW] final_result length: {len(str(result.get('final_result', '')))}"
            )
            print(
                f"🔍 [EXECUTE WORKFLOW] final_result preview: {str(result.get('final_result', ''))[:200]}"
            )
            print(f"🔍 [EXECUTE WORKFLOW] state value: {result.get('state')}")

            # Extract final_result - check multiple possible locations
            final_result = result.get("final_result", "")

            # Fallback: try to get from context if final_result is empty
            if not final_result:
                context = result.get("context")
                if context and hasattr(context, "final_summary"):
                    if context.final_summary and context.final_summary.content:
                        final_result = context.final_summary.content
                        print(
                            f"✅ [EXECUTE WORKFLOW] Extracted result from context.final_summary: {len(final_result)} chars"
                        )

            return {
                "success": True if final_result else False,
                "result": final_result,
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
