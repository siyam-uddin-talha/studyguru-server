import os
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_milvus import Zilliz
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field
from langchain_core.callbacks import AsyncCallbackHandler
from pymilvus import connections, Collection, utility
from app.core.config import settings
from app.config.langchain_config import StudyGuruConfig
import asyncio
from concurrent.futures import ThreadPoolExecutor


class StudyGuruCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for StudyGuru operations"""

    def __init__(self):
        self.tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.final_response = ""

    async def on_llm_end(self, response, **kwargs):
        """Capture token usage from LLM responses"""
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.tokens_used = token_usage.get("total_tokens", 0)
            self.input_tokens = token_usage.get("prompt_tokens", 0)
            self.output_tokens = token_usage.get("completion_tokens", 0)


class DocumentAnalysisOutput(BaseModel):
    """Pydantic model for structured document analysis output"""

    type: str = Field(description="Type of content: 'mcq', 'written', or 'other'")
    language: str = Field(description="Detected language of the content")
    title: str = Field(description="Short descriptive title for the content")
    summary_title: str = Field(description="Summary of how you will help the user")
    result: Dict[str, Any] = Field(description="The actual analysis result")


class GuardrailOutput(BaseModel):
    """Pydantic model for guardrail check output"""

    is_violation: bool = Field(description="Whether the content violates policies")
    violation_type: Optional[str] = Field(description="Type of violation if any")
    reasoning: str = Field(description="Explanation of the decision")


class MindmapNode(BaseModel):
    """Pydantic model for a mindmap node"""

    id: str = Field(description="Unique identifier for this node")
    content: str = Field(description="Content/text of this node")
    level: int = Field(description="Depth level in the tree (0 = root)")
    parent_id: Optional[str] = Field(
        default=None, description="ID of parent node, None for root"
    )
    color: str = Field(default="#4A90E2", description="Hex color code for this node")
    children: List["MindmapNode"] = Field(
        default_factory=list, description="List of child nodes"
    )


class MindmapNodeFlat(BaseModel):
    """Pydantic model for a flat mindmap node (no recursion for LLM)"""

    id: str = Field(description="Unique identifier for this node")
    content: str = Field(description="Content/text of this node")
    level: int = Field(description="Depth level in the tree (0 = root)")
    parent_id: Optional[str] = Field(
        default=None, description="ID of parent node, None for root"
    )
    color: str = Field(default="#4A90E2", description="Hex color code for this node")


class MindmapFlatOutput(BaseModel):
    """Pydantic model for flat mindmap generation output"""

    topic: str = Field(description="Main topic of the mindmap")
    nodes: List[MindmapNodeFlat] = Field(description="List of all nodes in the mindmap")
    total_nodes: int = Field(description="Total number of nodes in the mindmap")


class MindmapOutput(BaseModel):
    """Pydantic model for mindmap generation output (hierarchical)"""

    topic: str = Field(description="Main topic of the mindmap")
    root_node: MindmapNode = Field(description="Root node of the mindmap tree")
    total_nodes: int = Field(description="Total number of nodes in the mindmap")


class LangChainService:
    """LangChain-based service for StudyGuru operations"""

    def __init__(self):
        # Use configuration-based models
        self.llm = StudyGuruConfig.MODELS.get_chat_model()
        self.vision_llm = StudyGuruConfig.MODELS.get_vision_model()
        self.guardrail_llm = StudyGuruConfig.MODELS.get_guardrail_model()
        self.embeddings = StudyGuruConfig.MODELS.get_embeddings_model()

        self.vector_store = None
        self._initialize_vector_store()

        # Output parsers
        self.document_parser = JsonOutputParser(pydantic_object=DocumentAnalysisOutput)
        self.guardrail_parser = JsonOutputParser(pydantic_object=GuardrailOutput)
        self.mindmap_parser = JsonOutputParser(pydantic_object=MindmapOutput)
        self.string_parser = StrOutputParser()

        # Pre-configured chains
        self.document_analysis_chain = (
            StudyGuruConfig.CHAINS.get_document_analysis_chain()
        )
        self.guardrail_chain = StudyGuruConfig.CHAINS.get_guardrail_chain()

    def _initialize_vector_store(self):
        """Initialize Milvus vector store"""
        try:
            if not settings.ZILLIZ_URI or not settings.ZILLIZ_TOKEN:
                return

            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN,
                secure=True,
            )

            # Create or get collection (single collection for both models)
            collection_config = StudyGuruConfig.VECTOR_STORE.get_collection_config()
            collection_name = collection_config["collection_name"]
            if not utility.has_collection(collection_name):
                # Create collection with proper schema
                from pymilvus import FieldSchema, CollectionSchema, DataType

                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        auto_id=True,
                        max_length=64,
                    ),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(
                        name="interaction_id",
                        dtype=DataType.VARCHAR,
                        max_length=64,
                        nullable=True,
                    ),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(
                        name="metadata",
                        dtype=DataType.VARCHAR,
                        max_length=4096,
                        nullable=True,
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=StudyGuruConfig.VECTOR_STORE.get_collection_config()[
                            "dimension"
                        ],
                    ),  # Dynamic dimension based on model
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description="StudyGuru embeddings collection",
                    enable_dynamic_field=False,
                )

                collection = Collection(
                    name=collection_name, schema=schema, consistency_level="Bounded"
                )

                # Create index
                collection.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": settings.ZILLIZ_INDEX_METRIC,
                        "params": {"nlist": 1024},
                    },
                )

            # Initialize vector store using Zilliz (single collection for both models)
            self.vector_store = Zilliz(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                connection_args={
                    "uri": settings.ZILLIZ_URI,
                    "token": settings.ZILLIZ_TOKEN,
                    "secure": True,
                },
            )

        except Exception as e:
            print(f"❌ Vector store initialization failed: {e}")
            print(f"   ZILLIZ_URI: {settings.ZILLIZ_URI}")
            print(f"   ZILLIZ_TOKEN: {'***' if settings.ZILLIZ_TOKEN else 'None'}")
            self.vector_store = None

    async def analyze_document(
        self,
        file_url: str,
        max_tokens: int = 1000,
        visualize_model: Optional[str] = None,  # Model for document analysis
        subscription_plan: Optional[str] = None,  # User's subscription plan
    ) -> Dict[str, Any]:
        """Analyze document using LangChain with vision capabilities"""
        import time

        analysis_start = time.time()
        print(f"🔍 LANGCHAIN ANALYZE START: {analysis_start:.3f} - {file_url}")

        try:
            # Create callback handler to track tokens
            callback_handler = StudyGuruCallbackHandler()

            # Get the system message content from the existing template
            system_message_content = StudyGuruConfig.PROMPTS.DOCUMENT_ANALYSIS.messages[
                0
            ].prompt.template

            # Create document analysis prompt with file URL directly (no template variables)
            analysis_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_message_content),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Please analyze this document/image:",
                            },
                            {"type": "image_url", "image_url": {"url": file_url}},
                        ]
                    ),
                ]
            )

            # Create chain with vision model and parser
            # Use dynamic model if provided
            if visualize_model:
                print(f"🔍 Using custom vision model for analysis: {visualize_model}")
                vision_model = StudyGuruConfig.MODELS.get_vision_model(
                    temperature=0.3,
                    max_tokens=max_tokens,
                    model_name=visualize_model,
                    subscription_plan=subscription_plan,
                )
                chain = analysis_prompt | vision_model | self.document_parser
            else:
                chain = analysis_prompt | self.vision_llm | self.document_parser

            # Run analysis (no variables needed since we constructed the prompt directly)
            print(f"🔍 LANGCHAIN VISION MODEL CALL START: {time.time():.3f}")
            result = await chain.ainvoke({}, config={"callbacks": [callback_handler]})
            print(f"🔍 LANGCHAIN VISION MODEL CALL COMPLETE: {time.time():.3f}")

            # Add token usage
            result["token"] = callback_handler.tokens_used

            analysis_end = time.time()
            print(
                f"🔍 LANGCHAIN ANALYZE COMPLETE: {analysis_end:.3f} (took {analysis_end - analysis_start:.3f}s)"
            )

            return result

        except Exception as e:
            analysis_end = time.time()
            print(
                f"🔍 LANGCHAIN ANALYZE ERROR: {analysis_end:.3f} (took {analysis_end - analysis_start:.3f}s) - {e}"
            )
            return {
                "type": "error",
                "language": "unknown",
                "title": "Analysis Failed",
                "summary_title": "Sorry, we couldn't analyze this content",
                "token": 0,
                "_result": {
                    "error": "Unable to analyze the uploaded content. Please make sure it's a clear educational document or image.",
                    "details": str(e),
                },
            }

    async def generate_mcq_questions(
        self,
        topic_or_content: str,
        max_tokens: int = 1200,
        assistant_model: Optional[str] = None,
        subscription_plan: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate MCQ questions for a given topic or content"""
        try:
            # Create callback handler to track tokens
            callback_handler = StudyGuruCallbackHandler()

            # Get MCQ generation chain (use dynamic model if provided)
            chain = StudyGuruConfig.CHAINS.get_mcq_generation_chain(
                model_name=assistant_model,
                subscription_plan=subscription_plan,
            )

            # Run MCQ generation
            result = await chain.ainvoke(
                {"topic_or_content": topic_or_content},
                config={"callbacks": [callback_handler]},
            )

            # Add token usage
            result["token"] = callback_handler.tokens_used

            return result

        except Exception as e:
            return {
                "type": "error",
                "language": "unknown",
                "title": "MCQ Generation Failed",
                "summary_title": "Sorry, we couldn't generate MCQs for this topic",
                "token": 0,
                "_result": {
                    "error": "Unable to generate MCQs. Please try again with a different topic.",
                    "details": str(e),
                },
            }

    async def check_guardrails(
        self,
        message: str,
        media_urls: Optional[List[str]] = None,
        assistant_model: Optional[str] = None,
        subscription_plan: Optional[str] = None,
    ) -> GuardrailOutput:
        """Check content against guardrails using LangChain - optimized for speed"""
        try:
            # Quick check for simple greetings - always allow these
            if message and not media_urls:
                from app.constants import SIMPLE_GREETINGS

                message_lower = message.lower().strip()
                if message_lower in SIMPLE_GREETINGS:
                    print(
                        f"🛡️ GUARDRAIL: Simple greeting detected - allowing: '{message}'"
                    )
                    return GuardrailOutput(
                        is_violation=False,
                        violation_type=None,
                        reasoning="Simple greeting - appropriate for conversation starter",
                    )

            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Use the configured guardrail prompt
            guardrail_prompt = StudyGuruConfig.PROMPTS.GUARDRAIL_CHECK

            # Create chain using dedicated guardrail model (use dynamic model if provided)
            if assistant_model:
                guardrail_model = StudyGuruConfig.MODELS.get_guardrail_model(
                    temperature=0.2,
                    max_tokens=400,
                    model_name=assistant_model,
                    subscription_plan=subscription_plan,
                )
                chain = guardrail_prompt | guardrail_model | self.guardrail_parser
            else:
                chain = guardrail_prompt | self.guardrail_llm | self.guardrail_parser

            # Build content - analyze images if they exist for better accuracy
            if media_urls:
                # For images, use vision model to properly analyze educational content
                print(
                    f"🛡️ GUARDRAIL: Analyzing {len(media_urls)} image(s) for educational content"
                )

                # Use vision model for image analysis
                vision_chain = (
                    guardrail_prompt | self.vision_llm | self.guardrail_parser
                )

                # Build multimodal content for vision model
                multimodal_content = self._build_multimodal_content(message, media_urls)

                # Run guardrail check with vision model
                result = await vision_chain.ainvoke(
                    {"content": multimodal_content},
                    config={"callbacks": [callback_handler]},
                )
            else:
                # Text-only content
                content = f"User message: {message}"
                result = await chain.ainvoke(
                    {"content": content},
                    config={"callbacks": [callback_handler]},
                )

            # Convert dict result to GuardrailOutput object
            if isinstance(result, dict):
                guardrail_output = GuardrailOutput(
                    is_violation=result.get("is_violation", False),
                    violation_type=result.get("violation_type"),
                    reasoning=result.get("reasoning", "No reasoning provided"),
                )
                return guardrail_output
            return result

        except Exception as e:
            return GuardrailOutput(
                is_violation=False,
                violation_type=None,
                reasoning=f"Guardrail check failed: {str(e)}",
            )

    def _build_multimodal_content(
        self, message: str, media_urls: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Build multimodal content for LangChain messages"""
        content = []

        if message:
            content.append({"type": "text", "text": f"User message: {message}"})

        if media_urls:
            for url in media_urls:
                content.append(self._create_media_content(url))

        if not content:
            content.append({"type": "text", "text": "No content provided"})

        return content

    def _create_media_content(self, url: str) -> Dict[str, Any]:
        """Create appropriate content structure based on media type"""
        # Extract file extension from URL
        file_extension = url.lower().split(".")[-1] if "." in url else ""

        # Define media type mappings
        image_extensions = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "svg"}
        document_extensions = {"pdf", "doc", "docx", "txt", "rtf"}

        if file_extension in image_extensions:
            return {"type": "image_url", "image_url": {"url": url}}
        elif file_extension in document_extensions:
            # For documents, we'll use text type with a note about the document
            return {"type": "text", "text": f"Please analyze this document: {url}"}
        else:
            # Default to image_url for unknown types (backward compatibility)
            return {"type": "image_url", "image_url": {"url": url}}

    def _is_simple_question(self, message: str) -> bool:
        """Check if a question is simple and doesn't need extensive context processing"""
        if not message:
            return True

        message_lower = message.lower().strip()

        # Simple question patterns that don't need extensive context
        simple_patterns = [
            r"^tell me about\s+",
            r"^what is\s+",
            r"^explain\s+",
            r"^define\s+",
            r"^how does\s+",
            r"^why\s+",
            r"^when\s+",
            r"^where\s+",
            r"^who\s+",
        ]

        # Check if message matches simple patterns
        for pattern in simple_patterns:
            if re.match(pattern, message_lower):
                return True

        # Check if message is short and doesn't contain complex terms
        if len(message) < 50 and not any(
            term in message_lower
            for term in [
                "solve",
                "calculate",
                "equation",
                "formula",
                "problem",
                "question",
                "mcq",
            ]
        ):
            return True

        return False

    async def generate_conversation_response(
        self,
        message: str,
        context: str = "",
        media_urls: Optional[List[str]] = None,
        interaction_title: Optional[str] = None,
        interaction_summary: Optional[str] = None,
        max_tokens: int = 5000,  # Updated default max_tokens
        visualize_model: Optional[str] = None,  # Model for image/document analysis
        assistant_model: Optional[str] = None,  # Model for text conversation
        subscription_plan: Optional[
            str
        ] = None,  # User's subscription plan for validation
    ) -> Tuple[str, int, int, int]:
        """Generate conversation response using LangChain with optimized settings"""
        try:
            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Build system prompt - check if this is a pure document analysis request
            is_document_analysis_only = (
                not message and media_urls and len(media_urls) > 0
            )

            if is_document_analysis_only:
                # Use the DOCUMENT_ANALYSIS prompt for pure image/document analysis
                system_prompt = StudyGuruConfig.PROMPTS.DOCUMENT_ANALYSIS.messages[
                    0
                ].prompt.template
                print("🔍 Using DOCUMENT_ANALYSIS prompt for pure document analysis")
            elif interaction_title and interaction_summary:
                # Use the optimized conversation prompt from config
                system_prompt = (
                    StudyGuruConfig.PROMPTS.CONVERSATION_WITH_CONTEXT.messages[
                        0
                    ].prompt.template.format(
                        interaction_title=interaction_title,
                        interaction_summary=interaction_summary,
                    )
                )
            else:
                system_prompt = """
                You are StudyGuru AI, an educational assistant. Provide helpful, accurate educational assistance based on the user's question and any provided context. Keep responses concise but informative.
                """

            # Build user content
            user_content = []
            if context and not is_document_analysis_only:
                # Check if user is asking about a specific numbered item
                is_specific_query = re.search(
                    r"(equation|question|problem|mcq)\s+(\d+)",
                    message or "",
                    re.IGNORECASE,
                )

                if is_specific_query:
                    match = re.search(
                        r"(equation|question|problem|mcq)\s+(\d+)",
                        message,
                        re.IGNORECASE,
                    )
                    item_type = match.group(1)
                    item_number = match.group(2)

                    user_content.append(
                        {
                            "type": "text",
                            "text": f"**URGENT: USER IS ASKING ABOUT SPECIFIC {item_type.upper()} {item_number}**\n\n**CONTEXT TO SEARCH:**\n{context}\n\n**CRITICAL INSTRUCTIONS:**\n1. The user is asking about {item_type} {item_number} specifically\n2. **YOU MUST FIND {item_type.upper()} {item_number} IN THE CONTEXT ABOVE**\n3. Look for patterns like '{item_number}.' or '{item_type} {item_number}' in the context\n4. **DO NOT give generic information** - find and explain the exact {item_type} {item_number} from the context\n5. If you cannot find {item_type} {item_number} in the context, say so clearly and ask for clarification\n\n",
                        }
                    )
                else:
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"**IMPORTANT: USER'S LEARNING CONTEXT AND HISTORY:**\n{context}\n\n**CRITICAL INSTRUCTIONS:**\n1. Use this context to provide personalized, relevant responses\n2. Reference previous discussions, build upon existing knowledge, and maintain consistency with the user's learning journey\n3. **IF THE USER ASKS ABOUT A SPECIFIC QUESTION NUMBER** (like 'mcq 3', 'question 2', 'problem 1'), search the context for that exact numbered question and provide a direct answer\n4. Look for numbered questions, MCQ questions, or problems in the context and answer the specific one the user is asking about\n\n",
                        }
                    )

            if message:
                # Check if this is a simple greeting - don't add prefix for simple messages
                from app.constants import ALLOWED_SIMPLE_MESSAGES

                if message.lower().strip() in ALLOWED_SIMPLE_MESSAGES:
                    # For simple greetings, just use the message directly
                    user_content.append({"type": "text", "text": message})
                else:
                    # For substantive questions, add the prefix
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"**CURRENT USER QUESTION:** {message}",
                        }
                    )
            elif media_urls:
                if is_document_analysis_only:
                    # Use the proper document analysis format
                    user_content.append(
                        {"type": "text", "text": "Please analyze this document/image:"}
                    )
                else:
                    user_content.append(
                        {"type": "text", "text": "**Document/Image Analysis Request:**"}
                    )

            # Add media files
            if media_urls:
                for url in media_urls:
                    user_content.append(self._create_media_content(url))

            # Create prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )

            # Create chain with optimized model
            if is_document_analysis_only:
                # Use vision model and document parser for pure document analysis
                model_to_use = visualize_model or None
                print(f"🔍 Using vision model: {model_to_use or 'default'}")
                optimized_llm = StudyGuruConfig.MODELS.get_vision_model(
                    temperature=0.3,
                    max_tokens=max_tokens,
                    model_name=model_to_use,
                    subscription_plan=subscription_plan,
                )
                chain = prompt | optimized_llm | self.document_parser
                print("🔍 Using vision model and document parser for document analysis")
            else:
                # Use chat model and string parser for regular conversations
                model_to_use = assistant_model or None
                print(f"💬 Using chat model: {model_to_use or 'default'}")
                optimized_llm = StudyGuruConfig.MODELS.get_chat_model(
                    temperature=0.2,
                    max_tokens=max_tokens,
                    model_name=model_to_use,
                    subscription_plan=subscription_plan,
                )
                chain = prompt | optimized_llm | self.string_parser

            # Generate response
            response = await chain.ainvoke({}, config={"callbacks": [callback_handler]})

            return (
                response,
                callback_handler.input_tokens,
                callback_handler.output_tokens,
                callback_handler.tokens_used,
            )

        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

    async def generate_conversation_response_streaming(
        self,
        message: str,
        context: str = "",
        media_urls: Optional[List[str]] = None,
        interaction_title: Optional[str] = None,
        interaction_summary: Optional[str] = None,
        max_tokens: int = 5000,
        visualize_model: Optional[str] = None,  # Model for image/document analysis
        assistant_model: Optional[str] = None,  # Model for text conversation
        subscription_plan: Optional[
            str
        ] = None,  # User's subscription plan for validation
    ):
        """Generate streaming conversation response for faster perceived performance"""
        try:
            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Build system prompt (same as non-streaming) - check if this is a pure document analysis request
            is_document_analysis_only = (
                not message and media_urls and len(media_urls) > 0
            )

            if is_document_analysis_only:
                # Use the DOCUMENT_ANALYSIS prompt for pure image/document analysis
                system_prompt = StudyGuruConfig.PROMPTS.DOCUMENT_ANALYSIS.messages[
                    0
                ].prompt.template

            elif interaction_title and interaction_summary:
                # Use the optimized conversation prompt from config
                system_prompt = (
                    StudyGuruConfig.PROMPTS.CONVERSATION_WITH_CONTEXT.messages[
                        0
                    ].prompt.template.format(
                        interaction_summary=interaction_summary,
                    )
                )
            else:
                # Optimized system prompt for streaming - simpler and faster
                system_prompt = """
                You are StudyGuru AI, an educational assistant. Provide helpful, accurate educational assistance. Keep responses concise and informative. For simple questions, give direct answers without extensive context processing.
                """

            # OPTIMIZATION: Detect simple queries for faster processing
            is_simple_query = self._is_simple_question(message or "")

            # Build user content
            user_content = []
            # OPTIMIZATION: Limit context length and skip for simple queries ONLY if no explicit context provided
            # If context is provided (e.g. mindmap), we should use it regardless of query simplicity
            should_use_context = context and not is_document_analysis_only

            if should_use_context:
                # Truncate context if too long (increased limit for more context)
                if len(context) > 8000:  # Increased from 5000 to 8000
                    context = (
                        context[:8000] + "...\n[Context truncated for faster response]"
                    )

                # Check if user is asking about a specific numbered item
                is_specific_query = re.search(
                    r"(equation|question|problem|mcq)\s+(\d+)",
                    message or "",
                    re.IGNORECASE,
                )

                if is_specific_query:
                    match = re.search(
                        r"(equation|question|problem|mcq)\s+(\d+)",
                        message,
                        re.IGNORECASE,
                    )
                    item_type = match.group(1)
                    item_number = match.group(2)

                    user_content.append(
                        {
                            "type": "text",
                            "text": f"**URGENT: USER IS ASKING ABOUT SPECIFIC {item_type.upper()} {item_number}**\n\n**CONTEXT TO SEARCH:**\n{context}\n\n**CRITICAL INSTRUCTIONS:**\n1. The user is asking about {item_type} {item_number} specifically\n2. **YOU MUST FIND {item_type.upper()} {item_number} IN THE CONTEXT ABOVE**\n3. Look for patterns like '{item_number}.' or '{item_type} {item_number}' in the context\n4. **DO NOT give generic information** - find and explain the exact {item_type} {item_number} from the context\n5. If you cannot find {item_type} {item_number} in the context, say so clearly and ask for clarification\n\n",
                        }
                    )
                else:
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"**IMPORTANT: USER'S LEARNING CONTEXT AND HISTORY:**\n{context}\n\n**CRITICAL INSTRUCTIONS:**\n1. Use this context to provide personalized, relevant responses\n2. Reference previous discussions, build upon existing knowledge, and maintain consistency with the user's learning journey\n3. **IF THE USER ASKS ABOUT A SPECIFIC QUESTION NUMBER** (like 'mcq 3', 'question 2', 'problem 1'), search the context for that exact numbered question and provide a direct answer\n4. Look for numbered questions, MCQ questions, or problems in the context and answer the specific one the user is asking about\n\n",
                        }
                    )

            if message:
                # Check if this is a simple greeting - don't add prefix for simple messages
                from app.constants import ALLOWED_SIMPLE_MESSAGES

                if message.lower().strip() in ALLOWED_SIMPLE_MESSAGES:
                    # For simple greetings, just use the message directly
                    user_content.append({"type": "text", "text": message})
                else:
                    # For substantive questions, add the prefix
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"**CURRENT USER QUESTION:** {message}",
                        }
                    )
            elif media_urls:
                if is_document_analysis_only:
                    # Use the proper document analysis format
                    user_content.append(
                        {"type": "text", "text": "Please analyze this document/image:"}
                    )
                else:
                    user_content.append(
                        {"type": "text", "text": "**Document/Image Analysis Request:**"}
                    )

            # Add media files
            if media_urls:
                for url in media_urls:
                    user_content.append(self._create_media_content(url))

            # Create prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )

            # Create streaming chain with optimized settings
            if is_document_analysis_only:
                # Use vision model for pure document analysis (streaming not supported for document parser)
                model_to_use = visualize_model or None
                print(
                    f"🔍 [STREAMING] Requested vision model: {model_to_use or 'default (auto-select)'}"
                )
                optimized_llm = StudyGuruConfig.MODELS.get_vision_model(
                    temperature=0.3,
                    max_tokens=max_tokens,
                    model_name=model_to_use,
                    subscription_plan=subscription_plan,
                )
                # Log actual model being used - use requested model name (it's what we're actually using)
                # Try to get from instance, but fallback to requested name
                actual_model = (
                    getattr(optimized_llm, "model_name", None)
                    or getattr(optimized_llm, "model", None)
                    or model_to_use
                    or "default"
                )
                print(
                    f"✅ [STREAMING] Using vision model: {actual_model} for document analysis"
                )
                chain = prompt | optimized_llm
            else:
                # Use chat model for regular conversations with speed optimizations
                model_to_use = assistant_model or None
                print(
                    f"💬 [STREAMING] Requested chat model: {model_to_use or 'default (auto-select)'}"
                )

                # OPTIMIZATION: Use lower max_tokens for simple queries
                optimized_max_tokens = (
                    min(max_tokens, 2000) if is_simple_query else max_tokens
                )

                optimized_llm = StudyGuruConfig.MODELS.get_chat_model(
                    temperature=0.1,  # Lower temperature for faster, more deterministic responses
                    max_tokens=optimized_max_tokens,  # Reduced tokens for simple queries
                    model_name=model_to_use,
                    subscription_plan=subscription_plan,
                    streaming=True,  # Ensure streaming is enabled
                )
                # Log actual model being used - use requested model name (it's what we're actually using)
                # Try to get from instance, but fallback to requested name
                actual_model = (
                    getattr(optimized_llm, "model_name", None)
                    or getattr(optimized_llm, "model", None)
                    or model_to_use
                    or "default"
                )
                print(
                    f"✅ [STREAMING] Using chat model: {actual_model} (temperature=0.1, max_tokens={optimized_max_tokens}, simple_query={is_simple_query})"
                )
                chain = prompt | optimized_llm

            # Stream response using direct model streaming
            full_response = ""

            # Use the model's streaming capability directly
            if hasattr(optimized_llm, "astream"):

                # Try direct streaming from the model
                messages = prompt.format_messages()
                async for chunk in optimized_llm.astream(
                    messages,
                    config={"callbacks": [callback_handler]},
                ):
                    if hasattr(chunk, "content") and chunk.content:

                        full_response += chunk.content
                        yield chunk.content
            else:

                # Fallback: Use chain streaming with proper configuration
                async for chunk in chain.astream(
                    {},
                    config={"callbacks": [callback_handler], "stream_mode": "values"},
                ):
                    if hasattr(chunk, "content") and chunk.content:

                        full_response += chunk.content
                        yield chunk.content

            # Store the final response and token usage in the callback handler
            callback_handler.final_response = full_response

        except Exception as e:
            raise Exception(f"Failed to generate streaming response: {str(e)}")

    async def similarity_search(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using enhanced vector optimization service with cross-model compatibility"""
        try:
            # Use the new vector optimization service for better results
            from app.services.vector_optimization_service import (
                vector_optimization_service,
                SearchQuery,
            )

            search_query = SearchQuery(
                query=query,
                user_id=user_id,
                top_k=top_k,
                use_hybrid_search=True,
                use_query_expansion=True,
                boost_recent=True,
            )

            # Get enhanced search results
            search_results = await vector_optimization_service.hybrid_search(
                search_query
            )

            # Convert SearchResult objects to expected format
            results = []
            for result in search_results:
                # Truncate content for faster processing (increased limit for more context)
                content = result.content
                original_length = len(content)
                if len(content) > 1000:  # Increased from 300 to 1000 for more context
                    content = content[:1000] + "..."
                    print(
                        f"🔍 [SIMILARITY SEARCH] Content truncated: {original_length} -> {len(content)} chars"
                    )
                else:
                    print(
                        f"🔍 [SIMILARITY SEARCH] Content length: {len(content)} chars"
                    )
                print(
                    f"🔍 [SIMILARITY SEARCH] Title: {result.title}, Type: {result.content_type}"
                )

                results.append(
                    {
                        "id": result.id,
                        "interaction_id": result.interaction_id,
                        "title": result.title,
                        "content": content,
                        "metadata": result.metadata,
                        "score": result.score,
                        "type": result.content_type,
                        "relevance_score": result.relevance_score,
                        "recency_score": result.recency_score,
                        "importance_score": result.importance_score,
                        "topic_tags": result.topic_tags,
                        "question_numbers": result.question_numbers,
                    }
                )

            return results

        except Exception as e:

            # Fallback to basic search
            return await self._basic_similarity_search(query, user_id, top_k)

    async def _basic_similarity_search(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Fallback basic similarity search implementation"""
        try:
            if not self.vector_store:
                print("⚠️ Vector store not available for similarity search")
                return []

            # Create retriever with user filter and reduced top_k for faster search
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": min(top_k, 5), "expr": f"user_id == '{user_id}'"}
            )

            # Run search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=1), retriever.invoke, query
            )

            # Convert to expected format with content truncation for faster processing
            results = []
            for doc in docs:
                content = doc.page_content
                # Truncate content early to reduce processing time
                if len(content) > 300:  # Reduced from 500
                    content = content[:300] + "..."

                # Extract interaction_id from metadata for faster access
                metadata = doc.metadata or {}
                interaction_id = metadata.get("interaction_id", "")

                results.append(
                    {
                        "id": doc.metadata.get("original_id", doc.metadata.get("id")),
                        "interaction_id": interaction_id,  # Add interaction_id for faster access
                        "title": doc.metadata.get("title", ""),
                        "content": content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", 0.0),
                        "type": metadata.get(
                            "type", ""
                        ),  # Add type for better filtering
                    }
                )

            return results

        except Exception as e:
            return []

    async def similarity_search_by_interaction(
        self, query: str, user_id: str, interaction_id: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Fast similarity search prioritizing specific interaction results using enhanced optimization"""
        try:
            # Use the new vector optimization service for better results
            from app.services.vector_optimization_service import (
                vector_optimization_service,
                SearchQuery,
            )

            search_query = SearchQuery(
                query=query,
                user_id=user_id,
                interaction_id=interaction_id,
                top_k=top_k,
                use_hybrid_search=True,
                use_query_expansion=True,
                boost_recent=True,
                boost_interaction=True,  # Boost results from same interaction
            )

            # Get enhanced search results
            search_results = await vector_optimization_service.hybrid_search(
                search_query
            )

            # Convert SearchResult objects to expected format
            results = []
            for result in search_results:
                # Increased truncation limit for more context
                content = result.content
                original_length = len(content)
                if len(content) > 1000:  # Increased from 150 to 1000 for more context
                    content = content[:1000] + "..."
                    print(
                        f"🔍 [INTERACTION SEARCH] Content truncated: {original_length} -> {len(content)} chars"
                    )
                else:
                    print(
                        f"🔍 [INTERACTION SEARCH] Content length: {len(content)} chars"
                    )
                print(
                    f"🔍 [INTERACTION SEARCH] Title: {result.title}, Type: {result.content_type}, Score: {result.score}"
                )

                results.append(
                    {
                        "id": result.id,
                        "interaction_id": result.interaction_id,
                        "title": result.title,
                        "content": content,
                        "metadata": result.metadata,
                        "score": result.score,
                        "type": result.content_type,
                        "priority": "high",  # Mark as high priority since it's from same interaction
                        "relevance_score": result.relevance_score,
                        "recency_score": result.recency_score,
                        "importance_score": result.importance_score,
                        "topic_tags": result.topic_tags,
                        "question_numbers": result.question_numbers,
                    }
                )

            return results

        except Exception as e:
            print(
                f"⚠️ Enhanced interaction search failed, falling back to basic search: {e}"
            )
            # Fallback to original implementation
            return await self._basic_similarity_search_by_interaction(
                query, user_id, interaction_id, top_k
            )

    async def _basic_similarity_search_by_interaction(
        self, query: str, user_id: str, interaction_id: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Fallback basic similarity search by interaction implementation"""
        try:
            if not self.vector_store:
                return []

            # Create retriever with user and interaction filter for faster, more targeted search
            # Handle both cases: interaction_id field exists or fallback to metadata filtering
            try:
                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": min(top_k, 3),
                        "expr": f"user_id == '{user_id}' && interaction_id == '{interaction_id}'",
                    }
                )
            except Exception as expr_error:
                # Fallback: if interaction_id field doesn't exist, use user filter only
                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": min(
                            top_k * 3, 10
                        ),  # Get more results to filter client-side
                        "expr": f"user_id == '{user_id}'",
                    }
                )

            # Run search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=1), retriever.invoke, query
            )

            # Convert to expected format with minimal processing for speed
            results = []
            for doc in docs:
                content = doc.page_content
                # Minimal truncation for speed
                if len(content) > 150:  # Even more aggressive truncation
                    content = content[:150] + "..."

                metadata = doc.metadata or {}
                doc_interaction_id = metadata.get("interaction_id")

                # If using fallback method, filter by interaction_id client-side
                if doc_interaction_id != interaction_id:
                    continue

                results.append(
                    {
                        "id": doc.metadata.get("original_id", doc.metadata.get("id")),
                        "interaction_id": interaction_id,  # Already filtered
                        "title": doc.metadata.get("title", ""),
                        "content": content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", 0.0),
                        "type": metadata.get("type", ""),
                        "priority": "high",  # Mark as high priority since it's from same interaction
                    }
                )

            return results

        except Exception as e:
            return []

    async def upsert_embedding(
        self,
        conv_id: str,
        user_id: str,
        text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert embedding using LangChain vector store with enhanced metadata"""
        try:
            if not self.vector_store:
                print(f"⚠️ Vector store not available for embedding creation")
                return False

            # Import json at the top of the function to avoid scope issues
            import json

            # Ensure text is a string - handle dict/object inputs
            if isinstance(text, dict):
                # Convert dict to JSON string for embedding
                text = json.dumps(text, indent=2)
            elif not isinstance(text, str):
                # Convert any other type to string
                text = str(text)

            # Truncate text to reduce embedding time and storage (increased limit for more context)
            original_text_length = len(text)
            if len(text) > 3000:  # Increased from 1000 to 3000 for more context
                text = text[:3000] + "..."
                print(
                    f"🔍 [EMBEDDING] Text truncated to 3000 chars (original: {original_text_length} chars)"
                )
            else:
                print(f"🔍 [EMBEDDING] Storing text: {len(text)} chars, title: {title}")

            # Extract interaction_id from metadata if present
            interaction_id = None
            if metadata and "interaction_id" in metadata:
                interaction_id = metadata["interaction_id"]

            # Enhanced metadata extraction
            enhanced_metadata = self._extract_enhanced_metadata(text, title, metadata)

            # Create document (ID will be auto-generated by Milvus)
            doc = Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "interaction_id": interaction_id,
                    "title": title or "",
                    "original_id": conv_id,  # Store original ID in metadata
                    "metadata": json.dumps(metadata or {}),
                    "created_at": datetime.now().isoformat(),
                    "content_type": enhanced_metadata.get("content_type", "unknown"),
                    "topic_tags": enhanced_metadata.get("topic_tags", []),
                    "question_numbers": enhanced_metadata.get("question_numbers", []),
                    "key_concepts": enhanced_metadata.get("key_concepts", []),
                    "difficulty_level": enhanced_metadata.get(
                        "difficulty_level", "unknown"
                    ),
                    "subject_area": enhanced_metadata.get("subject_area", "unknown"),
                    **(metadata or {}),
                },
            )

            # Run embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=1),
                self.vector_store.add_documents,
                [doc],
            )

            return True

        except Exception as e:
            print(f"❌ Error upserting embedding: {e}")
            return False

    def _extract_enhanced_metadata(
        self, text: str, title: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract enhanced metadata from text content"""
        try:
            # Import json at the top of the function to avoid scope issues
            import json

            enhanced_metadata = {
                "content_type": "unknown",
                "topic_tags": [],
                "question_numbers": [],
                "key_concepts": [],
                "difficulty_level": "unknown",
                "subject_area": "unknown",
            }

            # Ensure text is a string for processing
            if isinstance(text, dict):
                text = json.dumps(text, indent=2)
            elif not isinstance(text, str):
                text = str(text)

            text_lower = text.lower()
            title_lower = (title or "").lower()

            # Detect content type
            if any(
                keyword in text_lower
                for keyword in ["mcq", "multiple choice", "a)", "b)", "c)", "d)"]
            ):
                enhanced_metadata["content_type"] = "mcq"
            elif any(
                keyword in text_lower
                for keyword in ["equation", "formula", "solve", "calculate"]
            ):
                enhanced_metadata["content_type"] = "equation"
            elif any(
                keyword in text_lower
                for keyword in ["explain", "describe", "what is", "how does"]
            ):
                enhanced_metadata["content_type"] = "explanation"
            elif any(
                keyword in text_lower for keyword in ["problem", "question", "exercise"]
            ):
                enhanced_metadata["content_type"] = "problem"
            else:
                enhanced_metadata["content_type"] = "general"

            # Extract question numbers
            question_numbers = re.findall(r"\b(\d+)\.\s+", text)
            enhanced_metadata["question_numbers"] = list(set(question_numbers))

            # Extract topic tags based on common educational terms
            topic_keywords = {
                "mathematics": [
                    "math",
                    "algebra",
                    "geometry",
                    "calculus",
                    "trigonometry",
                    "equation",
                    "formula",
                ],
                "science": [
                    "physics",
                    "chemistry",
                    "biology",
                    "experiment",
                    "hypothesis",
                    "theory",
                ],
                "programming": [
                    "code",
                    "programming",
                    "algorithm",
                    "function",
                    "variable",
                    "loop",
                ],
                "language": [
                    "grammar",
                    "vocabulary",
                    "sentence",
                    "paragraph",
                    "essay",
                    "writing",
                ],
                "history": [
                    "historical",
                    "century",
                    "war",
                    "revolution",
                    "ancient",
                    "medieval",
                ],
            }

            topic_tags = []
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topic_tags.append(topic)

            enhanced_metadata["topic_tags"] = topic_tags

            # Extract key concepts (simple keyword extraction)
            key_concepts = []
            concept_patterns = [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized terms
                r"\b\w+ing\b",  # -ing words
                r"\b\w+tion\b",  # -tion words
            ]

            for pattern in concept_patterns:
                concepts = re.findall(pattern, text)
                key_concepts.extend(concepts[:5])  # Limit to 5 concepts

            enhanced_metadata["key_concepts"] = list(set(key_concepts))[:10]

            # Determine difficulty level
            if any(
                keyword in text_lower
                for keyword in ["basic", "simple", "easy", "beginner"]
            ):
                enhanced_metadata["difficulty_level"] = "beginner"
            elif any(
                keyword in text_lower
                for keyword in ["advanced", "complex", "difficult", "expert"]
            ):
                enhanced_metadata["difficulty_level"] = "advanced"
            else:
                enhanced_metadata["difficulty_level"] = "intermediate"

            # Determine subject area
            if enhanced_metadata["topic_tags"]:
                enhanced_metadata["subject_area"] = enhanced_metadata["topic_tags"][0]
            elif any(
                keyword in text_lower
                for keyword in ["math", "mathematics", "algebra", "geometry"]
            ):
                enhanced_metadata["subject_area"] = "mathematics"
            elif any(
                keyword in text_lower
                for keyword in ["science", "physics", "chemistry", "biology"]
            ):
                enhanced_metadata["subject_area"] = "science"
            else:
                enhanced_metadata["subject_area"] = "general"

            return enhanced_metadata

        except Exception as e:
            print(f"⚠️ Error extracting enhanced metadata: {e}")
            return {
                "content_type": "unknown",
                "topic_tags": [],
                "question_numbers": [],
                "key_concepts": [],
                "difficulty_level": "unknown",
                "subject_area": "unknown",
            }

    async def delete_embeddings_by_interaction(self, interaction_id: str) -> bool:
        """Delete all embeddings for a specific interaction"""
        try:
            if not self.vector_store:
                return False

            # Get the collection from the vector store (single collection for both models)
            collection_config = StudyGuruConfig.VECTOR_STORE.get_collection_config()
            collection_name = collection_config["collection_name"]
            collection = Collection(collection_name)

            # Load collection to memory for operations
            collection.load()

            # Delete embeddings where interaction_id matches
            # Using expr to filter by interaction_id in metadata
            expr = f'interaction_id == "{interaction_id}"'

            # Execute deletion
            result = collection.delete(expr=expr)

            # Flush to ensure deletion is persisted
            collection.flush()

            return True

        except Exception as e:
            return False

    def calculate_dynamic_tokens(self, file_count: int = 0) -> int:
        """Calculate dynamic token limit based on file count"""
        from app.config.langchain_config import StudyGuruConfig

        return StudyGuruConfig.calculate_dynamic_tokens(file_count)

    def calculate_points_cost(self, tokens_used: int) -> int:
        """Calculate points cost based on tokens used"""
        return max(1, tokens_used // 100)

    async def generate_interaction_title(
        self,
        message: str,
        response_preview: str,
        assistant_model: Optional[str] = None,
        subscription_plan: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate interaction title and summary using cost-efficient model"""
        try:
            # Limit input sizes for cost efficiency
            limited_message = message[:200] if message else ""
            limited_response = response_preview[:300] if response_preview else ""

            # Create callback handler for minimal token tracking
            callback_handler = StudyGuruCallbackHandler()

            # Use the improved title generation chain with robust error handling
            from app.config.langchain_config import StudyGuruChains

            # Use the improved chain that handles GPT-5 JSON issues (use dynamic model if provided)
            title_chain = StudyGuruChains.get_title_generation_chain(
                model_name=assistant_model,
                subscription_plan=subscription_plan,
            )

            # Generate title with multiple fallback attempts and increased timeout
            result = None
            json_parse_error = False
            for attempt in range(3):  # Try up to 3 times
                try:
                    # Use asyncio.wait_for to add timeout protection
                    result = await asyncio.wait_for(
                        title_chain.ainvoke(
                            {
                                "message": limited_message,
                                "response_preview": limited_response,
                            },
                            config={"callbacks": [callback_handler]},
                        ),
                        timeout=25.0,  # 25 second timeout per attempt
                    )

                    # Validate result
                    if result and isinstance(result, dict) and result.get("title"):
                        break
                    else:
                        if attempt == 2:  # Last attempt
                            json_parse_error = True
                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(0.5)  # Brief delay before retry

                except asyncio.TimeoutError:
                    if attempt == 2:  # Last attempt
                        json_parse_error = True
                    if attempt < 2:  # Don't sleep on last attempt
                        await asyncio.sleep(
                            1.0
                        )  # Longer delay before retry for timeout
                except Exception as chain_error:
                    error_str = str(chain_error).lower()
                    # Check if it's a JSON parsing error
                    if (
                        "json" in error_str
                        or "parse" in error_str
                        or "output_parsing" in error_str
                    ):
                        json_parse_error = True
                        # Only log on last attempt to reduce noise
                        if attempt == 2:
                            print(
                                f"⚠️ Title generation failed: JSON parsing error after 3 attempts"
                            )
                    else:
                        # Log non-JSON errors
                        if attempt == 2:  # Only log on last attempt
                            print(f"⚠️ Title generation failed: {chain_error}")

                    if attempt < 2:  # Don't sleep on last attempt
                        await asyncio.sleep(0.5)  # Brief delay before retry

            # Extract title and summary with robust validation
            if result and isinstance(result, dict):
                title = result.get("title", "")[:50] if result.get("title") else None
                summary_title = (
                    result.get("summary_title", "")[:100]
                    if result.get("summary_title")
                    else None
                )
            else:
                title = None
                summary_title = None

            # If all attempts failed, use fallback
            if not title and not summary_title:
                # Fallback: create simple title from message
                if message:
                    simple_title = message[:40].strip()
                    return (simple_title, f"Help with {simple_title.lower()}")
                elif response_preview:
                    simple_title = response_preview[:40].strip()
                    return (simple_title, "Educational assistance")
                else:
                    return ("Study Session", "Educational assistance")

            return title, summary_title

        except Exception as e:
            # Only print traceback for non-JSON parsing errors to reduce noise
            if "json" not in str(e).lower() and "parse" not in str(e).lower():
                import traceback

                traceback.print_exc()

            # Fallback: create simple title from message
            if message:
                simple_title = message[:40].strip()
                fallback_result = (simple_title, f"Help with {simple_title.lower()}")

                return fallback_result
            elif response_preview:
                # Try to create title from response if no message
                simple_title = response_preview[:40].strip()
                fallback_result = (simple_title, "Educational assistance")

                return fallback_result
            else:
                # Last resort fallback
                fallback_result = ("Study Session", "Educational assistance")

                return fallback_result

    async def summarize_conversation(
        self, user_message: str, ai_response: str
    ) -> Dict[str, Any]:
        """
        Extract key facts and create a semantic summary from a single conversation exchange.

        Returns a dictionary with:
        - key_facts: List of important facts discussed
        - main_topics: List of main topics covered
        - semantic_summary: Concise summary of the conversation
        - important_terms: List of important terms mentioned
        - context_for_future: Context useful for future conversations
        - question_numbers: List of question numbers referenced
        - learning_progress: What the user has learned
        - potential_follow_ups: Potential follow-up questions
        - difficulty_level: Difficulty level of the content
        - subject_area: Subject area of the content
        """
        try:
            from app.services.semantic_summary_service import semantic_summary_service

            # Use the enhanced semantic summary service
            result = await semantic_summary_service.create_conversation_summary(
                user_message, ai_response
            )

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()

            # Return minimal summary on error
            return {
                "key_facts": [],
                "main_topics": ["General Discussion"],
                "semantic_summary": "Educational conversation about various topics.",
                "important_terms": [],
                "context_for_future": "User is engaged in educational learning.",
                "question_numbers": [],
                "learning_progress": "Learning in progress",
                "potential_follow_ups": [],
                "difficulty_level": "beginner",
                "subject_area": "other",
            }

    async def update_interaction_summary(
        self,
        current_summary: Optional[Dict[str, Any]],
        new_user_message: str,
        new_ai_response: str,
    ) -> Dict[str, Any]:
        """
        Update the running semantic summary of an interaction with a new conversation.

        Args:
            current_summary: The existing accumulated summary (or None for first conversation)
            new_user_message: The latest user message
            new_ai_response: The latest AI response

        Returns a dictionary with:
        - updated_summary: Comprehensive running summary
        - key_topics: All important topics covered so far
        - recent_focus: What the user has been focusing on recently
        - accumulated_facts: Critical facts to remember
        - question_numbers: All question numbers referenced
        - learning_progression: How the user's understanding has evolved
        - difficulty_trend: Current difficulty level
        - learning_patterns: Patterns in user's learning
        - struggling_areas: Areas where user needs help
        - mastered_concepts: Concepts the user has mastered
        - version: Version number of the summary
        - last_updated: Timestamp of last update
        """
        try:
            from app.services.semantic_summary_service import semantic_summary_service

            # Use the enhanced semantic summary service
            result = await semantic_summary_service.update_interaction_summary(
                current_summary, new_user_message, new_ai_response
            )

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()

            # Return current summary unchanged on error
            if current_summary:
                return current_summary
            else:
                return {
                    "updated_summary": "Educational conversation in progress.",
                    "key_topics": ["General Discussion"],
                    "recent_focus": "User learning",
                    "accumulated_facts": [],
                    "question_numbers": [],
                    "learning_progression": "Learning in progress",
                    "difficulty_trend": "beginner",
                    "learning_patterns": [],
                    "struggling_areas": [],
                    "mastered_concepts": [],
                    "version": 1,
                    "last_updated": datetime.now().isoformat(),
                }

    async def generate_mindmap(
        self,
        topic: str,
        context: str = "",
        max_tokens: int = 2000,
        assistant_model: Optional[str] = None,
        subscription_plan: Optional[str] = None,
    ) -> MindmapOutput:
        """
        Generate hierarchical mindmap structure for a given topic.

        Args:
            topic: The main topic for the mindmap
            max_tokens: Maximum tokens for generation
            assistant_model: Model to use for generation
            subscription_plan: User's subscription plan

        Returns:
            MindmapOutput with hierarchical node structure
        """
        try:
            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Build system prompt for mindmap generation
            system_prompt = """You are an expert at creating educational mindmaps that help students visualize and understand complex topics.

CRITICAL INSTRUCTIONS:
1. **Create a hierarchical structure** with the main topic as the root
2. **Each node must have**:
   - Unique ID (use format: "node_1", "node_2", etc.)
   - Content (concise, 2-5 words)
   - Level (0 for root, 1 for main branches, 2+ for sub-branches)
   - Parent ID (null for root)
   - Color (use different colors for different levels)
   - Children (list of child nodes)
3. **Aim for 8-15 total nodes** for clarity
4. **Use colors**:
   - Level 0 (root): #4A90E2 (blue)
   - Level 1: #E74C3C (red), #2ECC71 (green), #F39C12 (orange)
   - Level 2+: #9B59B6 (purple), #1ABC9C (teal), #E67E22 (dark orange)
5. **Keep content concise** - each node should be a key concept

Format your response as JSON with this structure:
{
  "topic": "Photosynthesis",
  "root_node": {
    "id": "node_0",
    "content": "Photosynthesis",
    "level": 0,
    "parent_id": null,
    "color": "#4A90E2",
    "children": [
      {
        "id": "node_1",
        "content": "Light Reactions",
        "level": 1,
        "parent_id": "node_0",
        "color": "#E74C3C",
        "children": []
      }
    ]
  },
  "total_nodes": 12
}"""

            # Build user content
            user_content = []

            # Add context if available
            if context:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"**USER'S LEARNING CONTEXT:**\\n{context}\\n\\n**INSTRUCTIONS:** Use this context to create a personalized mindmap that builds on concepts the user has already discussed and focuses on their areas of interest.\\n\\n",
                    }
                )

            user_content.append(
                {
                    "type": "text",
                    "text": f"**TOPIC:** {topic}\\n\\n**INSTRUCTIONS:** Create a comprehensive mindmap with 8-15 nodes covering the key concepts, subtopics, and relationships.",
                }
            )

            # Create prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )

            # Use chat model
            model = StudyGuruConfig.MODELS.get_chat_model(
                temperature=0.4,  # Slightly higher for creativity
                max_tokens=max_tokens,
                model_name=assistant_model,
                subscription_plan=subscription_plan,
            )

            # Use structured output with FLAT model to avoid recursion issues
            structured_model = model.with_structured_output(MindmapFlatOutput)

            # Create chain
            chain = prompt | structured_model

            # Generate flat mindmap
            flat_result = await chain.ainvoke(
                {}, config={"callbacks": [callback_handler]}
            )

            # Reconstruct tree from flat list
            nodes_map = {}
            root_node = None

            # First pass: Create MindmapNodes
            for flat_node in flat_result.nodes:
                node = MindmapNode(
                    id=flat_node.id,
                    content=flat_node.content,
                    level=flat_node.level,
                    parent_id=flat_node.parent_id,
                    color=flat_node.color,
                    children=[],
                )
                nodes_map[node.id] = node
                if node.level == 0 or node.parent_id is None:
                    root_node = node

            # Second pass: Build hierarchy
            for node_id, node in nodes_map.items():
                if node.parent_id and node.parent_id in nodes_map:
                    nodes_map[node.parent_id].children.append(node)

            # Fallback if no root found
            if not root_node and nodes_map:
                # Use the first node as root
                root_node = list(nodes_map.values())[0]

            if not root_node:
                raise ValueError(
                    "Could not reconstruct mindmap tree: No nodes generated"
                )

            return MindmapOutput(
                topic=flat_result.topic,
                root_node=root_node,
                total_nodes=flat_result.total_nodes,
            )

        except Exception as e:
            print(f"❌ Mindmap generation error: {e}")
            # Return fallback mindmap
            return MindmapOutput(
                topic=topic,
                root_node=MindmapNode(
                    id="node_0",
                    content=topic[:20],  # Truncate if too long
                    level=0,
                    parent_id=None,
                    color="#4A90E2",
                    children=[
                        MindmapNode(
                            id="node_1",
                            content="Error occurred",
                            level=1,
                            parent_id="node_0",
                            color="#E74C3C",
                            children=[],
                        )
                    ],
                ),
                total_nodes=2,
            )


# Global instance
langchain_service = LangChainService()
