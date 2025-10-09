import os
import json
from typing import Dict, Any, Optional, List, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
                print("Vector database not configured, skipping initialization")
                return

            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN,
                secure=True,
            )

            # Create or get collection
            collection_name = settings.ZILLIZ_COLLECTION
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
                        name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536
                    ),  # text-embedding-3-small dimension
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

            # Initialize vector store using Zilliz
            self.vector_store = Zilliz(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                connection_args={
                    "uri": settings.ZILLIZ_URI,
                    "token": settings.ZILLIZ_TOKEN,
                    "secure": True,
                },
            )

            print("Vector store initialized successfully")

        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            self.vector_store = None

    async def analyze_document(
        self, file_url: str, max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Analyze document using LangChain with vision capabilities"""
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
            chain = analysis_prompt | self.vision_llm | self.document_parser

            # Run analysis (no variables needed since we constructed the prompt directly)
            result = await chain.ainvoke({}, config={"callbacks": [callback_handler]})

            # Add token usage
            result["token"] = callback_handler.tokens_used

            return result

        except Exception as e:
            print(f"Document analysis error: {str(e)}")
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

    async def check_guardrails(
        self, message: str, image_urls: Optional[List[str]] = None
    ) -> GuardrailOutput:
        """Check content against guardrails using LangChain"""
        try:
            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Use the configured guardrail prompt
            guardrail_prompt = StudyGuruConfig.PROMPTS.GUARDRAIL_CHECK

            # Create chain using dedicated guardrail model
            chain = guardrail_prompt | self.guardrail_llm | self.guardrail_parser

            # Run guardrail check
            result = await chain.ainvoke(
                {"content": self._build_multimodal_content(message, image_urls)},
                config={"callbacks": [callback_handler]},
            )

            # Convert dict result to GuardrailOutput object
            if isinstance(result, dict):
                return GuardrailOutput(
                    is_violation=result.get("is_violation", False),
                    violation_type=result.get("violation_type"),
                    reasoning=result.get("reasoning", "No reasoning provided"),
                )
            return result

        except Exception as e:
            return GuardrailOutput(
                is_violation=False,
                violation_type=None,
                reasoning=f"Guardrail check failed: {str(e)}",
            )

    def _build_multimodal_content(
        self, message: str, image_urls: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Build multimodal content for LangChain messages"""
        content = []

        if message:
            content.append({"type": "text", "text": f"User message: {message}"})

        if image_urls:
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})

        if not content:
            content.append({"type": "text", "text": "No content provided"})

        return content

    async def generate_conversation_response(
        self,
        message: str,
        context: str = "",
        image_urls: Optional[List[str]] = None,
        interaction_title: Optional[str] = None,
        interaction_summary: Optional[str] = None,
        max_tokens: int = 800,  # Reduced default max_tokens
    ) -> Tuple[str, int, int, int]:
        """Generate conversation response using LangChain with optimized settings"""
        try:
            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Build system prompt
            if interaction_title and interaction_summary:
                system_prompt = f"""
                You are StudyGuru AI, an educational assistant. You have access to the user's learning history and context.
                
                Current conversation topic: {interaction_title}
                Context summary: {interaction_summary}
                
                Instructions:
                1. Use the retrieved context from the user's learning history when relevant to provide better, personalized responses
                2. If the context is not relevant to the current question, answer based on your knowledge
                3. Maintain consistency with the user's learning style and previous interactions
                4. Provide clear, educational explanations that build upon previous knowledge when possible
                5. If this is a follow-up question, reference relevant previous discussions when helpful
                
                Always provide helpful, accurate educational assistance. Keep responses concise but informative.
                """
            else:
                system_prompt = """
                You are StudyGuru AI, an educational assistant. Provide helpful, accurate educational assistance based on the user's question and any provided context. Keep responses concise but informative.
                """

            # Build user content
            user_content = []
            if context:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"**Relevant Learning Context:**\n{context}\n",
                    }
                )

            if message:
                user_content.append(
                    {"type": "text", "text": f"**Current Question:** {message}"}
                )
            elif image_urls:
                user_content.append(
                    {"type": "text", "text": "**Document/Image Analysis Request:**"}
                )

            # Add images
            if image_urls:
                for url in image_urls:
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )

            # Create prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )

            # Create chain with optimized model
            optimized_llm = StudyGuruConfig.MODELS.get_chat_model(
                temperature=0.2, max_tokens=max_tokens
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
        image_urls: Optional[List[str]] = None,
        interaction_title: Optional[str] = None,
        interaction_summary: Optional[str] = None,
        max_tokens: int = 800,
    ):
        """Generate streaming conversation response for faster perceived performance"""
        try:
            # Create callback handler
            callback_handler = StudyGuruCallbackHandler()

            # Build system prompt (same as non-streaming)
            if interaction_title and interaction_summary:
                system_prompt = f"""
                You are StudyGuru AI, an educational assistant. You have access to the user's learning history and context.
                
                Current conversation topic: {interaction_title}
                Context summary: {interaction_summary}
                
                Instructions:
                1. Use the retrieved context from the user's learning history when relevant to provide better, personalized responses
                2. If the context is not relevant to the current question, answer based on your knowledge
                3. Maintain consistency with the user's learning style and previous interactions
                4. Provide clear, educational explanations that build upon previous knowledge when possible
                5. If this is a follow-up question, reference relevant previous discussions when helpful
                
                Always provide helpful, accurate educational assistance. Keep responses concise but informative.
                """
            else:
                system_prompt = """
                You are StudyGuru AI, an educational assistant. Provide helpful, accurate educational assistance based on the user's question and any provided context. Keep responses concise but informative.
                """

            # Build user content
            user_content = []
            if context:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"**Relevant Learning Context:**\n{context}\n",
                    }
                )

            if message:
                user_content.append(
                    {"type": "text", "text": f"**Current Question:** {message}"}
                )
            elif image_urls:
                user_content.append(
                    {"type": "text", "text": "**Document/Image Analysis Request:**"}
                )

            # Add images
            if image_urls:
                for url in image_urls:
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )

            # Create prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )

            # Create streaming chain
            optimized_llm = StudyGuruConfig.MODELS.get_chat_model(
                temperature=0.2, max_tokens=max_tokens
            )
            chain = prompt | optimized_llm

            # Stream response
            full_response = ""
            async for chunk in chain.astream(
                {}, config={"callbacks": [callback_handler]}
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
        """Perform similarity search using LangChain vector store with async optimization"""
        try:
            if not self.vector_store:
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
            print(f"Similarity search error: {e}")
            return []

    async def similarity_search_by_interaction(
        self, query: str, user_id: str, interaction_id: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Fast similarity search prioritizing specific interaction results"""
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
                print(
                    f"Interaction field not available, using user filter only: {expr_error}"
                )
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
            print(f"Interaction-specific similarity search error: {e}")
            return []

    async def upsert_embedding(
        self,
        conv_id: str,
        user_id: str,
        text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert embedding using LangChain vector store with async optimization"""
        try:
            if not self.vector_store:
                return False

            # Truncate text to reduce embedding time and storage
            if len(text) > 1000:  # Limit text length for faster embedding
                text = text[:1000] + "..."

            # Extract interaction_id from metadata if present
            interaction_id = None
            if metadata and "interaction_id" in metadata:
                interaction_id = metadata["interaction_id"]

            # Create document (ID will be auto-generated by Milvus)
            doc = Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "interaction_id": interaction_id,
                    "title": title or "",
                    "original_id": conv_id,  # Store original ID in metadata
                    "metadata": json.dumps(metadata or {}),
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
            print(f"Failed to upsert embedding: {e}")
            return False

    async def delete_embeddings_by_interaction(self, interaction_id: str) -> bool:
        """Delete all embeddings for a specific interaction"""
        try:
            if not self.vector_store:
                print("Vector store not initialized, skipping embedding deletion")
                return False

            # Get the collection from the vector store
            collection_name = settings.ZILLIZ_COLLECTION
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

            print(f"✅ Deleted vector embeddings for interaction {interaction_id}")
            return True

        except Exception as e:
            print(
                f"⚠️ Failed to delete vector embeddings for interaction {interaction_id}: {e}"
            )
            return False

    def calculate_points_cost(self, tokens_used: int) -> int:
        """Calculate points cost based on tokens used"""
        return max(1, tokens_used // 100)

    async def generate_interaction_title(
        self, message: str, response_preview: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate interaction title and summary using cost-efficient model"""
        try:
            # Limit input sizes for cost efficiency
            limited_message = message[:200] if message else ""
            limited_response = response_preview[:300] if response_preview else ""

            # Create callback handler for minimal token tracking
            callback_handler = StudyGuruCallbackHandler()

            # Use the title generation chain
            from app.config.langchain_config import (
                StudyGuruModels,
                StudyGuruPrompts,
            )
            from langchain_core.output_parsers import JsonOutputParser

            # Create chain directly to avoid any import issues
            model = StudyGuruModels.get_title_model(temperature=0.3, max_tokens=100)
            parser = JsonOutputParser()
            title_chain = StudyGuruPrompts.TITLE_GENERATION | model | parser

            # Generate title
            result = await title_chain.ainvoke(
                {
                    "message": limited_message,
                    "response_preview": limited_response,
                },
                config={"callbacks": [callback_handler]},
            )

            # Extract title and summary
            title = result.get("title", "")[:50] if result.get("title") else None
            summary_title = (
                result.get("summary_title", "")[:100]
                if result.get("summary_title")
                else None
            )

            return title, summary_title

        except Exception as e:
            print(f"Title generation error: {e}")
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


# Global instance
langchain_service = LangChainService()
