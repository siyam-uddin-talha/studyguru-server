"""
LangGraph Integration Service
Connects LangGraph workflow with existing interaction system
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.langgraph_workflow_service import langgraph_workflow_service
from app.services.interaction import process_conversation_message
from app.models.user import User
from app.models.interaction import Interaction
from sqlalchemy.ext.asyncio import AsyncSession


class LangGraphIntegrationService:
    """Service to integrate LangGraph workflow with existing system"""

    def __init__(self):
        try:
            self.workflow_service = langgraph_workflow_service
            self.langgraph_available = True
        except ImportError as e:
            print(f"⚠️ LangGraph not available: {e}")
            self.workflow_service = None
            self.langgraph_available = False

    async def process_with_langgraph(
        self,
        user: User,
        interaction: Optional[Interaction],
        message: Optional[str],
        media_files: Optional[List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Process conversation with LangGraph workflow for complex multi-source tasks
        """

        # Check if LangGraph is available
        if not self.langgraph_available:
            # Fallback to standard processing if LangGraph is not available
            return await process_conversation_message(
                user=user,
                interaction=interaction,
                message=message,
                media_files=media_files,
                max_tokens=max_tokens,
                db=db,
            )

        # Analyze if LangGraph workflow is needed
        needs_langgraph = await self._should_use_langgraph(message, media_files)

        if not needs_langgraph:
            # Use existing simple processing
            return await process_conversation_message(
                user=user,
                interaction=interaction,
                message=message,
                media_files=media_files,
                max_tokens=max_tokens,
                db=db,
            )

        # Use LangGraph workflow for complex tasks
        try:
            # Prepare media files for workflow
            workflow_media_files = []
            if media_files:
                for media_file in media_files:
                    workflow_media_files.append(
                        {
                            "id": media_file.get("id", ""),
                            "url": media_file.get("url", ""),
                            "type": media_file.get("type", ""),
                            "name": media_file.get("name", ""),
                        }
                    )

            # Execute LangGraph workflow
            workflow_result = await self.workflow_service.execute_workflow(
                message=message or "",
                media_files=workflow_media_files,
                user_id=str(user.id),
                interaction_id=str(interaction.id) if interaction else "",
            )

            if workflow_result["success"]:
                # Process the result through existing system
                return await self._process_workflow_result(
                    workflow_result, user, interaction, db
                )
            else:
                # Fallback to simple processing on workflow failure
                return await process_conversation_message(
                    user=user,
                    interaction=interaction,
                    message=message,
                    media_files=media_files,
                    max_tokens=max_tokens,
                    db=db,
                )

        except Exception as e:
            # Fallback to simple processing on error
            return await process_conversation_message(
                user=user,
                interaction=interaction,
                message=message,
                media_files=media_files,
                max_tokens=max_tokens,
                db=db,
            )

    async def _should_use_langgraph(
        self, message: Optional[str], media_files: Optional[List[Dict[str, str]]]
    ) -> bool:
        """Determine if LangGraph workflow should be used"""

        # Check for complex indicators
        has_pdfs = any(
            file.get("type", "").lower() == "application/pdf"
            or file.get("url", "").lower().endswith(".pdf")
            or file.get("name", "").lower().endswith(".pdf")
            for file in (media_files or [])
        )
        has_links = bool(message and "http" in message)
        has_multiple_files = len(media_files or []) > 1

        # Check for analytical keywords
        analytical_keywords = [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "summarize",
            "research",
            "investigate",
            "examine",
            "assess",
            "review",
        ]
        has_analytical_keywords = any(
            keyword in (message or "").lower() for keyword in analytical_keywords
        )

        # Use LangGraph for complex scenarios
        return (
            (has_pdfs and has_links)  # PDFs + links
            or (has_pdfs and has_multiple_files)  # Multiple PDFs
            or (has_links and has_analytical_keywords)  # Links + analysis
            or (
                has_multiple_files and has_analytical_keywords
            )  # Multiple files + analysis
        )

    async def _process_workflow_result(
        self,
        workflow_result: Dict[str, Any],
        user: User,
        interaction: Optional[Interaction],
        db: Optional[AsyncSession],
    ) -> Dict[str, Any]:
        """Process LangGraph workflow result through existing system"""

        # Extract results
        ai_response = workflow_result.get("result", "")
        thinking_steps = workflow_result.get("thinking_steps", [])
        total_tokens = workflow_result.get("total_tokens", 0)

        # Create response in expected format
        response_data = {
            "success": True,
            "ai_response": ai_response,
            "thinking_steps": thinking_steps,
            "total_tokens": total_tokens,
            "workflow_type": "langgraph",
            "interaction_id": str(interaction.id) if interaction else None,
            "processing_metadata": {
                "workflow_state": workflow_result.get("workflow_state"),
                "context": workflow_result.get("context"),
                "timestamp": datetime.now().isoformat(),
            },
        }

        return response_data

    async def stream_workflow_with_thinking(
        self,
        user: User,
        interaction: Optional[Interaction],
        message: Optional[str],
        media_files: Optional[List[Dict[str, str]]],
        websocket=None,
    ):
        """Stream workflow execution with thinking steps"""

        try:
            # Check if LangGraph is available
            if not self.langgraph_available:
                # Fallback to standard streaming if LangGraph is not available
                from app.services.langchain_service import langchain_service

                async for (
                    chunk
                ) in langchain_service.generate_conversation_response_streaming(
                    message=message,
                    context="",
                    media_urls=[],
                    max_tokens=5000,
                ):
                    yield chunk
                return

            # Check if LangGraph is needed
            needs_langgraph = await self._should_use_langgraph(message, media_files)

            if not needs_langgraph:
                # Use existing streaming with media URLs
                from app.services.langchain_service import langchain_service

                # Extract media URLs from media_files
                media_urls = []
                if media_files:
                    for media_file in media_files:
                        if media_file.get("url"):
                            media_urls.append(media_file["url"])

                async for (
                    chunk
                ) in langchain_service.generate_conversation_response_streaming(
                    message=message,
                    context="",
                    media_urls=media_urls,
                    max_tokens=5000,
                ):
                    yield chunk
                return

            # Stream LangGraph workflow with error handling
            try:
                async for thinking_step in self._stream_workflow_thinking(
                    message, media_files, user, interaction, websocket
                ):
                    yield thinking_step
            except Exception as e:
                # Fallback to standard processing if LangGraph fails
                print(f"⚠️ LangGraph workflow failed: {e}")
                from app.services.langchain_service import langchain_service

                async for (
                    chunk
                ) in langchain_service.generate_conversation_response_streaming(
                    message=message,
                    context="",
                    media_urls=[],
                    max_tokens=5000,
                ):
                    yield chunk

        except Exception as e:
            yield {
                "type": "error",
                "content": f"Workflow execution failed: {str(e)}",
                "thinking_steps": ["❌ Workflow execution failed"],
            }

    async def _stream_workflow_thinking(
        self,
        message: Optional[str],
        media_files: Optional[List[Dict[str, str]]],
        user: User,
        interaction: Optional[Interaction],
        websocket=None,
    ):
        """Stream workflow execution with thinking steps"""

        # Prepare media files
        workflow_media_files = []
        if media_files:
            for media_file in media_files:
                workflow_media_files.append(
                    {
                        "id": media_file.get("id", ""),
                        "url": media_file.get("url", ""),
                        "type": media_file.get("type", ""),
                        "name": media_file.get("name", ""),
                    }
                )

        # Execute workflow with streaming
        try:
            # Send initial thinking step
            yield {
                "type": "thinking",
                "content": "🔍 Analyzing your request...",
                "thinking_steps": ["🔍 Analyzing your request..."],
            }

            # Execute workflow
            workflow_result = await self.workflow_service.execute_workflow(
                message=message or "",
                media_files=workflow_media_files,
                user_id=str(user.id),
                interaction_id=str(interaction.id) if interaction else "",
            )

            if workflow_result["success"]:
                # Stream thinking steps
                thinking_steps = workflow_result.get("thinking_steps", [])
                for step in thinking_steps:
                    yield {
                        "type": "thinking",
                        "content": step,
                        "thinking_steps": thinking_steps[
                            : thinking_steps.index(step) + 1
                        ],
                    }
                    await asyncio.sleep(0.5)  # Small delay for better UX

                # Stream final result
                yield {
                    "type": "response",
                    "content": workflow_result.get("result", ""),
                    "thinking_steps": thinking_steps,
                    "total_tokens": workflow_result.get("total_tokens", 0),
                    "workflow_type": "langgraph",
                }
            else:
                # Handle workflow failure - fallback to standard processing
                error_message = workflow_result.get("error", "Unknown error")
                print(f"⚠️ LangGraph workflow failed: {error_message}")

                # Fallback to standard processing with media URLs
                from app.services.langchain_service import langchain_service

                # Extract media URLs from media_files
                media_urls = []
                if media_files:
                    for media_file in media_files:
                        if media_file.get("url"):
                            media_urls.append(media_file["url"])

                async for (
                    chunk
                ) in langchain_service.generate_conversation_response_streaming(
                    message=message,
                    context="",
                    media_urls=media_urls,
                    max_tokens=5000,
                ):
                    yield chunk

        except Exception as e:
            print(f"⚠️ LangGraph workflow exception: {str(e)}")

            # Fallback to standard processing with media URLs
            from app.services.langchain_service import langchain_service

            # Extract media URLs from media_files
            media_urls = []
            if media_files:
                for media_file in media_files:
                    if media_file.get("url"):
                        media_urls.append(media_file["url"])

            async for (
                chunk
            ) in langchain_service.generate_conversation_response_streaming(
                message=message,
                context="",
                media_urls=media_urls,
                max_tokens=5000,
            ):
                yield chunk


# Global instance
langgraph_integration_service = LangGraphIntegrationService()
