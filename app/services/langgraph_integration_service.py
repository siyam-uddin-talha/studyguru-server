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
        visualize_model: Optional[str] = None,
        assistant_model: Optional[str] = None,
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
                visualize_model=visualize_model,
                assistant_model=assistant_model,
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
                visualize_model=visualize_model,
                assistant_model=assistant_model,
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
                    visualize_model=visualize_model,
                    assistant_model=assistant_model,
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
                visualize_model=visualize_model,
                assistant_model=assistant_model,
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

        # Check for web search keywords - use LangGraph for web search requests
        web_search_keywords = [
            "search the web",
            "search the internet",
            "find information about",
            "look up",
            "web search",
            "latest",
            "current",
            "recent",
            "2024",
            "2025",
            "news",
            "discoveries",
            "announcements",
            "press release",
        ]
        has_web_search_keywords = any(
            keyword in (message or "").lower() for keyword in web_search_keywords
        )

        # Debug logging
        print(
            f"🔍 [LANGGRAPH DECISION] has_pdfs={has_pdfs}, has_links={has_links}, has_multiple_files={has_multiple_files}, has_analytical_keywords={has_analytical_keywords}, has_web_search_keywords={has_web_search_keywords}"
        )

        # Use LangGraph for complex scenarios
        # URLs always need LangGraph for web scraping/searching
        # Web search requests should use LangGraph to use Serper directly
        should_use = (
            has_links  # URLs need web scraping - always use LangGraph
            or has_web_search_keywords  # Web search requests - use LangGraph with Serper
            or (has_pdfs and has_links)  # PDFs + links
            or (has_pdfs and has_multiple_files)  # Multiple PDFs
            or (has_links and has_analytical_keywords)  # Links + analysis
            or (
                has_multiple_files and has_analytical_keywords
            )  # Multiple files + analysis
        )

        print(f"🔍 [LANGGRAPH DECISION] Should use LangGraph: {should_use}")
        return should_use

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
        visualize_model: Optional[str] = None,
        assistant_model: Optional[str] = None,
        context: str = "",
    ):
        """Stream workflow execution with thinking steps"""

        # Get user's subscription plan
        subscription_plan = None
        if user.purchased_subscription:
            subscription_plan = (
                user.purchased_subscription.subscription.subscription_plan
            )

        # Log model selection
        print(f"🔄 [LANGGRAPH STREAMING] Starting workflow with models:")
        print(f"   👁️  Visualize Model: {visualize_model or 'default (auto-select)'}")
        print(f"   💬 Assistant Model: {assistant_model or 'default (auto-select)'}")
        print(f"   🔐 Subscription Plan: {subscription_plan or 'none'}")
        print(f"   📝 Context Length: {len(context) if context else 0} chars")

        try:
            # Check if LangGraph is available
            if not self.langgraph_available:
                # Fallback to standard streaming if LangGraph is not available
                print(
                    "⚠️ [LANGGRAPH STREAMING] LangGraph not available, using fallback streaming"
                )
                from app.services.langchain_service import langchain_service

                async for (
                    chunk
                ) in langchain_service.generate_conversation_response_streaming(
                    message=message,
                    context=context,
                    media_urls=[],
                    max_tokens=5000,
                    visualize_model=visualize_model,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
                ):
                    yield chunk
                return

            # Check if LangGraph is needed
            needs_langgraph = await self._should_use_langgraph(message, media_files)

            if not needs_langgraph:
                # Use existing streaming with media URLs
                print(
                    "✅ [LANGGRAPH STREAMING] Using standard streaming (LangGraph not needed)"
                )
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
                    context=context,
                    media_urls=media_urls,
                    max_tokens=5000,
                    visualize_model=visualize_model,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
                ):
                    yield chunk
                return

            # Stream LangGraph workflow with error handling
            print("🚀 [LANGGRAPH STREAMING] Using LangGraph workflow for complex task")
            try:
                async for thinking_step in self._stream_workflow_thinking(
                    message,
                    media_files,
                    user,
                    interaction,
                    websocket,
                    visualize_model,
                    assistant_model,
                    subscription_plan,
                    context,
                ):
                    yield thinking_step
            except Exception as e:
                # Fallback to standard processing if LangGraph fails
                print(f"⚠️ [LANGGRAPH STREAMING] LangGraph workflow failed: {e}")
                print(
                    f"🔄 [LANGGRAPH STREAMING] Falling back to standard streaming with models:"
                )
                print(
                    f"   👁️  Visualize Model: {visualize_model or 'default (auto-select)'}"
                )
                print(
                    f"   💬 Assistant Model: {assistant_model or 'default (auto-select)'}"
                )
                from app.services.langchain_service import langchain_service

                async for (
                    chunk
                ) in langchain_service.generate_conversation_response_streaming(
                    message=message,
                    context=context,
                    media_urls=[],
                    max_tokens=5000,
                    visualize_model=visualize_model,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
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
        visualize_model: Optional[str] = None,
        assistant_model: Optional[str] = None,
        subscription_plan: Optional[str] = None,
        context: str = "",
    ):
        """Stream workflow execution with thinking steps"""

        # Import thinking status helper
        from app.api.interaction_routes import send_thinking_status, THINKING_STATUSES

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
            # Send initial thinking step via websocket if available
            if websocket:
                # Stop any previous statuses
                if hasattr(websocket, "_progressive_thinking"):
                    await websocket._progressive_thinking.stop_all()
                await send_thinking_status(websocket, "preparing_response")

            # Also yield thinking step for compatibility
            yield {
                "type": "thinking",
                "content": "🔍 Analyzing your request...",
                "thinking_steps": ["🔍 Analyzing your request..."],
            }

            # Execute workflow with thinking status updates
            # Note: workflow_service currently manages its own context retrieval
            # We might want to pass the context here in the future if supported

            # Start workflow execution in background and monitor progress
            workflow_task = asyncio.create_task(
                self.workflow_service.execute_workflow(
                    message=message or "",
                    media_files=workflow_media_files,
                    user_id=str(user.id),
                    interaction_id=str(interaction.id) if interaction else "",
                )
            )

            # Monitor workflow progress and send thinking statuses
            workflow_start_time = asyncio.get_event_loop().time()

            # Send searching_web status if workflow takes longer (indicates web search)
            search_status_sent = False

            # Wait for workflow with periodic status updates
            while not workflow_task.done():
                await asyncio.sleep(0.5)  # Check every 0.5 seconds

                elapsed = asyncio.get_event_loop().time() - workflow_start_time

                # If workflow is taking longer, send appropriate thinking statuses
                if websocket:
                    if elapsed > 3.0 and not search_status_sent:
                        # Likely doing web search - check if message contains URLs or web search keywords
                        message_lower = (message or "").lower()
                        has_urls = "http" in message_lower or "www." in message_lower
                        if has_urls:
                            # Stop previous status and send searching_web status
                            if hasattr(websocket, "_progressive_thinking"):
                                await websocket._progressive_thinking.stop_status(
                                    "preparing_response"
                                )
                            await send_thinking_status(websocket, "searching_web")
                            search_status_sent = True
                            yield {
                                "type": "thinking",
                                "content": "Searching the web for current information...",
                                "thinking_steps": [
                                    "Searching the web for current information..."
                                ],
                            }

            # Get workflow result
            workflow_result = await workflow_task

            if workflow_result["success"]:
                # Stop previous thinking statuses
                if websocket and hasattr(websocket, "_progressive_thinking"):
                    await websocket._progressive_thinking.stop_status(
                        "preparing_response"
                    )
                    await websocket._progressive_thinking.stop_status("searching_web")

                # Send generating status for final response generation
                if websocket:
                    await send_thinking_status(websocket, "generating")

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

                # Get context from workflow result for streaming
                context = workflow_result.get("context")
                if context:
                    # Stream final summary using model's native streaming (like generate_conversation_response_streaming)
                    print(
                        f"🚀 [LANGGRAPH STREAMING] Streaming final summary from model..."
                    )
                    try:
                        async for (
                            chunk
                        ) in self.workflow_service._generate_final_summary_streaming(
                            context
                        ):
                            yield chunk
                        print(f"✅ [LANGGRAPH STREAMING] Streaming complete")
                    except Exception as stream_error:
                        print(
                            f"⚠️ [LANGGRAPH STREAMING] Streaming error: {stream_error}"
                        )
                        # Fallback to result text streaming
                        final_content = workflow_result.get("result", "")
                        if final_content:
                            chunk_size = 100
                            for i in range(0, len(final_content), chunk_size):
                                chunk = final_content[i : i + chunk_size]
                                yield chunk
                                await asyncio.sleep(0.01)
                else:
                    # Fallback: stream final result as token chunks if context not available
                    final_content = workflow_result.get("result", "")
                    print(
                        f"📝 [LANGGRAPH STREAMING] Streaming final result (fallback): {len(final_content)} chars"
                    )
                    chunk_size = 100
                    for i in range(0, len(final_content), chunk_size):
                        chunk = final_content[i : i + chunk_size]
                        yield chunk
                        await asyncio.sleep(0.01)  # Small delay to simulate streaming
                    print(f"✅ [LANGGRAPH STREAMING] Streaming complete")
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
                    context=context,
                    media_urls=media_urls,
                    max_tokens=5000,
                    visualize_model=visualize_model,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
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
                context=context,
                media_urls=media_urls,
                max_tokens=5000,
                visualize_model=visualize_model,
                assistant_model=assistant_model,
                subscription_plan=subscription_plan,
            ):
                yield chunk


# Global instance
langgraph_integration_service = LangGraphIntegrationService()
