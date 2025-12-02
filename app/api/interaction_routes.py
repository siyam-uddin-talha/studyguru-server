"""
WebSocket Streaming API routes for StudyGuru interactions
"""

import asyncio
import json
from typing import Optional, List, Dict, Any
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    Header,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.helpers.auth import verify_token
from app.helpers.websocket_auth import get_current_user_from_websocket
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.models.subscription import PurchasedSubscription

from app.services.langchain_service import langchain_service
from app.services.langgraph_integration_service import langgraph_integration_service
from app.config.langchain_config import StudyGuruConfig


# Router
interaction_router = APIRouter()


# Progressive thinking status manager
class ProgressiveThinkingStatus:
    """Manages progressive thinking status updates with automatic message progression"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.active_statuses: Dict[str, asyncio.Task] = {}
        self.status_start_times: Dict[str, float] = {}
        self.status_message_indices: Dict[str, int] = {}

    async def start_progressive_status(
        self,
        status_type: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Start a progressive thinking status that auto-advances messages"""
        # Cancel any existing status of the same type
        if status_type in self.active_statuses:
            self.active_statuses[status_type].cancel()

        # Get messages for this status type
        status_config = THINKING_STATUSES.get(status_type)
        if not status_config:
            print(f"⚠️ Unknown thinking status type: {status_type}")
            return

        messages = status_config.get("messages", [])
        if not messages:
            print(f"⚠️ No messages configured for status type: {status_type}")
            return

        # Initialize state
        self.status_start_times[status_type] = asyncio.get_event_loop().time()
        self.status_message_indices[status_type] = 0

        # Send first message immediately
        await self._send_status_message(status_type, messages[0], details)

        # Start background task for progressive updates
        self.active_statuses[status_type] = asyncio.create_task(
            self._progressive_update_loop(status_type, messages, details)
        )

    async def _progressive_update_loop(
        self,
        status_type: str,
        messages: List[str],
        details: Optional[Dict[str, Any]],
    ):
        """Background loop that progresses through messages every 2 seconds"""
        try:
            while True:
                await asyncio.sleep(2.0)  # Wait 2 seconds before next message

                # Check if status is still active
                if status_type not in self.status_start_times:
                    break

                # Get current message index
                current_index = self.status_message_indices.get(status_type, 0)

                # Move to next message if available
                if current_index < len(messages) - 1:
                    next_index = current_index + 1
                    self.status_message_indices[status_type] = next_index
                    await self._send_status_message(
                        status_type, messages[next_index], details
                    )
                else:
                    # Reached last message, keep showing it
                    break
        except asyncio.CancelledError:
            # Status was cancelled (replaced by new status)
            pass
        except Exception as e:
            print(f"⚠️ Progressive status update error: {e}")

    async def _send_status_message(
        self,
        status_type: str,
        message: str,
        details: Optional[Dict[str, Any]],
    ):
        """Send a single thinking status message"""
        try:
            status_data = {
                "type": "thinking",
                "status_type": status_type,
                "message": message,
                "timestamp": asyncio.get_event_loop().time(),
            }
            if details:
                status_data["details"] = details
            else:
                # Get default details from THINKING_STATUSES
                status_config = THINKING_STATUSES.get(status_type)
                if status_config:
                    status_data["details"] = status_config.get("details", {})

            await self.websocket.send_text(json.dumps(status_data))
            print(f"🧠 Sent thinking status: {status_type} - {message}")
        except Exception as e:
            print(f"⚠️ Failed to send thinking status: {e}")

    async def stop_status(self, status_type: str):
        """Stop a progressive thinking status"""
        if status_type in self.active_statuses:
            self.active_statuses[status_type].cancel()
            del self.active_statuses[status_type]
        if status_type in self.status_start_times:
            del self.status_start_times[status_type]
        if status_type in self.status_message_indices:
            del self.status_message_indices[status_type]

    async def stop_all(self):
        """Stop all active progressive thinking statuses"""
        for status_type in list(self.active_statuses.keys()):
            await self.stop_status(status_type)


# Helper function to send thinking status updates (backward compatible)
async def send_thinking_status(
    websocket: WebSocket,
    status_type: str,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    progressive: bool = True,
):
    """
    Send thinking status update to WebSocket client

    Args:
        websocket: WebSocket connection
        status_type: Type of thinking status
        message: Optional message (if None, uses first message from THINKING_STATUSES)
        details: Optional details dictionary
        progressive: If True, uses progressive status updates (default: True)
    """
    try:
        if progressive:
            # Use progressive status manager
            if not hasattr(websocket, "_progressive_thinking"):
                websocket._progressive_thinking = ProgressiveThinkingStatus(websocket)

            await websocket._progressive_thinking.start_progressive_status(
                status_type, details
            )
        else:
            # Legacy single message mode
            if message is None:
                status_config = THINKING_STATUSES.get(status_type)
                if status_config:
                    messages = status_config.get("messages", [])
                    message = messages[0] if messages else f"{status_type}..."
                else:
                    message = f"{status_type}..."

            status_data = {
                "type": "thinking",
                "status_type": status_type,
                "message": message,
                "timestamp": asyncio.get_event_loop().time(),
            }
            if details:
                status_data["details"] = details
            else:
                status_config = THINKING_STATUSES.get(status_type)
                if status_config:
                    status_data["details"] = status_config.get("details", {})

            await websocket.send_text(json.dumps(status_data))
            print(f"🧠 Sent thinking status: {status_type} - {message}")
    except Exception as e:
        print(f"⚠️ Failed to send thinking status: {e}")


# Background function for non-blocking title generation
async def _update_title_background(
    interaction_id: str,
    content_text: str,
    original_message: str,
    assistant_model: Optional[str] = None,
    subscription_plan: Optional[str] = None,
    websocket: Optional[WebSocket] = None,
):
    """
    Update interaction title in background without blocking response.
    If websocket is provided, sends the generated title through websocket.
    """
    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import select
        from app.models.interaction import Interaction
        from app.services.interaction import _extract_interaction_metadata_fast

        # Create a new database session for the background task
        async with AsyncSessionLocal() as db:
            # Fetch the interaction with a fresh session
            result = await db.execute(
                select(Interaction).where(Interaction.id == interaction_id)
            )
            interaction = result.scalar_one_or_none()

            if interaction:
                # Store original title to check if it changed
                old_title = interaction.title
                old_summary_title = interaction.summary_title

                await _extract_interaction_metadata_fast(
                    interaction=interaction,
                    content_text=content_text,
                    original_message=original_message,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
                )
                await db.commit()

                # Refresh to get updated values
                await db.refresh(interaction)

                # Send title through websocket if provided and title was generated/updated
                if websocket and interaction.title and interaction.title != old_title:
                    try:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "title_update",
                                    "title": interaction.title,
                                    "summary_title": interaction.summary_title,
                                    "interaction_id": interaction_id,
                                    "timestamp": asyncio.get_event_loop().time(),
                                }
                            )
                        )
                        print(f"✅ Title sent through websocket: '{interaction.title}'")
                    except Exception as ws_error:
                        print(f"⚠️ Failed to send title through websocket: {ws_error}")

                print(
                    f"✅ Background title generation completed and saved: '{interaction.title}'"
                )
            else:
                print(f"⚠️ Interaction {interaction_id} not found for title generation")
    except Exception as title_error:
        print(f"⚠️ Background title generation failed: {title_error}")
        import traceback

        traceback.print_exc()


# Thinking status types and messages with progressive updates
THINKING_STATUSES = {
    "analyzing": {
        "messages": [
            "Analyzing your question...",
            "Understanding the context and requirements...",
            "Breaking down the key components...",
        ],
        "details": {"stage": "input_analysis"},
    },
    "searching_context": {
        "messages": [
            "Searching through your previous conversations for context...",
            "Reviewing relevant past discussions...",
            "Gathering contextual information...",
        ],
        "details": {"stage": "context_retrieval"},
    },
    "checking_guardrails": {
        "messages": [
            "Checking content safety...",
            "Verifying compliance with guidelines...",
            "Ensuring appropriate content...",
        ],
        "details": {"stage": "safety_check"},
    },
    "preparing_response": {
        "messages": [
            "Preparing my response...",
            "Structuring the information...",
            "Organizing key points...",
            "Finalizing the details...",
            "Almost ready...",
        ],
        "details": {"stage": "response_preparation"},
    },
    "searching_web": {
        "messages": [
            "Searching the web for current information...",
            "Gathering up-to-date sources...",
            "Verifying information accuracy...",
            "Compiling relevant findings...",
        ],
        "details": {"stage": "web_search"},
    },
    "generating": {
        "messages": [
            "Generating response...",
            "Crafting detailed answer...",
            "Refining the content...",
            "Polishing the response...",
            "Finalizing output...",
        ],
        "details": {"stage": "ai_generation"},
    },
    "processing_media": {
        "messages": [
            "Processing your uploaded files...",
            "Analyzing file contents...",
            "Extracting relevant information...",
        ],
        "details": {"stage": "media_processing"},
    },
    # "saving": {
    #     "messages": [
    #         "Saving our conversation...",
    #         "Storing the interaction...",
    #         "Updating records...",
    #     ],
    #     "details": {"stage": "database_save"},
    # },
}


# Pydantic models for request/response
class StreamConversationRequest(BaseModel):
    interaction_id: Optional[str] = None
    message: Optional[str] = ""
    media_files: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = None


# Custom dependency to extract and verify user from token
async def get_user_from_token(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and verify user from authorization token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    try:
        # Split the authorization header
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    # Verify the token
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Get user ID from token
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Fetch user from database with eager loading of purchased_subscription
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.purchased_subscription).selectinload(
                PurchasedSubscription.subscription
            )
        )
        .where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@interaction_router.get("/test-thinking-status")
async def test_thinking_status():
    """Test endpoint to verify thinking status functionality"""
    return {
        "status": "success",
        "message": "Thinking status system is active",
        "available_statuses": list(THINKING_STATUSES.keys()),
        "timestamp": asyncio.get_event_loop().time(),
    }


@interaction_router.get("/debug-jwt")
async def debug_jwt():
    """Debug endpoint to check JWT configuration"""
    from app.core.config import settings

    return {
        "status": "success",
        "jwt_secret_key_length": len(settings.JWT_SECRET_KEY),
        "jwt_secret_key_preview": (
            settings.JWT_SECRET_KEY[:10] + "..."
            if len(settings.JWT_SECRET_KEY) > 10
            else settings.JWT_SECRET_KEY
        ),
        "timestamp": asyncio.get_event_loop().time(),
    }


@interaction_router.websocket("/stream-conversation")
async def websocket_stream_conversation(websocket: WebSocket):
    """
    WebSocket endpoint for full conversation streaming

    This endpoint provides the same functionality as the do_conversation mutation
    but streams the AI response in real-time using WebSocket.
    """

    try:
        # Authenticate user first

        user_id = await get_current_user_from_websocket(websocket)

        await websocket.accept()

    except Exception as e:

        raise

    try:
        # Get database session
        db_gen = get_db()
        db = await db_gen.__anext__()

        try:
            # Get user from database with eager loading of purchased_subscription
            result = await db.execute(
                select(User)
                .options(
                    selectinload(User.purchased_subscription).selectinload(
                        PurchasedSubscription.subscription
                    )
                )
                .where(User.id == user_id)
            )
            current_user = result.scalar_one_or_none()

            if not current_user:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "error": "User not found",
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )
                )
                await websocket.close()
                return

            # Receive the initial message
            data = await websocket.receive_text()
            payload = json.loads(data)

            message = payload.get("message", "")
            interaction_id = payload.get("interaction_id")
            media_files = payload.get("media_files", [])
            max_tokens = payload.get("max_tokens", 5000)
            visualize_model = payload.get(
                "visualize_model"
            )  # Model for document analysis
            assistant_model = payload.get(
                "assistant_model"
            )  # Model for text conversation

            # Extract custom context/mindmap from payload if provided
            custom_context_data = payload.get("context") or payload.get("mindmap")
            custom_context_str = ""
            if custom_context_data:
                print(f"🔍 Found custom context in payload")
                if isinstance(custom_context_data, dict):
                    # Check if it's a mindmap
                    if custom_context_data.get("type") == "mindmap":
                        try:
                            # Format mindmap context nicely
                            mindmap_result = custom_context_data.get("_result", {})
                            nodes = mindmap_result.get("nodes", {})
                            topic = mindmap_result.get("topic", "Mindmap")

                            # Helper to format node tree
                            def format_node_tree(node, depth=0):
                                indent = "  " * depth
                                content = node.get("content", "")
                                text = f"{indent}- {content}\n"
                                children = node.get("children", [])
                                for child in children:
                                    text += format_node_tree(child, depth + 1)
                                return text

                            custom_context_str = (
                                f"**CURRENT MINDMAP CONTEXT ({topic}):**\n"
                            )
                            if (
                                isinstance(nodes, dict) and "id" in nodes
                            ):  # Root node structure
                                custom_context_str += format_node_tree(nodes)
                            elif isinstance(nodes, list):  # Flat list or list of roots
                                for node in nodes:
                                    custom_context_str += format_node_tree(node)

                            print(
                                f"🔍 Formatted mindmap context: {len(custom_context_str)} chars"
                            )
                        except Exception as e:
                            print(f"⚠️ Error formatting mindmap context: {e}")
                            custom_context_str = f"**PROVIDED CONTEXT:**\n{json.dumps(custom_context_data)}"
                    else:
                        custom_context_str = (
                            f"**PROVIDED CONTEXT:**\n{json.dumps(custom_context_data)}"
                        )
                elif isinstance(custom_context_data, str):
                    custom_context_str = f"**PROVIDED CONTEXT:**\n{custom_context_data}"

            # Log model selection for streaming
            print(f"📊 [STREAMING] Model Selection:")
            print(
                f"   👁️  Visualize Model: {visualize_model or 'default (auto-select)'}"
            )
            print(
                f"   💬 Assistant Model: {assistant_model or 'default (auto-select)'}"
            )

            # Validate input
            if not message or not message.strip():
                if not media_files or len(media_files) == 0:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "error": "Please provide a message or attach a file",
                                "timestamp": asyncio.get_event_loop().time(),
                            }
                        )
                    )
                    await websocket.close()
                    return

            # Send initial thinking status
            await send_thinking_status(websocket, "analyzing")

            # Handle interaction setup (same logic as do_conversation)
            interaction = None
            is_fresh_interaction = False

            if interaction_id:
                result = await db.execute(
                    select(Interaction).where(
                        Interaction.id == interaction_id,
                        Interaction.user_id == current_user.id,
                    )
                )
                interaction = result.scalar_one_or_none()
                if not interaction:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "error": "No conversation found!",
                                "timestamp": asyncio.get_event_loop().time(),
                            }
                        )
                    )
                    await websocket.close()
                    return

                # Check if this is a fresh interaction
                conv_result = await db.execute(
                    select(Conversation).where(
                        Conversation.interaction_id == interaction.id
                    )
                )
                existing_conversations = conv_result.scalars().all()
                is_fresh_interaction = len(existing_conversations) == 0
            else:
                interaction = Interaction(
                    user_id=str(current_user.id),
                    title=None,
                    summary_title=None,
                )
                db.add(interaction)
                await db.commit()
                is_fresh_interaction = True

            # Send initial metadata
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "metadata",
                        "interaction_id": str(interaction.id),
                        "is_new_interaction": is_fresh_interaction,
                        "user_id": str(current_user.id),
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )
            )

            # For WebSocket streaming, we need to do the conversation processing
            # but skip the AI response generation part and do it with streaming instead

            # First, let's do the conversation setup and context retrieval
            # (similar to process_conversation_message but without the AI generation)

            # Process media files if present (optimized with single query)
            media_objects = []
            if media_files:
                # Stop previous status and send thinking status for media processing
                if hasattr(websocket, "_progressive_thinking"):
                    await websocket._progressive_thinking.stop_status("analyzing")
                await send_thinking_status(websocket, "processing_media")

                # Extract all media IDs
                media_ids = [
                    media_file.get("id")
                    for media_file in media_files
                    if media_file.get("id")
                ]

                # Single optimized query for all media files
                if media_ids:
                    result = await db.execute(
                        select(Media).where(Media.id.in_(media_ids))
                    )
                    media_objects = result.scalars().all()

            # Create user conversation entry
            user_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.USER,
                content={
                    "type": "text",
                    "_result": {
                        "note": message or "",
                        "media_ids": [media.id for media in media_objects],
                    },
                },
                status="completed",
            )
            db.add(user_conv)
            await db.flush()

            # Associate media files with the conversation using the junction table
            if media_objects:
                from app.models.interaction import conversation_files

                media_tasks = [
                    db.execute(
                        conversation_files.insert().values(
                            conversation_id=str(user_conv.id),
                            media_id=str(media_obj.id),
                        )
                    )
                    for media_obj in media_objects
                ]
                await asyncio.gather(*media_tasks, return_exceptions=True)

            # OPTIMIZATION: Run guardrails and context retrieval in parallel
            context_text = ""
            guardrail_result = None

            # Create parallel tasks
            async def run_guardrails_async():
                try:
                    # Stop previous statuses
                    if hasattr(websocket, "_progressive_thinking"):
                        await websocket._progressive_thinking.stop_status("analyzing")
                        await websocket._progressive_thinking.stop_status(
                            "processing_media"
                        )
                    await send_thinking_status(websocket, "checking_guardrails")
                    media_urls = []
                    if media_files:
                        for media_file in media_files:
                            if media_file.get("url"):
                                media_urls.append(media_file["url"])
                    return await langchain_service.check_guardrails(
                        message=message or "", media_urls=media_urls
                    )
                except Exception as e:
                    print(f"⚠️ Guardrail check failed: {e}")
                    return None

            async def get_context_async():
                if is_fresh_interaction:
                    print(
                        f"🔍 [CONTEXT] Fresh interaction - skipping context retrieval"
                    )
                    return ""
                try:
                    print(
                        f"🔍 [CONTEXT] Starting context retrieval for interaction: {interaction.id}"
                    )
                    print(f"🔍 [CONTEXT] Query: '{message or ''}'")
                    print(f"🔍 [CONTEXT] User ID: {current_user.id}")

                    # Stop previous statuses
                    if hasattr(websocket, "_progressive_thinking"):
                        await websocket._progressive_thinking.stop_status("analyzing")
                        await websocket._progressive_thinking.stop_status(
                            "processing_media"
                        )
                    await send_thinking_status(websocket, "searching_context")
                    if interaction_id:
                        try:
                            # OPTIMIZATION: Increased timeout for reliability
                            print(
                                f"🔍 [CONTEXT] Calling similarity_search_by_interaction..."
                            )
                            search_results = await asyncio.wait_for(
                                langchain_service.similarity_search_by_interaction(
                                    query=message or "",
                                    user_id=str(current_user.id),
                                    interaction_id=str(interaction.id),
                                    top_k=2,  # Reduced for speed
                                ),
                                timeout=3.0,  # Increased from 1.5 to 3.0 seconds
                            )
                            print(
                                f"🔍 [CONTEXT] Search completed. Results count: {len(search_results) if search_results else 0}"
                            )

                            if search_results:
                                print(f"🔍 [CONTEXT] Raw search results:")
                                for i, result in enumerate(search_results):
                                    print(
                                        f"   [{i+1}] Title: {result.get('title', 'N/A')}"
                                    )
                                    print(
                                        f"   [{i+1}] Content preview: {result.get('content', '')[:100]}..."
                                    )
                                    print(
                                        f"   [{i+1}] Score: {result.get('score', 'N/A')}"
                                    )

                                context_parts = []
                                max_length = (
                                    8000  # Increased from 5000 to 8000 for more context
                                )
                                current_length = 0
                                for result in search_results:
                                    content = result.get("content", "")
                                    if current_length + len(content) > max_length:
                                        remaining = max_length - current_length
                                        if remaining > 100:
                                            content = content[:remaining] + "..."
                                            context_parts.append(
                                                f"**{result.get('title', 'Previous Discussion')}:**\n{content}"
                                            )
                                        break
                                    context_parts.append(
                                        f"**{result.get('title', 'Previous Discussion')}:**\n{content}"
                                    )
                                    current_length += len(content)

                                final_context = "\n\n".join(context_parts)
                                print(
                                    f"🔍 [CONTEXT] Final context length: {len(final_context)} characters"
                                )
                                print(
                                    f"🔍 [CONTEXT] Final context preview: {final_context[:200]}..."
                                )
                                return final_context
                            else:
                                print(f"🔍 [CONTEXT] No search results found")
                                return ""
                        except asyncio.TimeoutError:
                            print(
                                f"⏱️ [CONTEXT] Context search timed out - continuing without context"
                            )
                            return ""
                    else:
                        print(f"🔍 [CONTEXT] No interaction_id provided")
                        return ""
                except Exception as e:
                    print(f"⚠️ [CONTEXT] Context retrieval failed: {e}")
                    import traceback

                    traceback.print_exc()
                    return ""

            # Run in parallel with timeout
            guardrail_task = asyncio.create_task(run_guardrails_async())
            context_task = asyncio.create_task(get_context_async())

            # Wait for both with increased timeout
            try:
                guardrail_result, context_text = await asyncio.wait_for(
                    asyncio.gather(
                        guardrail_task, context_task, return_exceptions=True
                    ),
                    timeout=4.0,  # Increased from 2.0 to 4.0 seconds for reliability
                )

                # Debug: Print context retrieval result
                if isinstance(context_text, Exception):
                    print(
                        f"🔍 [CONTEXT] Context retrieval returned exception: {context_text}"
                    )
                    context_text = ""
                else:
                    print(
                        f"🔍 [CONTEXT] Context retrieval result: {len(context_text) if context_text else 0} characters"
                    )
                    if context_text:
                        print(f"🔍 [CONTEXT] Context will be used in response")
                    else:
                        print(
                            f"🔍 [CONTEXT] No context available - proceeding without context"
                        )

            except asyncio.TimeoutError:
                print(
                    f"⏱️ [CONTEXT] Guardrails/context retrieval timed out - continuing"
                )
                if not guardrail_task.done():
                    guardrail_result = None
                if not context_task.done():
                    context_text = ""
                    print(f"🔍 [CONTEXT] Context task timed out - no context available")

            # Handle guardrail result
            if guardrail_result and not isinstance(guardrail_result, Exception):
                if guardrail_result.is_violation:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "error": f"Content policy violation: {guardrail_result.reasoning}",
                                "timestamp": asyncio.get_event_loop().time(),
                            }
                        )
                    )
                    await websocket.close()
                    return

            # === DETECT SPECIAL REQUEST TYPES (AFTER GUARDRAILS/CONTEXT) ===
            from app.services.interaction import (
                detect_mindmap_request,
                extract_topic_from_message,
                serialize_mindmap_tree,
            )

            # Extract image URLs
            image_urls = []
            if media_files:
                for media_file in media_files:
                    if media_file.get("url"):
                        image_urls.append(media_file["url"])

            is_mindmap_request = detect_mindmap_request(message)

            print(f"🔍 WebSocket Request type detection (after guardrails/context):")
            print(f"   Mindmap request: {is_mindmap_request}")
            print(
                f"   Context available: {len(context_text) if context_text else 0} chars"
            )

            # Handle Mindmap Generation (with context)
            if is_mindmap_request:
                print(f"🗺️ WEBSOCKET MINDMAP GENERATOR MODE ACTIVATED (with context)")
                try:
                    # Stop previous statuses and send thinking status
                    if hasattr(websocket, "_progressive_thinking"):
                        await websocket._progressive_thinking.stop_all()
                    await send_thinking_status(websocket, "generating")

                    # Extract topic
                    topic = extract_topic_from_message(message)

                    # Generate mindmap WITH CONTEXT
                    mindmap = await langchain_service.generate_mindmap(
                        topic=topic,
                        context=context_text,  # NEW: Pass context
                        max_tokens=max_tokens,
                        assistant_model=assistant_model,
                        subscription_plan=(
                            current_user.purchased_subscription.subscription.subscription_plan
                            if current_user.purchased_subscription
                            else None
                        ),
                    )

                    # Prepare content
                    ai_content = {
                        "type": "mindmap",
                        "_result": {
                            "topic": mindmap.topic,
                            "nodes": serialize_mindmap_tree(mindmap.root_node),
                            "total_nodes": mindmap.total_nodes,
                        },
                    }

                    # Save to DB
                    ai_conv = Conversation(
                        interaction_id=str(interaction.id),
                        role=ConversationRole.AI,
                        content=ai_content,
                        question_type="mindmap",
                        status="completed",
                    )
                    db.add(ai_conv)
                    await db.flush()
                    await db.commit()

                    # Send response
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "completed",
                                "content": json.dumps(ai_content),
                                "timestamp": asyncio.get_event_loop().time(),
                            }
                        )
                    )

                    print(f"✅ Mindmap generated and sent via WebSocket")

                    # NEW: Background title generation
                    asyncio.create_task(
                        _update_title_background(
                            interaction_id=str(interaction.id),
                            content_text=f"Mindmap: {topic}",
                            original_message=message,
                            assistant_model=assistant_model,
                            subscription_plan=(
                                current_user.purchased_subscription.subscription.subscription_plan
                                if current_user.purchased_subscription
                                else None
                            ),
                        )
                    )

                    # NEW: Background vector DB storage
                    # Format mindmap as readable text for embedding
                    def format_mindmap_for_embedding(node, depth=0):
                        indent = "  " * depth
                        text = f"{indent}- {node.content}\n"
                        for child in node.children:
                            text += format_mindmap_for_embedding(child, depth + 1)
                        return text

                    formatted_text = f"Mindmap: {topic}\n\n{format_mindmap_for_embedding(mindmap.root_node)}"

                    asyncio.create_task(
                        langchain_service.upsert_embedding(
                            conv_id=str(ai_conv.id),
                            user_id=str(current_user.id),
                            text=formatted_text,
                            title=f"Mindmap: {topic}",
                            metadata={
                                "interaction_id": str(interaction.id),
                                "type": "mindmap",
                                "total_nodes": mindmap.total_nodes,
                            },
                        )
                    )

                    return

                except Exception as e:
                    print(f"❌ Mindmap generation failed: {e}")
                    import traceback

                    traceback.print_exc()
                    # Fall through to normal streaming

            # Generate streaming response using LangChain
            full_response = ""
            print(f"🚀 Starting WebSocket streaming response generation...")

            # Stop previous statuses
            if hasattr(websocket, "_progressive_thinking"):
                await websocket._progressive_thinking.stop_all()

            # Send thinking status for response preparation
            await send_thinking_status(websocket, "preparing_response")

            # Note: generating status will be sent by langgraph workflow if needed

            # Combine retrieved context with custom provided context
            final_context = context_text or ""
            if custom_context_str:
                if final_context:
                    final_context += "\n\n" + custom_context_str
                else:
                    final_context = custom_context_str
                print(
                    f"🔍 [CONTEXT] Added custom context. Total length: {len(final_context)} chars"
                )

            # Use LangGraph integration for intelligent workflow orchestration
            async for (
                chunk_data
            ) in langgraph_integration_service.stream_workflow_with_thinking(
                user=current_user,
                interaction=interaction,
                message=message,
                media_files=media_files,
                websocket=websocket,
                visualize_model=visualize_model,
                assistant_model=assistant_model,
                context=final_context,
            ):
                # Handle LangGraph workflow response format
                if isinstance(chunk_data, str):
                    chunk_content = chunk_data
                    chunk_timestamp = asyncio.get_event_loop().time()
                    chunk_elapsed = 0
                    chunk_number = 0
                    chunk_type = "token"
                elif isinstance(chunk_data, dict):
                    chunk_content = chunk_data.get("content", "")
                    chunk_timestamp = chunk_data.get(
                        "timestamp", asyncio.get_event_loop().time()
                    )
                    chunk_elapsed = chunk_data.get("elapsed_time", 0)
                    chunk_number = chunk_data.get("chunk_number", 0)
                    chunk_type = chunk_data.get("type", "token")

                    # Handle thinking steps
                    if chunk_type == "thinking":
                        thinking_steps = chunk_data.get("thinking_steps", [])
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "thinking",
                                    "content": chunk_content,
                                    "thinking_steps": thinking_steps,
                                    "timestamp": chunk_timestamp,
                                }
                            )
                        )
                        continue
                else:
                    continue

                # Only add non-error content to full_response for title generation
                if not chunk_content.startswith(
                    "Workflow execution failed"
                ) and not chunk_content.startswith("Failed to generate"):
                    full_response += chunk_content

                # Send each chunk with enhanced timing information
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": chunk_type,
                            "content": chunk_content,
                            "timestamp": chunk_timestamp,
                            "elapsed_time": chunk_elapsed,
                            "chunk_number": chunk_number,
                        }
                    )
                )

            print(
                f"✅ WebSocket streaming completed. Full response length: {len(full_response)}"
            )

            # Check if we have meaningful content for title generation
            if not full_response or full_response.strip() == "":
                print("⚠️ No meaningful content received, skipping title generation")
                full_response = "No response generated"

            # Now save the AI response to database and handle background operations
            title_task = None  # Initialize title task variable
            try:
                # Process AI content
                from app.services.interaction import _process_ai_content_fast

                ai_content_type, processed_ai_response = _process_ai_content_fast(
                    full_response
                )

                # Create AI conversation entry
                # Handle MCQ responses differently - store structured data directly
                if ai_content_type == "mcq" and isinstance(processed_ai_response, dict):
                    # Store MCQ data directly without wrapping
                    content = processed_ai_response
                else:
                    # Use the old structure for other content types
                    content = {
                        "type": ai_content_type,
                        "_result": {"content": processed_ai_response},
                    }

                ai_conv = Conversation(
                    interaction_id=str(interaction.id),
                    role=ConversationRole.AI,
                    content=content,
                    input_tokens=0,  # We don't have token info from streaming
                    output_tokens=0,
                    tokens_used=0,
                    points_cost=1,  # Default cost
                    status="completed",
                )
                db.add(ai_conv)
                await db.flush()

                # Commit the conversation FIRST (non-blocking title generation)
                await db.commit()

                # Get subscription plan for title generation
                subscription_plan = None
                if current_user.purchased_subscription:
                    subscription_plan = (
                        current_user.purchased_subscription.subscription.subscription_plan
                    )

                print(f"✅ AI response saved to database successfully")

                # Start title generation in background - it will send title through websocket when ready
                title_task = asyncio.create_task(
                    _update_title_background(
                        interaction_id=str(interaction.id),
                        content_text=full_response,
                        original_message=message or "",
                        assistant_model=assistant_model,
                        subscription_plan=subscription_plan,
                        websocket=websocket,  # Pass websocket to send title when ready
                    )
                )

                # Send completion status (internal only, not shown to user)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "completed",
                            "message": "Response completed successfully!",
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )
                )

                # Start background operations (embeddings, semantic summary, etc.)
                try:
                    from app.services.interaction import _background_operations_enhanced

                    asyncio.create_task(
                        _background_operations_enhanced(
                            user_conv.id,
                            ai_conv.id,
                            str(current_user.id),
                            str(interaction.id),
                            message or "",
                            full_response,
                        )
                    )
                    print(f"✅ Background operations queued")
                except Exception as bg_error:
                    print(f"⚠️ Background operations failed: {bg_error}")

            except Exception as db_error:
                print(f"⚠️ Database operations failed: {db_error}")
                # Continue - the streaming was successful even if DB operations failed

            # Stop all thinking statuses before sending completion
            if hasattr(websocket, "_progressive_thinking"):
                await websocket._progressive_thinking.stop_all()

            # Send completion event
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "completed",
                        "content": full_response,
                        "tokens_used": 0,  # We don't have exact token count from streaming
                        "points_cost": 1,  # Default cost
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )
            )

            # Wait for title generation to complete (with timeout) so title can be sent through websocket
            # Title generation typically takes 1-3 seconds, so we wait up to 8 seconds
            if title_task:
                print("⏳ Waiting for title generation to complete...")
                try:
                    await asyncio.wait_for(title_task, timeout=8.0)
                    print("✅ Title generation completed")
                except asyncio.TimeoutError:
                    print("⏰ Title generation timeout (continuing anyway)")
                except Exception as title_error:
                    print(f"⚠️ Title generation error: {title_error}")
            else:
                print("⚠️ Title task not created, skipping wait")

        finally:
            # Close the database session
            await db.close()

    except WebSocketDisconnect:
        print("🔌 WebSocket /stream-conversation - Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket /stream-conversation - Error: {e}")
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": f"Streaming error: {str(e)}",
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )
            )
        except:
            pass
    finally:
        try:
            # Stop all thinking statuses before closing
            if hasattr(websocket, "_progressive_thinking"):
                await websocket._progressive_thinking.stop_all()
            # Small delay before closing to allow any pending title generation messages
            await asyncio.sleep(0.5)
            await websocket.close()
            print("🔌 WebSocket closed")
        except:
            pass


@interaction_router.get("/health")
async def health_check():
    """Health check endpoint for the streaming service"""
    return {
        "status": "healthy",
        "service": "interaction-websocket-streaming",
        "timestamp": asyncio.get_event_loop().time(),
    }
