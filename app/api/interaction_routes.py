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
from app.services.interaction import process_conversation_message
from app.services.langchain_service import langchain_service
from app.services.langgraph_integration_service import langgraph_integration_service
from app.config.langchain_config import StudyGuruConfig


# Router
interaction_router = APIRouter()


# Helper function to send thinking status updates
async def send_thinking_status(
    websocket: WebSocket,
    status_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Send thinking status update to WebSocket client"""
    try:
        status_data = {
            "type": "thinking",
            "status_type": status_type,
            "message": message,
            "timestamp": asyncio.get_event_loop().time(),
        }
        if details:
            status_data["details"] = details

        await websocket.send_text(json.dumps(status_data))
        print(f"🧠 Sent thinking status: {status_type} - {message}")
    except Exception as e:
        print(f"⚠️ Failed to send thinking status: {e}")


# Thinking status types and messages
THINKING_STATUSES = {
    "analyzing": {
        "message": "Analyzing your question...",
        "details": {"stage": "input_analysis"},
    },
    "searching_context": {
        "message": "Searching through your previous conversations for context...",
        "details": {"stage": "context_retrieval"},
    },
    "checking_guardrails": {
        "message": "Checking content safety...",
        "details": {"stage": "safety_check"},
    },
    "preparing_response": {
        "message": "Preparing my response...",
        "details": {"stage": "response_preparation"},
    },
    "searching_web": {
        "message": "Searching the web for current information...",
        "details": {"stage": "web_search"},
    },
    "generating": {
        "message": "Generating response...",
        "details": {"stage": "ai_generation"},
    },
    "processing_media": {
        "message": "Processing your uploaded files...",
        "details": {"stage": "media_processing"},
    },
    # "saving": {
    #     "message": "Saving our conversation...",
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

            # Log model selection for streaming
            print(f"📊 [STREAMING] Model Selection:")
            print(
                f"   👁️  Visualize Model: {visualize_model or 'default (auto-select)'}"
            )
            print(
                f"   💬 Assistant Model: {assistant_model or 'default (auto-select)'}"
            )
            print(f"   📝 Message: {message[:100] if message else 'No message'}")
            print(f"   📎 Media Files: {len(media_files) if media_files else 0}")

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
            print("🧠 Sending initial thinking status...")
            await send_thinking_status(
                websocket, "analyzing", THINKING_STATUSES["analyzing"]["message"]
            )
            print("🧠 Initial thinking status sent")

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
                # Send thinking status for media processing
                await send_thinking_status(
                    websocket,
                    "processing_media",
                    THINKING_STATUSES["processing_media"]["message"],
                )

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

            # Check guardrails (same as do_conversation)
            try:
                # Send thinking status for guardrail check
                await send_thinking_status(
                    websocket,
                    "checking_guardrails",
                    THINKING_STATUSES["checking_guardrails"]["message"],
                )

                # Extract media URLs for guardrail checking
                media_urls = []
                if media_files:
                    for media_file in media_files:
                        if media_file.get("url"):
                            media_urls.append(media_file["url"])

                # Check guardrails
                guardrail_result = await langchain_service.check_guardrails(
                    message=message or "", media_urls=media_urls
                )

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

                print(f"✅ Guardrail check passed: {guardrail_result.reasoning}")

            except Exception as guardrail_error:
                print(f"⚠️ Guardrail check failed: {guardrail_error}")
                # Continue - don't let guardrail failure block the conversation

            # Get context for the conversation (RAG functionality)
            context_text = ""

            # Only search for context if this is NOT a fresh interaction
            if not is_fresh_interaction:
                try:
                    # Send thinking status for context search
                    await send_thinking_status(
                        websocket,
                        "searching_context",
                        THINKING_STATUSES["searching_context"]["message"],
                    )

                    # Get conversation context using similarity search
                    if interaction_id:
                        # Search within the same interaction for context only
                        print(
                            f"🔍 Searching context within interaction: {interaction.id}"
                        )
                        search_results = (
                            await langchain_service.similarity_search_by_interaction(
                                query=message or "",
                                user_id=str(current_user.id),
                                interaction_id=str(interaction.id),
                                top_k=3,
                            )
                        )
                    else:
                        # This should not happen for existing interactions
                        # If we reach here, it means interaction_id was None but is_fresh_interaction is False
                        # This is an edge case - skip context search to be safe
                        print(
                            f"⚠️ Edge case: No interaction_id but not fresh interaction - skipping context search"
                        )
                        search_results = []

                    # Build context from search results
                    if search_results:
                        context_parts = []
                        for result in search_results:
                            context_parts.append(
                                f"**{result.get('title', 'Previous Discussion')}:**\n{result.get('content', '')}"
                            )
                        context_text = "\n\n".join(context_parts)
                        print(f"🔍 Context length: {len(context_text)} characters")

                except Exception as context_error:
                    print(f"⚠️ Context retrieval failed: {context_error}")
                    # Continue without context - streaming will still work
            else:
                print(f"🆕 Fresh interaction detected - skipping context search")

            # Generate streaming response using LangChain
            full_response = ""
            print(f"🚀 Starting WebSocket streaming response generation...")

            # Send thinking status for response preparation
            await send_thinking_status(
                websocket,
                "preparing_response",
                THINKING_STATUSES["preparing_response"]["message"],
            )

            # Add a minimal delay to ensure status is sent
            await asyncio.sleep(0.1)

            # Check if the question might benefit from web search
            web_search_keywords = [
                "latest",
                "current",
                "recent",
                "2024",
                "2025",
                "news",
                "update",
                "today",
                "now",
            ]
            might_need_web_search = any(
                keyword in (message or "").lower() for keyword in web_search_keywords
            )

            if might_need_web_search:
                await send_thinking_status(
                    websocket,
                    "searching_web",
                    THINKING_STATUSES["searching_web"]["message"],
                )
                await asyncio.sleep(0.1)  # Brief pause to show web search status

            # Send thinking status for AI generation
            await send_thinking_status(
                websocket, "generating", THINKING_STATUSES["generating"]["message"]
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
            try:
                # Send thinking status for saving
                # await send_thinking_status(
                #     websocket, "saving", THINKING_STATUSES["saving"]["message"]
                # )

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

                # Update interaction title if needed
                try:
                    from app.services.interaction import (
                        _extract_interaction_metadata_fast,
                    )

                    await _extract_interaction_metadata_fast(
                        interaction=interaction,
                        content_text=full_response,
                        original_message=message or "",
                    )
                except Exception as title_error:
                    print(f"⚠️ Title generation failed: {title_error}")

                # Commit the conversation
                await db.commit()
                print(f"✅ AI response saved to database successfully")

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
            await websocket.close()
        except:
            pass


# @interaction_router.websocket("/stream-simple")
# async def websocket_stream_simple(websocket: WebSocket):
#     """
#     WebSocket endpoint for simple conversation streaming

#     This is a simpler version that just streams the AI response without
#     the full conversation processing pipeline.
#     """
#     print("🔍 WebSocket /stream-simple - Connection attempt")

#     try:
#         # Authenticate user first
#         print("🔍 WebSocket /stream-simple - Starting authentication")
#         user_id = await get_current_user_from_websocket(websocket)
#         print(
#             f"🔍 WebSocket /stream-simple - Authentication successful, user_id: {user_id}"
#         )

#         await websocket.accept()
#         print("🔍 WebSocket /stream-simple - Connection accepted")
#     except Exception as e:
#         print(f"❌ WebSocket /stream-simple - Authentication failed: {e}")
#         raise

#     try:
#         # Receive the initial message
#         data = await websocket.receive_text()
#         payload = json.loads(data)

#         message = payload.get("message", "")
#         interaction_id = payload.get("interaction_id")

#         print(f"🔍 WebSocket /stream-simple - Received message: '{message[:50]}...'")

#         if not message:
#             await websocket.send_text(
#                 json.dumps(
#                     {
#                         "type": "error",
#                         "error": "Message is required",
#                         "timestamp": asyncio.get_event_loop().time(),
#                     }
#                 )
#             )
#             await websocket.close()
#             return

#         # Send initial metadata
#         await websocket.send_text(
#             json.dumps(
#                 {
#                     "type": "start",
#                     "message": "Starting response generation...",
#                     "timestamp": asyncio.get_event_loop().time(),
#                 }
#             )
#         )

#         # Generate streaming response using LangChain
#         full_response = ""
#         async for chunk in langchain_service.generate_conversation_response_streaming(
#             message=message,
#             context="",
#             image_urls=[],
#             max_tokens=2000,
#         ):
#             full_response += chunk

#             # Send each chunk
#             await websocket.send_text(
#                 json.dumps(
#                     {
#                         "type": "token",
#                         "content": chunk,
#                         "timestamp": asyncio.get_event_loop().time(),
#                     }
#                 )
#             )

#         # Send completion
#         await websocket.send_text(
#             json.dumps(
#                 {
#                     "type": "complete",
#                     "content": full_response,
#                     "timestamp": asyncio.get_event_loop().time(),
#                 }
#             )
#         )

#     except WebSocketDisconnect:
#         print("🔌 WebSocket /stream-simple - Client disconnected")
#     except Exception as e:
#         print(f"❌ WebSocket /stream-simple - Error: {e}")
#         try:
#             await websocket.send_text(
#                 json.dumps(
#                     {
#                         "type": "error",
#                         "error": f"Error: {str(e)}",
#                         "timestamp": asyncio.get_event_loop().time(),
#                     }
#                 )
#             )
#         except:
#             pass
#     finally:
#         try:
#             await websocket.close()
#         except:
#             pass


# @interaction_router.websocket("/test-stream")
# async def websocket_test_stream(websocket: WebSocket):
#     """
#     WebSocket endpoint for testing streaming functionality
#     """
#     print("🔍 WebSocket /test-stream - Connection attempt")

#     try:
#         # Authenticate user first
#         print("🔍 WebSocket /test-stream - Starting authentication")
#         user_id = await get_current_user_from_websocket(websocket)
#         print(
#             f"🔍 WebSocket /test-stream - Authentication successful, user_id: {user_id}"
#         )

#         await websocket.accept()
#         print("🔍 WebSocket /test-stream - Connection accepted")
#     except Exception as e:
#         print(f"❌ WebSocket /test-stream - Authentication failed: {e}")
#         raise

#     try:
#         # Receive the initial message
#         data = await websocket.receive_text()
#         payload = json.loads(data)

#         message = payload.get("message", "Hello")

#         print(f"🔍 WebSocket /test-stream - Received message: '{message}'")

#         # Send initial metadata
#         await websocket.send_text(
#             json.dumps(
#                 {
#                     "type": "start",
#                     "message": "Starting test streaming...",
#                     "timestamp": asyncio.get_event_loop().time(),
#                 }
#             )
#         )

#         # Generate a simple test response
#         test_response = f"Test response for: '{message}'. This is a streaming test. "
#         test_response += "Each word is being sent as it's generated. "
#         test_response += "The WebSocket streaming is working correctly!"

#         # Split the response into words for streaming
#         words = test_response.split()

#         for i, word in enumerate(words):
#             await websocket.send_text(
#                 json.dumps(
#                     {
#                         "type": "token",
#                         "content": word + " ",
#                         "timestamp": asyncio.get_event_loop().time(),
#                     }
#                 )
#             )

#             # Small delay to simulate real streaming
#             await asyncio.sleep(0.1)

#         # Send completion
#         await websocket.send_text(
#             json.dumps(
#                 {
#                     "type": "complete",
#                     "content": test_response,
#                     "timestamp": asyncio.get_event_loop().time(),
#                 }
#             )
#         )

#     except WebSocketDisconnect:
#         print("🔌 WebSocket /test-stream - Client disconnected")
#     except Exception as e:
#         print(f"❌ WebSocket /test-stream - Error: {e}")
#         try:
#             await websocket.send_text(
#                 json.dumps(
#                     {
#                         "type": "error",
#                         "error": f"Test error: {str(e)}",
#                         "timestamp": asyncio.get_event_loop().time(),
#                     }
#                 )
#             )
#         except:
#             pass
#     finally:
#         try:
#             await websocket.close()
#         except:
#             pass


@interaction_router.get("/health")
async def health_check():
    """Health check endpoint for the streaming service"""
    return {
        "status": "healthy",
        "service": "interaction-websocket-streaming",
        "timestamp": asyncio.get_event_loop().time(),
    }
