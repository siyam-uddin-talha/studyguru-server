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
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.helpers.auth import verify_token
from app.helpers.websocket_auth import get_current_user_from_websocket
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.services.interaction import process_conversation_message
from app.services.langchain_service import langchain_service
from app.config.langchain_config import StudyGuruConfig


# Router
interaction_router = APIRouter()


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

    # Fetch user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@interaction_router.websocket("/stream-conversation")
async def websocket_stream_conversation(websocket: WebSocket):
    """
    WebSocket endpoint for full conversation streaming

    This endpoint provides the same functionality as the do_conversation mutation
    but streams the AI response in real-time using WebSocket.
    """
    print("🔍 WebSocket /stream-conversation - Connection attempt")

    try:
        # Authenticate user first
        print("🔍 WebSocket /stream-conversation - Starting authentication")
        user_id = await get_current_user_from_websocket(websocket)
        print(
            f"🔍 WebSocket /stream-conversation - Authentication successful, user_id: {user_id}"
        )

        await websocket.accept()
        print("🔍 WebSocket /stream-conversation - Connection accepted")
    except Exception as e:
        print(f"❌ WebSocket /stream-conversation - Authentication failed: {e}")
        raise

    try:
        # Get database session
        db_gen = get_db()
        db = await db_gen.__anext__()

        try:
            # Get user from database
            result = await db.execute(select(User).where(User.id == user_id))
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

            print(
                f"🔍 WebSocket /stream-conversation - Received payload: message='{message[:50]}...', interaction_id={interaction_id}"
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

            # Check guardrails (same as do_conversation)
            try:
                # Extract media URLs for guardrail checking
                media_urls = []
                if media_files:
                    for media_file in media_files:
                        if media_file.get("url"):
                            media_urls.append(media_file["url"])

                # Check guardrails
                guardrail_result = await langchain_service.check_guardrails(
                    message=message or "", image_urls=media_urls
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

            try:
                # Get conversation context using similarity search
                if interaction_id:
                    # Search within the same interaction for context
                    search_results = (
                        await langchain_service.similarity_search_by_interaction(
                            query=message or "",
                            user_id=str(current_user.id),
                            interaction_id=str(interaction.id),
                            top_k=3,
                        )
                    )
                else:
                    # Search across all user conversations for context
                    search_results = await langchain_service.similarity_search(
                        query=message or "",
                        user_id=str(current_user.id),
                        top_k=5,
                    )

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

            # Generate streaming response using LangChain
            full_response = ""
            print(f"🚀 Starting WebSocket streaming response generation...")

            # Use LangChain streaming for real AI responses
            async for (
                chunk
            ) in langchain_service.generate_conversation_response_streaming(
                message=message,
                context=context_text,
                image_urls=media_urls,
                max_tokens=max_tokens,
            ):
                full_response += chunk

                # Send each chunk
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "token",
                            "content": chunk,
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )
                )

            print(
                f"✅ WebSocket streaming completed. Full response length: {len(full_response)}"
            )

            # Now save the AI response to database and handle background operations
            try:
                # Process AI content
                from app.services.interaction import _process_ai_content_fast

                ai_content_type, processed_ai_response = _process_ai_content_fast(
                    full_response
                )

                # Create AI conversation entry
                ai_conv = Conversation(
                    interaction_id=str(interaction.id),
                    role=ConversationRole.AI,
                    content={
                        "type": ai_content_type,
                        "_result": {"content": processed_ai_response},
                    },
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
                        "type": "complete",
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
