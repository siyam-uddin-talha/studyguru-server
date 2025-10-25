from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
import json
import asyncio
import jwt
from app.helpers.websocket_auth import get_current_user_from_websocket
from app.services.langchain_service import langchain_service
from app.core.config import settings

router = APIRouter()


# Test endpoint to verify JWT token decoding
@router.get("/test-jwt")
async def test_jwt_token(token: str):
    """Test endpoint to verify JWT token decoding"""
    try:
        print(f"🔍 Testing JWT token: {token[:20]}...")
        print(f"🔍 JWT Secret Key: {settings.JWT_SECRET_KEY[:10]}...")

        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        print(f"🔍 Decoded payload: {payload}")

        return {"success": True, "payload": payload, "user_id": payload.get("sub")}
    except jwt.InvalidTokenError as e:
        print(f"❌ JWT decode error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ General error: {e}")
        return {"success": False, "error": str(e)}


# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        # Dictionary to store user_id -> websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_personal_message(self, message: dict, user_id: str):
        if (
            user_id not in self.active_connections
            or not self.active_connections[user_id]
        ):
            raise Exception(f"No active WebSocket connections for user {user_id}")

        # Send to all connections for this user
        sent_successfully = False
        for connection in self.active_connections[user_id].copy():
            try:
                await connection.send_text(json.dumps(message))
                sent_successfully = True
            except Exception as e:
                # Remove dead connections
                self.active_connections[user_id].remove(connection)

        if not sent_successfully:
            raise Exception(
                f"Failed to send message to any WebSocket connection for user {user_id}"
            )


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Authenticate user
    user_id = await get_current_user_from_websocket(websocket)

    await manager.connect(websocket, user_id)
    try:
        # Send connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connected",
                    "data": {
                        "user_id": user_id,
                        "message": "WebSocket connected successfully",
                    },
                }
            )
        )

        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle different message types from client
                if message.get("type") == "ping":
                    await websocket.send_text(
                        json.dumps({"type": "pong", "data": message.get("data")})
                    )
            except json.JSONDecodeError:
                # Handle non-JSON messages
                await websocket.send_text(
                    json.dumps({"type": "error", "data": "Invalid JSON format"})
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


# Function to send message received notification
async def notify_message_received(
    user_id: str, interaction_id: str, conversation_id: str
):
    """Notify frontend that user message was received and stored"""
    message = {
        "type": "message_received",
        "data": {
            "interaction_id": interaction_id,
            "conversation_id": conversation_id,
            "status": "processing",
        },
    }
    await manager.send_personal_message(message, user_id)


# Function to send AI response ready notification
async def notify_ai_response_ready(user_id: str, interaction_id: str, ai_response: str):
    """Notify frontend that AI response is ready"""
    message = {
        "type": "ai_response_ready",
        "data": {"interaction_id": interaction_id, "ai_response": ai_response},
    }
    await manager.send_personal_message(message, user_id)


# NOTE: WebSocket streaming endpoints have been moved to interaction_routes.py
# for better organization and to work exactly like the do_conversation mutation


@router.websocket("/ws-simple")
async def websocket_simple_stream(websocket: WebSocket):
    """WebSocket endpoint for simple conversation streaming with authentication"""
    print("🔍 WebSocket /ws-simple - Connection attempt")
    print(f"🔍 WebSocket URL: {websocket.url}")
    print(f"🔍 WebSocket headers: {dict(websocket.headers)}")
    print(f"🔍 WebSocket query params: {dict(websocket.query_params)}")

    try:
        # Authenticate user
        print("🔍 WebSocket /ws-simple - Starting authentication")
        user_id = await get_current_user_from_websocket(websocket)
        print(
            f"🔍 WebSocket /ws-simple - Authentication successful, user_id: {user_id}"
        )

        await websocket.accept()
        print("🔍 WebSocket /ws-simple - Connection accepted")
    except Exception as e:
        print(f"❌ WebSocket /ws-simple - Connection failed: {e}")
        raise

    try:
        # Receive the initial message
        data = await websocket.receive_text()
        payload = json.loads(data)

        message = payload.get("message", "")
        interaction_id = payload.get("interaction_id")
        max_tokens = payload.get("max_tokens", 2000)

        if not message:
            await websocket.send_text(
                json.dumps({"type": "error", "error": "Message is required"})
            )
            await websocket.close()
            return

        # Send initial metadata
        await websocket.send_text(
            json.dumps(
                {
                    "type": "start",
                    "message": "Starting response generation...",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        )

        # Generate streaming response with optimized settings
        full_response = ""
        print(f"🚀 Starting WebSocket streaming for: '{message[:50]}...'")

        async for chunk in langchain_service.generate_conversation_response_streaming(
            message=message,
            context="",  # No context for simple questions to reduce overhead
            media_urls=[],
            max_tokens=max_tokens,
        ):
            full_response += chunk

            # Send each chunk immediately
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

        # Send completion
        await websocket.send_text(
            json.dumps(
                {
                    "type": "complete",
                    "content": full_response,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        )

    except Exception as e:
        print(f"❌ WebSocket /ws-simple - Error: {e}")
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "error": f"Error: {str(e)}",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        )
    finally:
        await websocket.close()


@router.websocket("/ws-conversation")
async def websocket_conversation_stream(websocket: WebSocket):
    """WebSocket endpoint for full conversation streaming with media support"""
    print("🔍 WebSocket /ws-conversation - Connection attempt")

    try:
        # Authenticate user first
        print("🔍 WebSocket /ws-conversation - Starting authentication")
        user_id = await get_current_user_from_websocket(websocket)
        print(
            f"🔍 WebSocket /ws-conversation - Authentication successful, user_id: {user_id}"
        )

        await websocket.accept()
        print("🔍 WebSocket /ws-conversation - Connection accepted")
    except Exception as e:
        print(f"❌ WebSocket /ws-conversation - Connection failed: {e}")
        raise

    try:
        # Receive the initial message
        data = await websocket.receive_text()
        payload = json.loads(data)

        message = payload.get("message", "")
        interaction_id = payload.get("interaction_id")
        media_files = payload.get("media_files", [])
        max_tokens = payload.get("max_tokens", 5000)

        print(
            f"🔍 WebSocket /ws-conversation - Received payload: message='{message[:50]}...', interaction_id={interaction_id}, media_files={len(media_files)}"
        )

        if not message:
            await websocket.send_text(
                json.dumps({"type": "error", "error": "Message is required"})
            )
            await websocket.close()
            return

        # Send initial metadata
        await websocket.send_text(
            json.dumps(
                {
                    "type": "start",
                    "message": "Starting conversation processing...",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        )

        # Extract image URLs from media files
        image_urls = []
        if media_files:
            for media_file in media_files:
                if media_file.get("url"):
                    image_urls.append(media_file["url"])

        # Generate streaming response with optimized settings
        full_response = ""
        print(f"🚀 Starting WebSocket conversation streaming for: '{message[:50]}...'")

        async for chunk in langchain_service.generate_conversation_response_streaming(
            message=message,
            context="",  # No context for faster responses
            media_urls=image_urls,
            max_tokens=max_tokens,
        ):
            full_response += chunk

            # Send each chunk immediately
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
            f"✅ WebSocket conversation streaming completed. Full response length: {len(full_response)}"
        )

        # Send completion
        await websocket.send_text(
            json.dumps(
                {
                    "type": "complete",
                    "content": full_response,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        )

    except Exception as e:
        print(f"❌ WebSocket /ws-conversation - Error: {e}")
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "error": f"Error: {str(e)}",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        )
    finally:
        await websocket.close()


# @router.websocket("/ws-test")
# async def websocket_test_stream(websocket: WebSocket):
#     """WebSocket endpoint for testing streaming functionality"""
#     # Authenticate user first
#     user_id = await get_current_user_from_websocket(websocket)

#     await websocket.accept()

#     try:
#         # Receive the initial message
#         data = await websocket.receive_text()
#         payload = json.loads(data)

#         message = payload.get("message", "Hello")

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

#     except Exception as e:
#         await websocket.send_text(
#             json.dumps(
#                 {
#                     "type": "error",
#                     "error": f"Test error: {str(e)}",
#                     "timestamp": asyncio.get_event_loop().time(),
#                 }
#             )
#         )
#     finally:
#         await websocket.close()
