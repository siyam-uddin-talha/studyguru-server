from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, List
import json
import asyncio
from app.helpers.user import get_current_user_from_context

router = APIRouter()


# Store active SSE connections
class SSEManager:
    def __init__(self):
        # Dictionary to store user_id -> list of queues
        self.active_connections: Dict[str, List[asyncio.Queue]] = {}

    def add_connection(self, user_id: str, queue: asyncio.Queue):
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(queue)

    def remove_connection(self, user_id: str, queue: asyncio.Queue):
        if user_id in self.active_connections:
            if queue in self.active_connections[user_id]:
                self.active_connections[user_id].remove(queue)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            # Send to all connections for this user
            for queue in self.active_connections[user_id].copy():
                try:
                    await queue.put(message)
                except:
                    # Remove dead connections
                    self.active_connections[user_id].remove(queue)


sse_manager = SSEManager()


@router.get("/events")
async def stream_events(current_user=Depends(get_current_user_from_context)):
    """Server-Sent Events endpoint for real-time notifications"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    async def event_generator():
        # Create a queue for this connection
        queue = asyncio.Queue()
        user_id = str(current_user.id)

        # Add connection to manager
        sse_manager.add_connection(user_id, queue)

        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'data': {'user_id': user_id}})}\n\n"

            while True:
                try:
                    # Wait for messages with timeout
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                except Exception as e:
                    print(f"SSE error: {e}")
                    break

        finally:
            # Clean up connection
            sse_manager.remove_connection(user_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


# Function to send message received notification
async def notify_message_received_sse(
    user_id: str, interaction_id: str, conversation_id: str
):
    """Notify frontend via SSE that user message was received and stored"""
    message = {
        "type": "message_received",
        "data": {
            "interaction_id": interaction_id,
            "conversation_id": conversation_id,
            "status": "processing",
        },
    }
    await sse_manager.send_message(user_id, message)


# Function to send AI response ready notification
async def notify_ai_response_ready_sse(
    user_id: str, interaction_id: str, ai_response: str
):
    """Notify frontend via SSE that AI response is ready"""
    message = {
        "type": "ai_response_ready",
        "data": {"interaction_id": interaction_id, "ai_response": ai_response},
    }
    await sse_manager.send_message(user_id, message)
