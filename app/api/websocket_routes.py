from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
import json
import asyncio
from app.helpers.websocket_auth import get_current_user_from_websocket

router = APIRouter()


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
                print(f"Failed to send WebSocket message: {e}")
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
