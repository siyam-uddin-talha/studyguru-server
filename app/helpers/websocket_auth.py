from fastapi import WebSocket, WebSocketException, status
from typing import Optional
import jwt
from app.core.config import settings


async def get_current_user_from_websocket(websocket: WebSocket) -> Optional[str]:
    """
    Extract user ID from WebSocket connection.
    This is a simplified version - you might want to implement proper JWT validation
    """
    try:
        # Get token from query parameters or headers
        token = websocket.query_params.get("token")
        if not token:
            # Try to get from headers
            token = websocket.headers.get("authorization", "").replace("Bearer ", "")

        if not token:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Authentication token required",
            )

        # Decode JWT token (simplified - you should implement proper validation)
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("sub")
            if not user_id:
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token"
                )
            return user_id
        except jwt.InvalidTokenError:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token"
            )

    except WebSocketException:
        raise
    except Exception:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed"
        )
