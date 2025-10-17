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
        print(
            f"🔍 WebSocket auth - Token from query params: {token[:20] if token else 'None'}..."
        )

        if not token:
            # Try to get from headers
            token = websocket.headers.get("authorization", "").replace("Bearer ", "")
            print(
                f"🔍 WebSocket auth - Token from headers: {token[:20] if token else 'None'}..."
            )

        if not token:
            print("❌ WebSocket auth - No token found")
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Authentication token required",
            )

        # Decode JWT token (simplified - you should implement proper validation)
        try:
            print(
                f"🔍 WebSocket auth - Decoding token with secret key: {settings.JWT_SECRET_KEY[:10] if settings.JWT_SECRET_KEY else 'None'}..."
            )
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("sub")
            print(f"🔍 WebSocket auth - Decoded user_id: {user_id}")

            if not user_id:
                print("❌ WebSocket auth - No user_id in token payload")
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token"
                )
            return user_id
        except jwt.InvalidTokenError as e:
            print(f"❌ WebSocket auth - JWT decode error: {e}")
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token"
            )

    except WebSocketException:
        raise
    except Exception as e:
        print(f"❌ WebSocket auth - General error: {e}")
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed"
        )
