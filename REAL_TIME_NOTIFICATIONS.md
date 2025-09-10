# Real-Time Notifications Implementation

This implementation provides real-time notifications for your React Native StudyGuru app using Server-Sent Events (SSE) as the primary method, with WebSocket as an alternative.

## How It Works

### Backend Flow:

1. **User sends message** → GraphQL `do_conversation` mutation
2. **Message stored** → User conversation saved to database
3. **Notification sent** → Frontend receives "message_received" event
4. **AI processing** → Guardrails, vector search, AI generation
5. **Response ready** → Frontend receives "ai_response_ready" event

### Frontend Flow:

1. **Send message** → Immediately show user message with "sending" status
2. **Message received** → Update status to "received" (loading stops)
3. **AI response** → Show AI response and remove typing indicator

## Files Created/Modified

### Backend:

- `server/app/api/websocket_routes.py` - WebSocket implementation
- `server/app/api/sse_routes.py` - Server-Sent Events implementation
- `server/app/helpers/websocket_auth.py` - WebSocket authentication
- `server/app/services/interaction.py` - Updated to send notifications
- `server/app/main.py` - Added WebSocket and SSE routes

### Frontend:

- `app/src/services/EventSource.ts` - React Native SSE client
- `app/src/hooks/useRealTimeNotifications.ts` - React hook for real-time notifications
- `app/src/screens/interaction/Interaction.tsx` - Updated to use real-time notifications
- `app/src/graphql/query/interaction.ts` - Updated GraphQL queries

## Usage

### 1. Backend Setup

The backend automatically sends notifications when:

- A user message is received and stored
- An AI response is ready

### 2. Frontend Integration

```typescript
import { useRealTimeNotifications } from "../hooks/useRealTimeNotifications";

const { isConnected } = useRealTimeNotifications({
  onMessageReceived: (data) => {
    // Handle message received notification
    console.log("Message received:", data);
  },
  onAIResponseReady: (data) => {
    // Handle AI response ready
    console.log("AI response:", data.ai_response);
  },
});
```

### 3. Connection Status

The app shows connection status in the header:

- 🟢 **Live** - Real-time notifications active
- 🟠 **Offline** - Fallback mode (polling or manual refresh)

## Key Features

### ✅ Immediate Feedback

- User messages appear instantly
- Loading state stops when message is received
- AI responses appear when ready

### ✅ Connection Management

- Automatic reconnection on network issues
- App state handling (background/foreground)
- Fallback when real-time is unavailable

### ✅ Error Handling

- Graceful degradation when notifications fail
- User-friendly error messages
- Automatic retry mechanisms

## Configuration

### Environment Variables

Update your API URL in `useRealTimeNotifications.ts`:

```typescript
const apiUrl = "https://your-api-domain.com"; // Replace with your actual API URL
```

### Authentication

The hook automatically uses your auth token from the Redux store. Make sure your auth store has either:

- `token` property, or
- `accessToken` property

## Fallback Behavior

When real-time notifications are not available:

1. Messages still send successfully
2. User sees "Processing your message..." instead of "SG Pro is typing..."
3. After 10 seconds, user gets a warning to refresh manually
4. All core functionality remains intact

## Testing

### WebSocket (Alternative)

If you prefer WebSocket over SSE, you can use the `WebSocketService.ts` file instead of `EventSource.ts`.

### Manual Testing

1. Send a message
2. Check console logs for notification events
3. Verify connection status indicator
4. Test with network interruptions

## Troubleshooting

### Common Issues:

1. **No notifications received** - Check auth token and API URL
2. **Connection keeps dropping** - Check network stability
3. **Messages not updating** - Verify GraphQL mutation response format

### Debug Mode:

Enable console logging to see all notification events and connection status.

## Performance Notes

- SSE connections are lightweight and efficient
- Automatic cleanup on component unmount
- Minimal battery impact with proper app state handling
- Connection pooling for multiple components

This implementation provides a smooth, real-time chat experience while maintaining reliability and performance.
