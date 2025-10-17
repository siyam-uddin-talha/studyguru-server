# StudyGuru Streaming API Guide

## Overview

The StudyGuru Streaming API provides real-time conversation streaming using Server-Sent Events (SSE). This allows clients to receive AI responses as they are generated, providing a more interactive and responsive user experience.

## Features

- **Real-time streaming**: Receive AI responses token by token
- **Full conversation support**: Same functionality as the GraphQL `do_conversation` mutation
- **Authentication**: JWT token-based authentication
- **Error handling**: Comprehensive error handling with streaming error events
- **Caching integration**: Leverages the caching mechanisms for improved performance

## Endpoints

### 1. Full Conversation Streaming

**Endpoint**: `POST /api/stream/stream-conversation`

**Description**: Full-featured streaming conversation endpoint with the same capabilities as the GraphQL mutation.

**Headers**:

```
Authorization: Bearer <jwt-token>
Content-Type: application/json
Accept: text/event-stream
```

**Request Body**:

```json
{
  "interaction_id": "optional-existing-interaction-id",
  "message": "Your message here",
  "media_files": [
    {
      "id": "media-id",
      "url": "optional-media-url"
    }
  ],
  "max_tokens": 5000
}
```

**Response**: Server-Sent Events stream with the following event types:

#### Event Types

1. **Metadata Event**:

```json
{
  "type": "metadata",
  "interaction_id": "interaction-uuid",
  "is_new_interaction": true,
  "user_id": "user-uuid"
}
```

2. **Token Event** (streaming content):

```json
{
  "type": "token",
  "content": "partial response text",
  "timestamp": 1234567890.123
}
```

3. **Complete Event**:

```json
{
  "type": "complete",
  "content": "full response text",
  "tokens_used": 150,
  "points_cost": 5,
  "timestamp": 1234567890.123
}
```

4. **Error Event**:

```json
{
  "type": "error",
  "error": "error message",
  "timestamp": 1234567890.123
}
```

### 2. Simple Streaming

**Endpoint**: `GET /api/stream/stream-simple`

**Description**: Simplified streaming endpoint for quick testing and simple conversations.

**Headers**:

```
Authorization: Bearer <jwt-token>
Accept: text/event-stream
```

**Query Parameters**:

- `message` (required): The message to send
- `interaction_id` (optional): Existing interaction ID

**Example**:

```
GET /api/stream/stream-simple?message=What%20is%202%2B2%3F&interaction_id=optional-id
```

**Response**: Same SSE format as full streaming, but with simplified event types.

### 3. Health Check

**Endpoint**: `GET /api/stream/health`

**Description**: Check if the streaming service is healthy.

**Response**:

```json
{
  "status": "healthy",
  "service": "interaction-streaming",
  "timestamp": 1234567890.123
}
```

## Usage Examples

### JavaScript/TypeScript Client

```javascript
class StudyGuruStreamingClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  async streamConversation(message, interactionId = null) {
    const url = `${this.baseUrl}/api/stream/stream-conversation`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        message,
        interaction_id: interactionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));

          switch (data.type) {
            case "metadata":
              console.log("Metadata:", data);
              break;
            case "token":
              // Append token to UI
              this.appendToUI(data.content);
              break;
            case "complete":
              console.log("Complete:", data);
              break;
            case "error":
              console.error("Error:", data.error);
              break;
          }
        }
      }
    }
  }

  appendToUI(content) {
    // Implement UI update logic
    const chatContainer = document.getElementById("chat-container");
    const lastMessage = chatContainer.lastElementChild;
    if (lastMessage && lastMessage.classList.contains("ai-message")) {
      lastMessage.textContent += content;
    } else {
      const newMessage = document.createElement("div");
      newMessage.className = "ai-message";
      newMessage.textContent = content;
      chatContainer.appendChild(newMessage);
    }
  }
}

// Usage
const client = new StudyGuruStreamingClient(
  "http://localhost:8000",
  "your-jwt-token"
);
client.streamConversation("Hello, can you help me with math?");
```

### Python Client

```python
import asyncio
import aiohttp
import json

async def stream_conversation(message, token=None):
    url = "http://localhost:8000/api/stream/stream-conversation"

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {"message": message}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()

                if line.startswith('data: '):
                    data = json.loads(line[6:])

                    if data['type'] == 'token':
                        print(data['content'], end='', flush=True)
                    elif data['type'] == 'complete':
                        print(f"\n\nComplete: {data}")
                        break
                    elif data['type'] == 'error':
                        print(f"\nError: {data['error']}")
                        break

# Usage
asyncio.run(stream_conversation("What is 2+2?"))
```

### React Native Client

```typescript
import { EventSourcePolyfill } from "react-native-event-source";

class StudyGuruStreamingService {
  private baseUrl: string;
  private token: string;

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  async streamConversation(
    message: string,
    onToken: (token: string) => void,
    onComplete: (data: any) => void,
    onError: (error: string) => void
  ) {
    const url = `${this.baseUrl}/api/stream/stream-conversation`;

    const eventSource = new EventSourcePolyfill(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "token":
          onToken(data.content);
          break;
        case "complete":
          onComplete(data);
          eventSource.close();
          break;
        case "error":
          onError(data.error);
          eventSource.close();
          break;
      }
    };

    eventSource.onerror = (error) => {
      onError("Connection error");
      eventSource.close();
    };

    return eventSource;
  }
}
```

## Error Handling

The streaming API provides comprehensive error handling:

1. **Authentication Errors**: 401 status with error details
2. **Validation Errors**: 400 status with validation messages
3. **Streaming Errors**: Error events in the SSE stream
4. **Connection Errors**: Proper cleanup and error reporting

## Performance Considerations

1. **Caching**: The streaming API leverages the same caching mechanisms as the regular API
2. **Connection Management**: Properly close connections to avoid resource leaks
3. **Rate Limiting**: Consider implementing rate limiting for production use
4. **Memory Usage**: Stream processing is memory-efficient for large responses

## Testing

Use the provided test client to test the streaming functionality:

```bash
cd server
python test_streaming_client.py
```

The test client includes:

- Health check testing
- Simple streaming test
- Full conversation streaming test
- Error handling validation

## Integration with Existing System

The streaming API is fully integrated with the existing StudyGuru system:

- **Authentication**: Uses the same JWT token system
- **Database**: Same interaction and conversation models
- **Caching**: Leverages the caching implementation
- **Models**: Uses the same LangChain model configurations
- **Context**: Same context retrieval and processing

## Production Deployment

For production deployment:

1. **Load Balancing**: Ensure SSE connections are properly handled by load balancers
2. **Timeout Configuration**: Set appropriate timeouts for long-running streams
3. **Monitoring**: Monitor connection counts and stream performance
4. **Error Logging**: Implement comprehensive error logging
5. **Rate Limiting**: Add rate limiting to prevent abuse

## Troubleshooting

### Common Issues

1. **Connection Drops**: Check network stability and timeout settings
2. **Authentication Failures**: Verify JWT token validity
3. **Empty Responses**: Check if the AI service is properly configured
4. **Memory Issues**: Monitor memory usage during long streams

### Debug Mode

Enable debug logging by setting the environment variable:

```bash
ENABLE_LOGS=true
```

This will provide detailed logging of the streaming process.
