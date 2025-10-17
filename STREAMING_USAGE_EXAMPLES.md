# StudyGuru Streaming API Usage Examples

## React Component Example

```tsx
import React, { useState, useRef, useEffect } from "react";

interface StreamingChatProps {
  token: string;
  baseUrl?: string;
}

const StreamingChat: React.FC<StreamingChatProps> = ({
  token,
  baseUrl = "http://localhost:8000",
}) => {
  const [messages, setMessages] = useState<
    Array<{ id: string; role: "user" | "ai"; content: string }>
  >([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentAiMessage, setCurrentAiMessage] = useState("");
  const eventSourceRef = useRef<EventSource | null>(null);

  const startStreaming = async (message: string) => {
    if (!message.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      role: "user" as const,
      content: message,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsStreaming(true);
    setCurrentAiMessage("");

    // Create AI message placeholder
    const aiMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      {
        id: aiMessageId,
        role: "ai",
        content: "",
      },
    ]);

    try {
      const response = await fetch(
        `${baseUrl}/api/stream/stream-conversation`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({
            message: message,
            max_tokens: 2000,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body reader available");
      }

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              switch (data.type) {
                case "metadata":
                  console.log("Stream metadata:", data);
                  break;

                case "token":
                  setCurrentAiMessage((prev) => prev + data.content);
                  break;

                case "complete":
                  console.log("Stream complete:", data);
                  setIsStreaming(false);
                  // Update the AI message with final content
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === aiMessageId
                        ? { ...msg, content: data.content || currentAiMessage }
                        : msg
                    )
                  );
                  setCurrentAiMessage("");
                  break;

                case "error":
                  console.error("Stream error:", data.error);
                  setIsStreaming(false);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === aiMessageId
                        ? { ...msg, content: `Error: ${data.error}` }
                        : msg
                    )
                  );
                  setCurrentAiMessage("");
                  break;
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      setIsStreaming(false);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId
            ? { ...msg, content: `Connection error: ${error}` }
            : msg
        )
      );
      setCurrentAiMessage("");
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isStreaming && inputMessage.trim()) {
      startStreaming(inputMessage);
    }
  };

  // Update the AI message in real-time as tokens arrive
  useEffect(() => {
    if (currentAiMessage) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === (Date.now() + 1).toString() && msg.role === "ai"
            ? { ...msg, content: currentAiMessage }
            : msg
        )
      );
    }
  }, [currentAiMessage]);

  return (
    <div className="streaming-chat">
      <div className="messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="content">
              {message.content}
              {message.role === "ai" &&
                isStreaming &&
                message.content === currentAiMessage && (
                  <span className="cursor">|</span>
                )}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !inputMessage.trim()}>
          {isStreaming ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default StreamingChat;
```

## React Native Example

```tsx
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
} from "react-native";

interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
}

const StreamingChatRN: React.FC<{ token: string; baseUrl?: string }> = ({
  token,
  baseUrl = "http://localhost:8000",
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentAiMessage, setCurrentAiMessage] = useState("");

  const startStreaming = async (message: string) => {
    if (!message.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: message,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsStreaming(true);
    setCurrentAiMessage("");

    // Create AI message placeholder
    const aiMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      {
        id: aiMessageId,
        role: "ai",
        content: "",
      },
    ]);

    try {
      const response = await fetch(`${baseUrl}/api/stream/stream-simple`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "text/event-stream",
        },
        // Note: For GET requests, we need to encode the message as a query parameter
      });

      const url = new URL(`${baseUrl}/api/stream/stream-simple`);
      url.searchParams.append("message", message);

      const finalResponse = await fetch(url.toString(), {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "text/event-stream",
        },
      });

      if (!finalResponse.ok) {
        throw new Error(`HTTP error! status: ${finalResponse.status}`);
      }

      const reader = finalResponse.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body reader available");
      }

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              switch (data.type) {
                case "start":
                  console.log("Stream started");
                  break;

                case "token":
                  setCurrentAiMessage((prev) => prev + data.content);
                  break;

                case "complete":
                  console.log("Stream complete");
                  setIsStreaming(false);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === aiMessageId
                        ? { ...msg, content: data.content || currentAiMessage }
                        : msg
                    )
                  );
                  setCurrentAiMessage("");
                  break;

                case "error":
                  console.error("Stream error:", data.error);
                  setIsStreaming(false);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === aiMessageId
                        ? { ...msg, content: `Error: ${data.error}` }
                        : msg
                    )
                  );
                  setCurrentAiMessage("");
                  break;
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      setIsStreaming(false);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId
            ? { ...msg, content: `Connection error: ${error}` }
            : msg
        )
      );
      setCurrentAiMessage("");
    }
  };

  const handleSubmit = () => {
    if (!isStreaming && inputMessage.trim()) {
      startStreaming(inputMessage);
    }
  };

  // Update the AI message in real-time
  useEffect(() => {
    if (currentAiMessage) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === (Date.now() + 1).toString() && msg.role === "ai"
            ? { ...msg, content: currentAiMessage }
            : msg
        )
      );
    }
  }, [currentAiMessage]);

  return (
    <View style={styles.container}>
      <ScrollView style={styles.messages}>
        {messages.map((message) => (
          <View
            key={message.id}
            style={[
              styles.message,
              message.role === "user" ? styles.userMessage : styles.aiMessage,
            ]}
          >
            <Text style={styles.messageText}>
              {message.content}
              {message.role === "ai" &&
                isStreaming &&
                message.content === currentAiMessage && (
                  <Text style={styles.cursor}>|</Text>
                )}
            </Text>
          </View>
        ))}
      </ScrollView>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={inputMessage}
          onChangeText={setInputMessage}
          placeholder="Type your message..."
          editable={!isStreaming}
        />
        <TouchableOpacity
          style={[styles.button, isStreaming && styles.buttonDisabled]}
          onPress={handleSubmit}
          disabled={isStreaming || !inputMessage.trim()}
        >
          <Text style={styles.buttonText}>
            {isStreaming ? "Sending..." : "Send"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  messages: {
    flex: 1,
    padding: 16,
  },
  message: {
    marginBottom: 12,
    padding: 12,
    borderRadius: 8,
    maxWidth: "80%",
  },
  userMessage: {
    backgroundColor: "#007AFF",
    alignSelf: "flex-end",
  },
  aiMessage: {
    backgroundColor: "#E5E5EA",
    alignSelf: "flex-start",
  },
  messageText: {
    fontSize: 16,
    color: "#000",
  },
  cursor: {
    color: "#007AFF",
    fontWeight: "bold",
  },
  inputContainer: {
    flexDirection: "row",
    padding: 16,
    backgroundColor: "#fff",
    borderTopWidth: 1,
    borderTopColor: "#E5E5EA",
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#E5E5EA",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
  },
  button: {
    backgroundColor: "#007AFF",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    justifyContent: "center",
  },
  buttonDisabled: {
    backgroundColor: "#C7C7CC",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "bold",
  },
});

export default StreamingChatRN;
```

## Vue.js Example

```vue
<template>
  <div class="streaming-chat">
    <div class="messages">
      <div
        v-for="message in messages"
        :key="message.id"
        :class="['message', message.role]"
      >
        <div class="content">
          {{ message.content }}
          <span
            v-if="
              message.role === 'ai' &&
              isStreaming &&
              message.content === currentAiMessage
            "
            class="cursor"
            >|</span
          >
        </div>
      </div>
    </div>

    <form @submit.prevent="handleSubmit" class="input-form">
      <input
        v-model="inputMessage"
        type="text"
        placeholder="Type your message..."
        :disabled="isStreaming"
      />
      <button type="submit" :disabled="isStreaming || !inputMessage.trim()">
        {{ isStreaming ? "Sending..." : "Send" }}
      </button>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";

interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
}

const props = defineProps<{
  token: string;
  baseUrl?: string;
}>();

const messages = ref<Message[]>([]);
const inputMessage = ref("");
const isStreaming = ref(false);
const currentAiMessage = ref("");

const startStreaming = async (message: string) => {
  if (!message.trim()) return;

  // Add user message
  const userMessage: Message = {
    id: Date.now().toString(),
    role: "user",
    content: message,
  };
  messages.value.push(userMessage);
  inputMessage.value = "";
  isStreaming.value = true;
  currentAiMessage.value = "";

  // Create AI message placeholder
  const aiMessageId = (Date.now() + 1).toString();
  messages.value.push({
    id: aiMessageId,
    role: "ai",
    content: "",
  });

  try {
    const response = await fetch(
      `${
        props.baseUrl || "http://localhost:8000"
      }/api/stream/stream-conversation`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${props.token}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          message: message,
          max_tokens: 2000,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("No response body reader available");
    }

    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case "metadata":
                console.log("Stream metadata:", data);
                break;

              case "token":
                currentAiMessage.value += data.content;
                break;

              case "complete":
                console.log("Stream complete:", data);
                isStreaming.value = false;
                const aiMessage = messages.value.find(
                  (msg) => msg.id === aiMessageId
                );
                if (aiMessage) {
                  aiMessage.content = data.content || currentAiMessage.value;
                }
                currentAiMessage.value = "";
                break;

              case "error":
                console.error("Stream error:", data.error);
                isStreaming.value = false;
                const errorMessage = messages.value.find(
                  (msg) => msg.id === aiMessageId
                );
                if (errorMessage) {
                  errorMessage.content = `Error: ${data.error}`;
                }
                currentAiMessage.value = "";
                break;
            }
          } catch (e) {
            console.error("Error parsing SSE data:", e);
          }
        }
      }
    }
  } catch (error) {
    console.error("Streaming error:", error);
    isStreaming.value = false;
    const errorMessage = messages.value.find((msg) => msg.id === aiMessageId);
    if (errorMessage) {
      errorMessage.content = `Connection error: ${error}`;
    }
    currentAiMessage.value = "";
  }
};

const handleSubmit = () => {
  if (!isStreaming.value && inputMessage.value.trim()) {
    startStreaming(inputMessage.value);
  }
};

// Update the AI message in real-time
watch(currentAiMessage, (newContent) => {
  if (newContent) {
    const lastAiMessage = messages.value.find(
      (msg) => msg.role === "ai" && msg.content === ""
    );
    if (lastAiMessage) {
      lastAiMessage.content = newContent;
    }
  }
});
</script>

<style scoped>
.streaming-chat {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
}

.messages {
  height: 400px;
  overflow-y: auto;
  border: 1px solid #ddd;
  padding: 16px;
  margin-bottom: 16px;
  background: #f9f9f9;
}

.message {
  margin-bottom: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  max-width: 80%;
}

.message.user {
  background: #007aff;
  color: white;
  margin-left: auto;
}

.message.ai {
  background: #e5e5ea;
  color: black;
}

.cursor {
  color: #007aff;
  font-weight: bold;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%,
  50% {
    opacity: 1;
  }
  51%,
  100% {
    opacity: 0;
  }
}

.input-form {
  display: flex;
  gap: 8px;
}

.input-form input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.input-form button {
  padding: 8px 16px;
  background: #007aff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-form button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
```

## Testing the Implementation

1. **Start the server**:

```bash
cd server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Test with the provided client**:

```bash
python test_streaming_client.py
```

3. **Test with curl**:

```bash
curl -N -H "Accept: text/event-stream" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -X POST \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?"}' \
     http://localhost:8000/api/stream/stream-conversation
```

These examples show how to integrate the StudyGuru streaming API into various frontend frameworks, providing real-time AI responses for a better user experience.
