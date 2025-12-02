# RAG System Flow for Existing Chats

This document explains how the RAG (Retrieval-Augmented Generation) system retrieves context when a user sends a message to an **existing chat**.

---

## 🚀 STREAMLINED RAG SYSTEM (Recommended - December 2025)

The RAG system has been **streamlined for better performance**. The new simplified approach:

### Key Changes

- **Only 2 context sources** instead of 5:
  1. **Document Content** (if question-specific)
  2. **Vector Search** (hybrid: semantic + keyword)
- **No query expansion** (semantic embeddings already capture meaning)
- **Reduced top_k** from 10 to 5 (less noise)
- **Max context reduced** from 8000 to 4000 chars (better focus)

### Performance Improvements

| Metric                 | Before (5 sources) | After (2 sources)  | Improvement     |
| ---------------------- | ------------------ | ------------------ | --------------- |
| Context Retrieval Time | 2-3.5s             | 0.8-1.2s           | **~60% faster** |
| Context Sources        | 5 parallel queries | 2 parallel queries | **60% fewer**   |
| Vector Search Results  | Top 8-10           | Top 5              | **50% fewer**   |
| Max Context Length     | 8000 chars         | 4000 chars         | **50% less**    |
| Background Operations  | 4 queued tasks     | 2 parallel tasks   | **50% simpler** |

### New Entry Point

```python
from app.services.simplified_context_service import simplified_context_service

context_result = await simplified_context_service.get_simplified_context(
    user_id=user_id,
    interaction_id=interaction_id,
    message=user_message,
    max_context_length=4000  # Reduced for focus
)
```

### Background Operations (Simplified)

```python
from app.services.simplified_background_service import run_simplified_background_operations

# Only creates embeddings - no complex queues
await run_simplified_background_operations(
    user_conv_id=user_conv.id,
    ai_conv_id=ai_conv.id,
    user_id=user_id,
    interaction_id=interaction_id,
    message=message,
    ai_content=ai_response,
)
```

### Removed Components

- `ConversationContext` table (never used in retrieval)
- `Interaction.semantic_summary` field (redundant with vector search)
- `UserLearningProfile.related_interactions` field (complexity without benefit)
- Query expansion in vector search
- Priority queue system for background tasks

### Metrics Tracking

```python
from app.services.rag_metrics_service import track_retrieval_metrics

await track_retrieval_metrics(
    retrieval_time=1.2,
    num_results=5,
    context_length=3500,
    query_type="question_specific"  # or "general"
)
```

---

## Legacy System Documentation (Below)

> ⚠️ **Note**: The documentation below describes the legacy 5-source system.
> New code should use the simplified system described above.

---

## Overview (Legacy)

When a user sends a message to an existing chat, the system performs **multi-level parallel context retrieval** to gather relevant information from:

1. **~~Semantic Summary~~** (REMOVED - from `Interaction` table)
2. **Vector Database** (similarity search via Zilliz/Milvus) ✓ KEPT
3. **Document Content** (from `DocumentContext` table) ✓ KEPT
4. **~~Cross-Interaction Context~~** (REMOVED - from `UserLearningProfile`)
5. **~~Related Conversations~~** (REMOVED - redundant with vector search)

## Entry Point

The flow starts in `process_conversation_message()` which calls:

```python
context_service.get_comprehensive_context(
    user_id=user_id,
    interaction_id=interaction_id,
    message=user_message,
    include_cross_interaction=True,
    max_context_length=8000
)
```

## Context Retrieval Process

### 1. Semantic Summary Context (`_get_semantic_summary_context`)

**Source**: `Interaction.semantic_summary` (JSON field in database)

**What is retrieved**:

- `updated_summary`: Main conversation summary
- `recent_focus`: What the user has been focusing on recently
- `key_topics`: List of topics covered (top 5)
- `accumulated_facts`: Critical facts to remember (top 3)

**How it works**:

- Queries the `Interaction` table for the current interaction
- Extracts the semantic summary JSON
- Formats it into a structured context string

**Example output**:

```
**Conversation Summary:**
User has been learning about quadratic equations and solving problems...

**Recent Focus:**
User is currently working on problem 6 about factoring...

**Topics Covered:** algebra, quadratic equations, factoring

**Key Facts:**
- Quadratic formula: x = (-b ± √(b²-4ac)) / 2a
- Factoring method: Find two numbers that multiply to ac and add to b
```

---

### 2. Vector Search Context (`_get_vector_search_context`)

**Source**: Vector Database (Zilliz/Milvus) via `vector_optimization_service.hybrid_search()`

**What is retrieved**:

- Top 8-10 most relevant past conversations/chunks based on semantic similarity
- Each result includes:
  - `content`: The actual text content (truncated to 1000 chars)
  - `title`: Title of the conversation/document
  - `score`: Similarity score (0.0 to 1.0)
  - `interaction_id`: Which interaction it came from
  - `relevance_score`, `recency_score`, `importance_score`: Ranking scores
  - `topic_tags`: Topics covered
  - `question_numbers`: Question numbers referenced

**How it works**:

#### Step 1: Query Expansion

- Expands the user's query with synonyms and related terms
- Example: "solve equation 6" → ["solve equation 6", "find equation 6", "calculate equation 6", "6.", "problem 6", "question 6"]

#### Step 2: Hybrid Search

Performs **two parallel searches**:

**a) Semantic Search**:

- Converts query to embedding vector
- Searches vector database for similar embeddings
- Uses cosine similarity to find closest matches
- Filters by `user_id` and optionally `interaction_id`

**b) Keyword Search**:

- Searches for exact keyword matches in text content
- Uses BM25-like scoring for relevance

#### Step 3: Combine and Rank

- Merges results from both searches
- Removes duplicates
- Ranks by combined score

#### Step 4: Apply Boosters

- **Recency Boost**: Recent results get +0.1 to score
- **Interaction Boost**: Results from same interaction get +0.2 to score
- **Relevance Boost**: Higher semantic similarity = higher score

#### Step 5: Final Filtering

- Filters by minimum score threshold
- Limits to top K results (8-10)
- Truncates content to 1000 characters per result

**Example output**:

```python
[
    {
        "id": "vec_123",
        "interaction_id": "inter_456",
        "title": "Quadratic Equations Discussion",
        "content": "To solve equation 6, we need to use the quadratic formula...",
        "score": 0.92,
        "relevance_score": 0.9,
        "recency_score": 0.95,
        "question_numbers": ["6"],
        "topic_tags": ["algebra", "quadratic_equations"]
    },
    ...
]
```

---

### 3. Document Content Context (`_get_document_context`)

**Source**: `DocumentContext` table in database

**What is retrieved**:

- If user asks about specific question numbers (e.g., "question 6"), retrieves that exact question
- Otherwise, retrieves all document content from the current interaction
- Each result includes:
  - `content`: Question text + answer
  - `document_type`: "mcq", "written", "mixed", etc.
  - `main_topics`: Topics covered in the document
  - `question_number`: Specific question number
  - `subject_area`: Math, Science, etc.
  - `difficulty_level`: beginner, intermediate, advanced
  - `key_concepts`: Important concepts

**How it works**:

#### Strategy 1: Specific Question Retrieval

If message contains question numbers (e.g., "explain question 6"):

1. Extracts question numbers from message using regex patterns
2. Queries `DocumentContext` for those specific questions
3. Retrieves question text and answer from `question_mapping` and `answer_key`

#### Strategy 2: Full Document Retrieval

If no specific questions or Strategy 1 fails:

1. Queries all `DocumentContext` records for the current interaction
2. Extracts all questions from `question_mapping`
3. Returns all questions with their answers

**Example output**:

```python
[
    {
        "id": "doc_q_6",
        "media_id": "media_789",
        "document_type": "mcq",
        "content": "Question 6: Solve x² + 5x + 6 = 0\n\nAnswer: x = -2 or x = -3",
        "question_number": "6",
        "main_topics": ["quadratic_equations", "factoring"],
        "subject_area": "mathematics",
        "difficulty_level": "intermediate",
        "relevance_score": 0.95
    }
]
```

---

### 4. Cross-Interaction Context (`_get_cross_interaction_context`)

**Source**: `UserLearningProfile.related_interactions` → `Interaction.semantic_summary`

**What is retrieved**:

- Semantic summaries from related interactions (other chats by the same user)
- Only includes interactions where topics are relevant to current message

**How it works**:

1. Queries `UserLearningProfile` for the user
2. Gets list of `related_interaction_ids`
3. Queries `Interaction` table for those interactions
4. Checks if topics in semantic summary match current message
5. Returns summaries of relevant interactions

**Example output**:

```python
[
    {
        "interaction_id": "inter_999",
        "title": "Previous Algebra Session",
        "summary": "User learned about linear equations...",
        "key_topics": ["algebra", "linear_equations"],
        "relevance_score": 0.7
    }
]
```

---

### 5. Related Conversations (`_get_related_conversations`)

**Source**: `Conversation` table (same interaction)

**What is retrieved**:

- Last 3 AI responses from the current interaction
- Provides immediate conversation context

**How it works**:

1. Queries `Conversation` table for current `interaction_id`
2. Filters by `role == "AI"` (only AI responses)
3. Orders by `created_at DESC` (most recent first)
4. Limits to 3 conversations
5. Extracts content from `content._result.note`

**Example output**:

```python
[
    {
        "id": "conv_123",
        "content": "To solve this quadratic equation, you need to...",
        "created_at": "2024-01-15T10:30:00",
        "relevance_score": 0.8
    }
]
```

---

## Context Ranking and Optimization

After retrieving all context sources, the system:

1. **Ranks by relevance**: Prioritizes vector search results (most relevant)
2. **Applies length limits**: Ensures total context doesn't exceed `max_context_length` (8000 chars)
3. **Builds final context string**: Combines all sources into a formatted string

**Priority order**:

1. Semantic Summary (always included if available)
2. Vector Search Results (highest relevance)
3. Document Content (high relevance for specific questions)
4. Related Conversations (contextual relevance)
5. Cross-Interaction Context (lower priority)

---

## Final Context Format

The final context string sent to the LLM looks like:

```
**Conversation Summary:**
[Semantic summary content]

**Previous Discussion:**
[Vector search result 1]
[Vector search result 2]
...

**Document Content:**
[Document question/answer 1]
[Document question/answer 2]
...

**Related Discussion:**
[Recent conversation 1]
[Recent conversation 2]
...

**Related Topic:**
[Cross-interaction summary]
```

---

## Key Features

### 1. Parallel Retrieval

All 5 context sources are retrieved **in parallel** using `asyncio.gather()` for faster performance.

### 2. Hybrid Vector Search

- Combines semantic similarity (embeddings) + keyword matching
- Uses query expansion for better recall
- Applies multiple boosters (recency, interaction, relevance)

### 3. Smart Question Detection

- Automatically detects when user asks about specific question numbers
- Extracts question numbers using regex patterns
- Retrieves exact questions from document context

### 4. Caching

- Context results are cached for 15 minutes
- Cache key includes: `user_id`, `interaction_id`, `message[:100]`
- Reduces redundant database/vector queries

### 5. Timeout Protection

- Context retrieval has a 3.5 second timeout
- Prevents slow queries from blocking the response

---

## Database Tables Used

1. **`Interaction`**: Stores semantic summary
2. **`Conversation`**: Stores individual messages/responses
3. **`DocumentContext`**: Stores document structure and content
4. **`UserLearningProfile`**: Stores related interactions
5. **`ConversationContext`**: Stores preprocessed context snapshots (for future optimization)

---

## Vector Database Structure

**Collection**: Single collection in Zilliz/Milvus

**Fields**:

- `id`: Auto-generated ID
- `user_id`: User identifier
- `interaction_id`: Interaction identifier (nullable)
- `title`: Title of the content
- `text`: The actual text content (max 4096 chars)
- `metadata`: JSON metadata
- `vector`: Embedding vector (dimension based on model)

**Index**: IVF_FLAT index on `vector` field for fast similarity search

---

## Performance Metrics

- **Context Retrieval Time**: ~2-3.5 seconds (with timeout)
- **Vector Search Results**: Top 8-10 results
- **Max Context Length**: 8000 characters
- **Cache TTL**: 15 minutes
- **Query Expansion**: Up to 3 query variations

---

## Example Flow

**User Message**: "Can you explain question 6 again?"

1. **Semantic Summary**: Retrieved from `Interaction.semantic_summary`

   - Contains: "User working on quadratic equations, specifically problem 6"

2. **Vector Search**: Searches for "explain question 6"

   - Finds: Previous explanation of question 6 (score: 0.92)
   - Finds: Related quadratic equation discussions (score: 0.85)

3. **Document Content**: Extracts question 6 from `DocumentContext`

   - Question: "Solve x² + 5x + 6 = 0"
   - Answer: "x = -2 or x = -3"

4. **Related Conversations**: Gets last 3 AI responses

   - Previous explanation of factoring method

5. **Final Context**: All sources combined and sent to LLM
   - LLM generates response using this comprehensive context

---

---

## Web Search Flow

The system uses **Google Serper API** for web search when generating responses. Web search is **automatically enabled** for all regular conversations (not pure document analysis).

### When Web Search is Used

Web search is enabled when:

- User sends a text message (not just document analysis)
- The conversation is not a pure document analysis request
- The LLM agent decides to search for current information

### How Web Search Works

#### Step 1: Agent Setup

The system creates a **LangChain Agent** with a Serper tool:

```python
# Setup Serper tool
search = GoogleSerperAPIWrapper(serper_api_key=settings.SERPER_API_KEY)
serper_tool = Tool(
    name="google_search",
    func=search.run,
    description="Useful for searching the internet for current information, facts, and educational content.",
)
tools = [serper_tool]

# Create Agent with tool calling capability
agent = create_tool_calling_agent(optimized_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

#### Step 2: Agent Decision

The LLM agent **automatically decides** when to use web search based on:

- User's question (e.g., "What is the latest research on...")
- Need for current information
- Fact verification requirements

The agent can call the `google_search` tool during response generation.

#### Step 3: Search Execution

When the agent calls the tool:

1. **Query is sent to Serper API**: `https://google.serper.dev/search`
2. **Serper returns search results**: Top results with snippets, titles, links
3. **Results are formatted** and returned to the agent
4. **Agent incorporates results** into the response

#### Step 4: Response Generation

The agent uses the search results to:

- Provide current, up-to-date information
- Verify facts
- Cite sources when referencing web content
- Enhance educational responses with real-world examples

### Web Search Integration Points

1. **Regular Conversations**: Web search is enabled by default
2. **Streaming Responses**: Web search works with streaming (agent calls tool, then streams response)
3. **Non-Streaming Responses**: Same agent-based approach

### Example Flow

**User Message**: "What are the latest developments in quantum computing?"

1. **Agent receives message** with context
2. **Agent decides** to search for current information
3. **Agent calls `google_search` tool** with query: "latest developments quantum computing 2024"
4. **Serper API returns** top 10 results with snippets
5. **Agent incorporates** search results into response
6. **Response is streamed** to user with citations

### Web Search Limitations

- **API Rate Limits**: Serper API has rate limits based on subscription
- **Cost**: Each search query costs API credits
- **Latency**: Adds ~1-2 seconds to response time
- **Not for Document Analysis**: Disabled for pure document/image analysis

---

## Save to Database Flow

After a user sends a message and receives a response, the system saves data to multiple places in the background.

### 1. Conversation Records (Immediate Save)

**When**: Immediately after user message and AI response

**What is saved**:

#### User Conversation Record

```python
Conversation(
    interaction_id=interaction.id,
    role=ConversationRole.USER,
    content={
        "type": "text",
        "result": {"content": message}
    },
    status="processing"  # Later updated to "completed"
)
```

#### AI Conversation Record

```python
Conversation(
    interaction_id=interaction.id,
    role=ConversationRole.AI,
    content={
        "type": ai_content_type,  # "text", "mcq", etc.
        "_result": {"content": ai_response}
    },
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    tokens_used=total_tokens,
    points_cost=calculated_points,
    status="completed"
)
```

**Database**: `Conversation` table

**Timing**:

- User message: Saved immediately (before AI response)
- AI response: Saved immediately after generation

### 2. Embeddings (Background Save)

**When**: After conversation is saved, in background operations

**What is saved**:

#### User Message Embedding

- **Text**: User's message
- **Title**: "User message in {interaction_id}"
- **Metadata**: Includes interaction_id, conversation_id, topics, facts

#### AI Response Embedding

- **Text**: AI's response (converted to string if needed)
- **Title**: "AI response in {interaction_id}"
- **Metadata**: Same as user message

**How it works**:

1. **Text Processing**:

   - Truncates text to 3000 characters (if longer)
   - Converts to string (handles dict/object inputs)
   - Extracts enhanced metadata (content_type, topic_tags, question_numbers, etc.)

2. **Embedding Creation**:

   - Converts text to embedding vector using the configured embedding model
   - Creates a `Document` object with:
     - `page_content`: The text
     - `metadata`: Enhanced metadata including user_id, interaction_id, title, etc.

3. **Vector Store Insertion**:
   - Adds document to Zilliz/Milvus vector database
   - Auto-generates unique ID
   - Stores vector, text, and all metadata

**Database**: Zilliz/Milvus vector database

**Timing**: Background task (queued via `real_time_context_service`)

**Example**:

```python
# User message embedding
await langchain_service.upsert_embedding(
    conv_id=str(user_conv_id),
    user_id=user_id,
    text="Can you explain question 6?",
    title=f"User message in {interaction_id}",
    metadata={
        "interaction_id": interaction_id,
        "conversation_id": str(user_conv_id),
        "content_type": "question",
        "topic_tags": ["algebra", "quadratic_equations"],
        "question_numbers": ["6"]
    }
)
```

### 3. Semantic Summary (Background Update)

**When**: After conversation is saved, in background operations

**What is updated**:

#### Interaction-Level Semantic Summary

- **Location**: `Interaction.semantic_summary` (JSON field)
- **Content**: Running summary of the entire interaction

**How it works**:

1. **Create Conversation Summary**:

   - Extracts key facts, topics, and summary from the conversation pair
   - Uses LLM to generate semantic summary

2. **Update Running Summary**:

   - Retrieves current `Interaction.semantic_summary`
   - Merges new conversation with existing summary
   - Updates:
     - `updated_summary`: Comprehensive running summary
     - `key_topics`: All important topics covered
     - `recent_focus`: What user has been focusing on recently
     - `accumulated_facts`: Critical facts to remember
     - `question_numbers`: All question numbers referenced
     - `learning_progression`: How user's understanding has evolved
     - `version`: Version number (increments)
     - `last_updated`: Timestamp

3. **Save to Database**:
   - Updates `Interaction.semantic_summary` field
   - Commits transaction

**Database**: `Interaction` table

**Timing**: Background task (queued via `real_time_context_service`)

**Example Update**:

```python
# Before
{
    "updated_summary": "User learning about quadratic equations...",
    "key_topics": ["algebra", "quadratic_equations"],
    "accumulated_facts": ["Quadratic formula: x = (-b ± √(b²-4ac)) / 2a"]
}

# After (new conversation about question 6)
{
    "updated_summary": "User learning about quadratic equations, specifically working on problem 6 about factoring...",
    "key_topics": ["algebra", "quadratic_equations", "factoring"],
    "recent_focus": "Solving problem 6 using factoring method",
    "accumulated_facts": [
        "Quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
        "Factoring method: Find two numbers that multiply to ac and add to b"
    ],
    "question_numbers": ["6"],
    "version": 2,
    "last_updated": "2024-01-15T10:35:00"
}
```

### 4. Conversation Context (Background Save)

**When**: After conversation is saved, in background operations

**What is saved**:

#### ConversationContext Record

- **Context Type**: "conversation_pair"
- **Context Data**: User message + AI response
- **Relevance/Importance Scores**: For ranking in future retrievals

**Database**: `ConversationContext` table (for future optimization)

**Timing**: Background task (queued via `real_time_context_service`)

**Note**: Currently queued but table is designed for future use

### Background Operations Flow

The system uses **two methods** for background operations:

#### Method 1: Enhanced (Real-Time Context Service) - Primary

```python
async def _background_operations_enhanced(
    user_conv_id, ai_conv_id, user_id, interaction_id, message, ai_content
):
    # Step 1: Queue semantic summary update (Priority 1 - High)
    semantic_task_id = await real_time_context_service.queue_context_update(
        update_type="semantic_summary",
        payload={"user_message": message, "ai_response": ai_content},
        priority=1
    )

    # Step 2: Queue embedding updates (Priority 2 - Medium)
    # User message embedding
    user_embedding_task_id = await real_time_context_service.queue_context_update(
        update_type="embedding",
        payload={"text": message, "title": "...", "metadata": {...}},
        priority=2
    )

    # AI response embedding
    ai_embedding_task_id = await real_time_context_service.queue_context_update(
        update_type="embedding",
        payload={"text": ai_content, "title": "...", "metadata": {...}},
        priority=2
    )

    # Step 3: Queue conversation context update (Priority 3 - Low)
    context_task_id = await real_time_context_service.queue_context_update(
        update_type="conversation_context",
        payload={"context_data": {...}},
        priority=3
    )
```

**Benefits**:

- Task queue system for reliability
- Priority-based processing
- Retry on failure
- Non-blocking (doesn't slow down response)

#### Method 2: Fallback (Direct Operations)

If enhanced method fails, falls back to direct operations:

- Creates conversation summary directly
- Updates interaction summary directly
- Creates embeddings directly (in parallel)

### Save Flow Timeline

```
User sends message
    ↓
[IMMEDIATE] Save user conversation to DB
    ↓
Generate AI response (with context retrieval + web search if needed)
    ↓
[IMMEDIATE] Save AI conversation to DB
    ↓
[BACKGROUND] Queue semantic summary update
[BACKGROUND] Queue user message embedding
[BACKGROUND] Queue AI response embedding
[BACKGROUND] Queue conversation context update
    ↓
[BACKGROUND] Process tasks (priority order):
    1. Semantic summary (high priority)
    2. Embeddings (medium priority)
    3. Conversation context (low priority)
```

### Key Points

1. **Immediate Saves**: Conversations are saved immediately for instant user feedback
2. **Background Saves**: Embeddings and summaries are saved in background to not slow down response
3. **Task Queue**: Uses real-time context service with task queue for reliability
4. **Parallel Processing**: Embeddings are created in parallel for speed
5. **Error Handling**: Failures in background operations don't block the main response
6. **Metadata Rich**: All saves include rich metadata for better retrieval later

---

## Notes

- **ConversationContext table**: Currently defined but not actively used in retrieval flow. It's designed for storing preprocessed context snapshots for future optimization.
- **Fresh interactions**: If it's a brand new interaction (no previous messages), context retrieval is skipped.
- **Error handling**: If any context source fails, it's logged but doesn't block the response generation.
- **Web search**: Automatically enabled for regular conversations, disabled for pure document analysis.
- **Background operations**: All background saves are non-blocking and use task queues for reliability.
