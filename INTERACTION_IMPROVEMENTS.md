# Interaction Processing Improvements

## Overview

This document outlines the improvements made to the interaction processing system to handle fresh interactions and enhance vector database retrieval for existing interactions.

## Key Improvements

### 1. Fresh Interaction Handling

**Problem**: Fresh interactions (new conversations) were not properly extracting and storing title and summary_title from AI responses.

**Solution**:

- Added logic to detect fresh interactions (when `interaction.title` and `interaction.summary_title` are both None)
- Parse AI response as JSON to extract structured data including `title` and `summary_title`
- Update the Interaction table with extracted metadata
- Format MCQ questions nicely for better user experience

**Code Changes**:

```python
# For fresh interactions, try to extract title and summary_title from AI response
if not interaction.title and not interaction.summary_title:
    try:
        parsed_response = json.loads(content_text)
        if isinstance(parsed_response, dict):
            extracted_title = parsed_response.get("title")
            extracted_summary_title = parsed_response.get("summary_title")

            if extracted_title:
                interaction.title = extracted_title
            if extracted_summary_title:
                interaction.summary_title = extracted_summary_title
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
```

### 2. Enhanced Vector Database Retrieval (RAG Pattern)

**Problem**: Vector database retrieval was basic and didn't follow proper RAG (Retrieval-Augmented Generation) patterns.

**Solution**:

- Enhanced query building to include interaction context
- Increased top_k from 5 to 8 for better context retrieval
- Added score threshold filtering (0.3) to ensure quality matches
- Implemented duplicate prevention and content truncation
- Better context formatting with clear separation

**Code Changes**:

```python
# Build comprehensive query for vector search
vector_query_parts = []

# Add current message
if message:
    vector_query_parts.append(message)

# Add media descriptions if available
if media_objects:
    for media in media_objects:
        vector_query_parts.append(f"Document: {media.original_filename}")

# For existing interactions, include interaction context
if interaction.title:
    vector_query_parts.append(f"Topic: {interaction.title}")
if interaction.summary_title:
    vector_query_parts.append(f"Context: {interaction.summary_title}")

# Combine all parts for vector search
vector_query = " ".join(vector_query_parts).strip()
```

### 3. Improved System Prompts

**Problem**: System prompts were generic and didn't leverage context effectively.

**Solution**:

- Different prompts for fresh vs existing interactions
- Better context presentation in user headers
- Clear instructions for using retrieved context

**Fresh Interaction Prompt**:

```
You are StudyGuru AI analyzing educational content. Analyze the given image/document and provide a structured response:
1. First, detect the language of the content
2. Identify if this contains MCQ (Multiple Choice Questions) or written questions
3. Provide a short, descriptive title for the page/content
4. Provide a summary title that describes what you will help the user with
...
```

**Existing Interaction Prompt**:

```
You are StudyGuru AI, an educational assistant. You have access to the user's learning history and context.

Current conversation topic: {interaction.title}
Context summary: {interaction.summary_title}

Instructions:
1. Use the retrieved context from the user's learning history when relevant
2. Maintain consistency with the user's learning style
3. Build upon previous knowledge when possible
...
```

### 4. GraphQL Schema Updates

**Problem**: GraphQL types didn't match the actual model structure and lacked important fields.

**Solution**:

- Updated `InteractionType` to include all model fields
- Added `ai_response` field to `InteractionResponse`
- Enhanced resolver to return proper interaction state

**Updated InteractionType**:

```python
@strawberry.type
class InteractionType:
    id: str
    user_id: str
    file_id: Optional[str] = None
    analysis_response: Optional[Dict[str, Any]] = None
    question_type: Optional[str] = None
    detected_language: Optional[str] = None
    title: Optional[str] = None
    summary_title: Optional[str] = None
    tokens_used: Optional[int] = None
    points_cost: Optional[int] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file: Optional[MediaType] = None
```

## Benefits

1. **Better User Experience**: Fresh interactions now properly extract and store metadata for better organization
2. **Improved Context**: Enhanced vector retrieval provides more relevant context for follow-up questions
3. **Consistent Responses**: System prompts ensure consistent behavior across different interaction types
4. **Better API**: GraphQL schema now properly reflects the data model and provides AI responses

## Testing

A test script `test_interaction_improvements.py` has been created to verify:

- Fresh interaction handling with title/summary_title extraction
- Existing interaction processing with vector retrieval
- Vector database functionality

## Usage

### Fresh Interaction

```graphql
mutation {
  doConversation(
    input: { message: "What is photosynthesis?", maxTokens: 500 }
  ) {
    success
    message
    interactionId
    isNewInteraction
    interaction {
      title
      summaryTitle
    }
    aiResponse
  }
}
```

### Existing Interaction

```graphql
mutation {
  doConversation(
    input: {
      interactionId: "existing-interaction-id"
      message: "Can you explain more about the light-dependent reactions?"
      maxTokens: 500
    }
  ) {
    success
    message
    interactionId
    isNewInteraction
    aiResponse
  }
}
```

## Future Enhancements

1. **Conversation History**: Store and retrieve full conversation history for better context
2. **Topic Clustering**: Group related interactions by topic for better organization
3. **Learning Analytics**: Track user progress and learning patterns
4. **Advanced RAG**: Implement more sophisticated retrieval strategies
