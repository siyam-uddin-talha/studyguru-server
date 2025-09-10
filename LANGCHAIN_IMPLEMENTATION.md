# LangChain Implementation for StudyGuru Pro

This document describes the complete LangChain implementation that replaces the raw OpenAI module codes and provides a robust vector database integration.

## 🚀 Overview

The LangChain implementation provides:

- **Structured AI Operations**: Document analysis, conversation generation, and guardrail checks
- **Vector Database Integration**: Milvus/Zilliz for semantic search and RAG
- **Configuration Management**: Centralized configuration for all LangChain components
- **Backward Compatibility**: Legacy OpenAI service wrapper for existing code

## 📁 File Structure

```
server/
├── app/
│   ├── config/
│   │   └── langchain_config.py          # Centralized LangChain configuration
│   ├── services/
│   │   ├── langchain_service.py         # Main LangChain service
│   │   └── openai_service.py            # Legacy wrapper (updated)
│   └── services/
│       └── interaction.py               # Updated to use LangChain
└── test_langchain.py                    # Test script
```

## 🔧 Installation

The required packages have been installed:

```bash
pip install langchain langchain-openai langchain-community langchain-core pymilvus
```

## 🏗️ Architecture

### 1. Configuration System (`langchain_config.py`)

Centralized configuration for all LangChain components:

```python
from app.config.langchain_config import StudyGuruConfig

# Get configured models
chat_model = StudyGuruConfig.MODELS.get_chat_model()
vision_model = StudyGuruConfig.MODELS.get_vision_model()
embeddings_model = StudyGuruConfig.MODELS.get_embeddings_model()

# Get pre-configured chains
doc_chain = StudyGuruConfig.CHAINS.get_document_analysis_chain()
guardrail_chain = StudyGuruConfig.CHAINS.get_guardrail_chain()
conversation_chain = StudyGuruConfig.CHAINS.get_conversation_chain()
```

### 2. Main Service (`langchain_service.py`)

The `LangChainService` class provides all AI operations:

```python
from app.services.langchain_service import langchain_service

# Document analysis
result = await langchain_service.analyze_document(file_url, max_tokens=1000)

# Guardrail check
guardrail_result = await langchain_service.check_guardrails(message, image_urls)

# Conversation generation
response, input_tokens, output_tokens, total_tokens = await langchain_service.generate_conversation_response(
    message="Hello",
    context="Previous context",
    image_urls=["https://example.com/image.jpg"],
    interaction_title="Math Help",
    interaction_summary="Algebra basics",
    max_tokens=1000
)

# Vector operations
await langchain_service.upsert_embedding(doc_id, user_id, text, title, metadata)
results = await langchain_service.similarity_search(query, user_id, top_k=5)
```

### 3. Vector Database Integration

**Milvus/Zilliz Configuration:**

- **Collection**: `studyguru_embeddings`
- **Dimension**: 1536 (text-embedding-3-small)
- **Index**: IVF_FLAT with L2 distance
- **Fields**: id, user_id, title, content, metadata, vector

**Features:**

- Automatic collection creation
- User-specific filtering
- Semantic similarity search
- Metadata-based filtering

## 🔄 Migration from Raw OpenAI

### Before (Raw OpenAI):

```python
# Old implementation
response = client.responses.create(
    model="gpt-5",
    input=[...],
    max_output_tokens=1000
)
content = response.output_text
```

### After (LangChain):

```python
# New implementation
response, input_tokens, output_tokens, total_tokens = await langchain_service.generate_conversation_response(
    message=message,
    context=context,
    max_tokens=1000
)
```

## 🎯 Key Features

### 1. Document Analysis

- **Vision Model**: GPT-4o for image/document analysis
- **Structured Output**: JSON format with type, language, title, and content
- **MCQ Support**: Automatic question extraction and formatting
- **Multi-language**: Automatic language detection

### 2. Guardrail System

- **Content Filtering**: Educational purpose validation
- **Code Generation**: Context-aware code writing detection
- **Safety Checks**: Adult content and inappropriate material filtering
- **Structured Output**: Violation type and reasoning

### 3. Conversation Generation

- **Context-Aware**: Uses interaction history and titles
- **RAG Integration**: Retrieves relevant context from vector database
- **Multi-modal**: Supports text and image inputs
- **Token Tracking**: Accurate token usage monitoring

### 4. Vector Database

- **Semantic Search**: Find relevant content using embeddings
- **User Isolation**: Each user's data is filtered separately
- **Metadata Support**: Rich metadata for advanced filtering
- **Automatic Indexing**: New content is automatically indexed

## 🧪 Testing

Run the test script to verify the implementation:

```bash
cd server
python test_langchain.py
```

The test script verifies:

- ✅ Configuration system
- ✅ Model initialization
- ✅ Guardrail checks
- ✅ Document analysis
- ✅ Conversation generation
- ✅ Vector store operations
- ✅ Points calculation

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Vector Database (Milvus/Zilliz)
ZILLIZ_URI=your_zilliz_uri
ZILLIZ_TOKEN=your_zilliz_token
ZILLIZ_COLLECTION=studyguru_embeddings
ZILLIZ_DIMENSION=1536
ZILLIZ_INDEX_METRIC=L2
ZILLIZ_CONSISTENCY_LEVEL=Bounded
```

### Model Configuration

```python
# In langchain_config.py
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5
POINTS_PER_TOKEN = 100
```

## 📊 Performance Benefits

### 1. **Structured Operations**

- Consistent input/output formats
- Better error handling
- Improved debugging capabilities

### 2. **Vector Database**

- Fast semantic search (sub-second)
- Scalable to millions of documents
- User-specific data isolation

### 3. **Configuration Management**

- Centralized settings
- Easy model switching
- Environment-specific configurations

### 4. **Token Management**

- Accurate token counting
- Cost optimization
- Usage analytics

## 🔄 Backward Compatibility

The existing `OpenAIService` class has been updated to use LangChain internally:

```python
# This still works exactly the same
from app.services.openai_service import OpenAIService

result = await OpenAIService.analyze_document(file_url)
embeddings = await OpenAIService.generate_embedding(text)
similar = await OpenAIService.similarity_search(query, user_id=user_id)
```

## 🚀 Usage Examples

### 1. Document Analysis

```python
# Analyze a document/image
result = await langchain_service.analyze_document(
    file_url="https://example.com/document.jpg",
    max_tokens=1000
)

print(f"Type: {result['type']}")
print(f"Language: {result['language']}")
print(f"Title: {result['title']}")
```

### 2. Conversation with Context

```python
# Generate response with context
response, input_tokens, output_tokens, total_tokens = await langchain_service.generate_conversation_response(
    message="Explain derivatives",
    context="Previous calculus discussion...",
    interaction_title="Calculus Help",
    interaction_summary="Basic calculus concepts",
    max_tokens=500
)
```

### 3. Vector Search

```python
# Find similar content
results = await langchain_service.similarity_search(
    query="calculus derivatives",
    user_id="user123",
    top_k=5
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Score: {result['score']}")
```

### 4. Guardrail Check

```python
# Check content safety
guardrail_result = await langchain_service.check_guardrails(
    message="Help me with my homework",
    image_urls=["https://example.com/math_problem.jpg"]
)

if guardrail_result.is_violation:
    print(f"Blocked: {guardrail_result.violation_type}")
else:
    print("Content is safe")
```

## 🔍 Monitoring and Debugging

### 1. **Token Usage Tracking**

```python
# All operations return token usage
response, input_tokens, output_tokens, total_tokens = await langchain_service.generate_conversation_response(...)
print(f"Used {total_tokens} tokens (input: {input_tokens}, output: {output_tokens})")
```

### 2. **Error Handling**

```python
try:
    result = await langchain_service.analyze_document(file_url)
except Exception as e:
    print(f"Analysis failed: {e}")
    # Fallback logic
```

### 3. **Vector Store Status**

```python
# Check if vector store is available
if langchain_service.vector_store is None:
    print("Vector store not available - using fallback")
```

## 🎉 Benefits Summary

1. **🔧 Better Architecture**: Structured, maintainable code
2. **🚀 Improved Performance**: Optimized vector operations
3. **🛡️ Enhanced Safety**: Robust guardrail system
4. **📊 Better Analytics**: Accurate token and usage tracking
5. **🔄 Easy Migration**: Backward compatible with existing code
6. **⚡ Scalable**: Vector database supports millions of documents
7. **🎯 Context-Aware**: RAG-powered responses with relevant context

The LangChain implementation provides a robust, scalable, and maintainable foundation for StudyGuru Pro's AI capabilities while maintaining full backward compatibility with existing code.
