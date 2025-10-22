# Web Search Implementation for StudyGuru Pro

## Overview

This document describes the implementation of web search functionality for StudyGuru Pro using Gemini's native Google Search grounding tool. This feature enables the AI to automatically search the web for current information and provide up-to-date, accurate responses.

## Features

- **Native Google Search Integration**: Uses Gemini's built-in Google Search tool
- **Automatic Web Search**: AI automatically determines when to search for information
- **Current Information**: Provides up-to-date information from the web
- **Source Citations**: Includes citations from web search results
- **Educational Focus**: Optimized for educational content and learning

## Implementation Details

### 1. Dependencies

The implementation requires the following imports:

```python
# Import native Google Search tool
try:
    from google.generativeai.types import Tool
    from google.generativeai.protos import GoogleSearchRetrieval
except ImportError:
    print("\nRun 'pip install google-generativeai' to get native tool support.\n")
    Tool = None
    GoogleSearchRetrieval = None
```

### 2. Model Configuration

The web search functionality is implemented in the `StudyGuruModels` class with multiple approaches:

#### A. Dedicated Web Search Model

```python
@staticmethod
def get_web_search_model(
    temperature: float = 0.2, max_tokens: int = 5000, streaming: bool = True
):
    """Get configured model with Google Search tool integration - Gemini only"""
    if not StudyGuruModels._is_gemini_model():
        raise ValueError("Web search functionality is only available with Gemini models")

    if Tool is None or GoogleSearchRetrieval is None:
        raise ImportError("Native Google Search tool not available. Run 'pip install google-generativeai' to get native tool support.")

    # Define the native grounding tool
    grounding_tool = Tool(google_search_retrieval=GoogleSearchRetrieval())

    # Initialize the model and pass it the tool
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=temperature,
        google_api_key=settings.GOOGLE_API_KEY,
        max_output_tokens=max_tokens,
        request_timeout=120,
        cache=cache_manager.get_response_cache(),
        tools=[grounding_tool],  # This enables native "AUTO WEBSEARCH"
        streaming=streaming,
    )
```

#### B. Chat Model with Web Search Parameter

```python
@staticmethod
def get_chat_model(
    temperature=0.2,
    max_tokens=5000,
    reasoning_effort="low",
    verbosity="low",
    streaming=True,
    web_search=True,  # NEW: Enable/disable web search
):
    """Get chat model with optional web search capability"""
    cache = StudyGuruModels._get_cache()

    if StudyGuruModels._is_gemini_model():
        # Prepare tools list for web search
        tools = []
        if web_search and Tool is not None and GoogleSearchRetrieval is not None:
            grounding_tool = Tool(google_search_retrieval=GoogleSearchRetrieval())
            tools = [grounding_tool]

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            max_output_tokens=max_tokens,
            request_timeout=120,
            cache=cache,
            tools=tools,  # Include web search tool if enabled
        )
    # ... rest of implementation
```

#### C. Vision Model with Web Search Parameter

```python
@staticmethod
def get_vision_model(
    temperature: float = 0.3,
    max_tokens: int = 5000,
    verbosity: str = "low",
    streaming: bool = True,
    web_search: bool = True,  # NEW: Enable/disable web search
):
    """Get configured vision model with optional web search capability"""
    if StudyGuruModels._is_gemini_model():
        # Prepare tools list for web search
        tools = []
        if web_search and Tool is not None and GoogleSearchRetrieval is not None:
            grounding_tool = Tool(google_search_retrieval=GoogleSearchRetrieval())
            tools = [grounding_tool]

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            max_output_tokens=max_tokens,
            request_timeout=120,
            cache=cache_manager.get_response_cache(),
            tools=tools,  # Include web search tool if enabled
        )
    # ... rest of implementation
```

### 3. Prompt Template

A specialized prompt template for web search conversations:

```python
WEB_SEARCH_CONVERSATION = ChatPromptTemplate.from_messages([
    (
        "system",
        """StudyGuru AI - Educational assistant with web search capabilities.

You have access to Google Search to find current information, verify facts, and provide up-to-date educational content.

FORMAT: ### headers, number questions (1., 2.), options (A., B., C., D.) if exist else direct solution, Answer: [solution] (no bold), Explanation: [text] (no bold), • for lists, plain text math (e.g., "x squared"), clear paragraphs.

When using web search:
- Search for current information when needed
- Verify facts and provide accurate, up-to-date information
- Cite sources when referencing web search results
- Be encouraging, professional, focused on learning

Use web search to enhance your educational responses with current, accurate information.""",
    ),
    ("human", "{question}"),
])
```

### 4. Chain Configuration

The web search chain combines the model with the prompt template:

```python
@staticmethod
def get_web_search_chain():
    """Get web search conversation chain with Google Search tool integration"""
    model = StudyGuruModels.get_web_search_model()
    parser = StrOutputParser()
    return StudyGuruPrompts.WEB_SEARCH_CONVERSATION | model | parser
```

## Usage Examples

### 1. Dedicated Web Search Chain

```python
from app.config.langchain_config import StudyGuruConfig

# Get the web search chain
web_search_chain = StudyGuruConfig.CHAINS.get_web_search_chain()

# Ask a question that benefits from current information
question = "What are the latest developments in quantum computing in 2024?"
response = web_search_chain.invoke({"question": question})
print(response)
```

### 2. Chat Model with Web Search Enabled

```python
from app.config.langchain_config import StudyGuruConfig
from langchain_core.prompts import ChatPromptTemplate

# Get chat model with web search enabled (default)
chat_model = StudyGuruConfig.MODELS.get_chat_model(web_search=True)

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# Create chain
chain = prompt | chat_model

# Ask a question
question = "What is the current status of renewable energy adoption worldwide?"
response = chain.invoke({"question": question})
print(response.content)
```

### 3. Vision Model with Web Search

```python
from app.config.langchain_config import StudyGuruConfig

# Get vision model with web search enabled
vision_model = StudyGuruConfig.MODELS.get_vision_model(web_search=True)

# Ask a question
question = "Explain the latest research on artificial intelligence in education"
response = vision_model.invoke([{"role": "user", "content": question}])
print(response.content)
```

### 4. Disabling Web Search

```python
from app.config.langchain_config import StudyGuruConfig

# Get chat model with web search disabled
chat_model = StudyGuruConfig.MODELS.get_chat_model(web_search=False)

# This will use only the model's training data, no web search
question = "What are the current trends in machine learning for 2024?"
response = chat_model.invoke([{"role": "user", "content": question}])
print(response.content)
```

### 5. Async Usage

```python
import asyncio

async def ask_with_web_search():
    web_search_chain = StudyGuruConfig.CHAINS.get_web_search_chain()
    response = await web_search_chain.ainvoke({"question": question})
    return response

# Run the async function
response = asyncio.run(ask_with_web_search())
```

## Requirements

### Environment Variables

Make sure you have the following environment variables set:

```bash
LLM_MODEL=gemini
GOOGLE_API_KEY=your_google_api_key_here
```

### Model Compatibility

- **Gemini Only**: Web search functionality is only available with Gemini models
- **Model Version**: Uses `gemini-2.5-pro` for optimal performance
- **Fallback**: If not using Gemini, the system will raise a clear error message

## How It Works

1. **User Query**: User asks a question that may benefit from current information
2. **Model Analysis**: Gemini analyzes the query to determine if web search is needed
3. **Automatic Search**: If needed, the model automatically performs Google searches
4. **Information Synthesis**: The model processes search results and synthesizes information
5. **Response Generation**: Returns an educational response with current information and citations

## Benefits

- **Current Information**: Always provides up-to-date information
- **Automatic**: No need to manually trigger searches
- **Educational Focus**: Optimized for learning and educational content
- **Source Citations**: Includes references to web sources
- **Seamless Integration**: Works with existing StudyGuru infrastructure

## Error Handling

The implementation includes proper error handling:

- **Model Check**: Validates that Gemini is being used
- **API Errors**: Handles Google API errors gracefully
- **Timeout Handling**: Includes request timeout configuration
- **Fallback Messages**: Provides clear error messages for troubleshooting

## Performance Considerations

- **Caching**: Uses response caching for improved performance
- **Streaming**: Supports streaming responses for better user experience
- **Timeout**: Configured with appropriate timeouts (120 seconds)
- **Token Limits**: Configurable token limits for different use cases

## Testing

Use the provided example script to test the implementation:

```bash
cd server
python example_web_search_usage.py
```

## Future Enhancements

Potential future improvements:

1. **Search Result Filtering**: Add domain-specific filtering
2. **Custom Search Parameters**: Allow customization of search parameters
3. **Result Caching**: Cache search results for repeated queries
4. **Multi-language Support**: Enhanced support for non-English searches
5. **Search Analytics**: Track search usage and performance

## Troubleshooting

### Common Issues

1. **"Web search functionality is only available with Gemini models"**

   - Solution: Set `LLM_MODEL=gemini` in your environment variables

2. **Import errors for GoogleSearch**

   - Solution: Ensure you have the latest version of `google-generativeai` package

3. **API key errors**

   - Solution: Verify your `GOOGLE_API_KEY` is correctly set and valid

4. **Timeout errors**
   - Solution: Increase the `request_timeout` parameter if needed

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The web search implementation provides StudyGuru Pro with powerful capabilities to access current information and provide up-to-date educational responses. The integration is seamless, automatic, and optimized for educational use cases.
