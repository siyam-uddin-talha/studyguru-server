# LangGraph Multi-Source Summarization Implementation Guide

## 🎯 Overview

This implementation provides intelligent orchestration for complex multi-source summarization workflows using LangGraph. It automatically detects when advanced processing is needed and routes tasks through appropriate workflows.

## 🏗️ Architecture

### Core Components

1. **LangGraphWorkflowService** - Main workflow orchestration
2. **LangGraphIntegrationService** - Integration with existing system
3. **ThinkingConfigManager** - Automatic thinking configuration
4. **InputAnalyzer** - Intelligent input analysis and routing

### Workflow States

```
Input Analysis → PDF Processing → Web Search → Source Integration → Summary Generation
```

## 🚀 Key Features

### 1. Intelligent Orchestration

**Automatic Workflow Selection:**

- Simple text → Standard processing
- PDFs + Links → LangGraph workflow
- Analytical tasks → Enhanced processing
- Multi-source content → Comprehensive integration

### 2. ThinkingConfig Integration

**For Gemini Models:**

```python
thinking_config = {
    "thinking_config": {
        "include_thoughts": True,
        "thinking_budget": 2048  # For analytical tasks
    }
}
```

**For GPT Models:**

```python
thinking_config = {
    "reasoning_effort": "high"  # For complex tasks
}
```

### 3. Real-time Thinking Display

**UI Integration:**

- Thinking steps displayed in real-time
- No database storage required
- Progressive thinking updates
- User-friendly status messages

## 📋 Usage Examples

### 1. Simple Text Processing

```python
# This will use standard processing (no LangGraph)
message = "Hello, how are you?"
media_files = []

result = await langgraph_integration_service.process_with_langgraph(
    user=user,
    interaction=interaction,
    message=message,
    media_files=media_files
)
```

### 2. PDF Processing

```python
# This will trigger LangGraph workflow
message = "Please analyze this document"
media_files = [
    {
        'id': 'pdf1',
        'url': 'https://example.com/document.pdf',
        'type': 'application/pdf',
        'name': 'document.pdf'
    }
]

result = await langgraph_integration_service.process_with_langgraph(
    user=user,
    interaction=interaction,
    message=message,
    media_files=media_files
)
```

### 3. Hybrid Processing (PDFs + Links)

```python
# This will trigger comprehensive LangGraph workflow
message = "Please analyze these documents and research current information: https://example.com/article"
media_files = [
    {
        'id': 'pdf1',
        'url': 'https://example.com/document1.pdf',
        'type': 'application/pdf',
        'name': 'document1.pdf'
    },
    {
        'id': 'pdf2',
        'url': 'https://example.com/document2.pdf',
        'type': 'application/pdf',
        'name': 'document2.pdf'
    }
]

result = await langgraph_integration_service.process_with_langgraph(
    user=user,
    interaction=interaction,
    message=message,
    media_files=media_files
)
```

### 4. Analytical Processing

```python
# This will trigger analytical LangGraph workflow
message = "Please analyze and compare the latest developments in AI research"
media_files = []

result = await langgraph_integration_service.process_with_langgraph(
    user=user,
    interaction=interaction,
    message=message,
    media_files=media_files
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for Gemini models
GOOGLE_API_KEY=your_google_api_key

# Model selection
LLM_MODEL=gemini  # or gpt

# Optional: Disable thinking for simple tasks
DISABLE_THINKING=false
```

### ThinkingConfig Settings

**Automatic Detection:**

- **Simple tasks**: No thinking config
- **Moderate tasks**: Basic thinking
- **Complex tasks**: Enhanced thinking
- **Analytical tasks**: Maximum thinking budget

**Manual Override:**

```python
# Force thinking for specific tasks
thinking_config = ThinkingConfigManager.get_thinking_config(
    ComplexityLevel.ANALYTICAL,
    "complex_analysis"
)
```

## 📊 Workflow Decision Matrix

| Input Type | PDFs | Links | Analytical | Workflow  |
| ---------- | ---- | ----- | ---------- | --------- |
| Text only  | ❌   | ❌    | ❌         | Standard  |
| Text only  | ❌   | ❌    | ✅         | LangGraph |
| Text only  | ❌   | ✅    | ❌         | LangGraph |
| Text only  | ❌   | ✅    | ✅         | LangGraph |
| With PDFs  | ✅   | ❌    | ❌         | LangGraph |
| With PDFs  | ✅   | ❌    | ✅         | LangGraph |
| With PDFs  | ✅   | ✅    | ❌         | LangGraph |
| With PDFs  | ✅   | ✅    | ✅         | LangGraph |

## 🎨 UI Integration

### WebSocket Message Types

**Thinking Steps:**

```json
{
  "type": "thinking",
  "content": "🔍 Analyzing your request...",
  "thinking_steps": ["🔍 Analyzing your request..."],
  "timestamp": 1234567890.123
}
```

**Response Chunks:**

```json
{
  "type": "response",
  "content": "Based on my analysis...",
  "timestamp": 1234567890.123,
  "elapsed_time": 1.5,
  "chunk_number": 1
}
```

**Workflow Status:**

```json
{
  "type": "workflow",
  "workflow_type": "langgraph",
  "total_tokens": 1500,
  "thinking_steps": [
    "🔍 Analyzing...",
    "📄 Processing PDFs...",
    "🌐 Searching web..."
  ]
}
```

### Frontend Implementation

```javascript
// Handle thinking steps
websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "thinking") {
    // Display thinking steps
    updateThinkingDisplay(data.thinking_steps);
  } else if (data.type === "response") {
    // Display response content
    appendToResponse(data.content);
  }
};
```

## 🧪 Testing

### Run Test Suite

```bash
cd server
python test_langgraph_workflow.py
```

### Test Scenarios

1. **Simple Text Processing** - Should use standard processing
2. **PDF Processing** - Should trigger LangGraph workflow
3. **Link Processing** - Should trigger LangGraph workflow
4. **Hybrid Processing** - Should trigger comprehensive workflow
5. **Analytical Processing** - Should trigger analytical workflow

## 🔍 Monitoring and Debugging

### Workflow Logs

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor workflow execution
workflow_result = await langgraph_workflow_service.execute_workflow(...)
print(f"Workflow state: {workflow_result.get('workflow_state')}")
print(f"Thinking steps: {workflow_result.get('thinking_steps')}")
```

### Performance Metrics

- **Processing Time**: Track workflow execution time
- **Token Usage**: Monitor token consumption
- **Thinking Steps**: Count and analyze thinking steps
- **Success Rate**: Track workflow success/failure rates

## 🚨 Error Handling

### Fallback Strategy

1. **LangGraph Workflow Fails** → Fallback to standard processing
2. **ThinkingConfig Unavailable** → Continue without thinking
3. **Web Search Fails** → Continue with available sources
4. **PDF Processing Fails** → Continue with other sources

### Error Messages

```python
# Workflow errors
{
    "success": False,
    "error": "Workflow execution failed: [error details]",
    "fallback_used": True
}

# Thinking errors
{
    "success": True,
    "result": "Response content",
    "thinking_steps": ["⚠️ Thinking unavailable"],
    "thinking_disabled": True
}
```

## 📈 Performance Optimization

### Parallel Processing

- PDFs processed in parallel
- Web search runs concurrently
- Context retrieval optimized
- Source integration streamlined

### Caching Strategy

- Input analysis cached
- Thinking config cached
- Workflow state persisted
- Results cached for reuse

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Routing** - More sophisticated workflow selection
2. **Custom Workflows** - User-defined workflow templates
3. **Performance Analytics** - Detailed workflow metrics
4. **A/B Testing** - Workflow optimization testing
5. **Multi-language Support** - International workflow support

### Integration Opportunities

1. **External APIs** - Additional data sources
2. **Custom Models** - Specialized processing models
3. **Workflow Templates** - Predefined workflow patterns
4. **Real-time Collaboration** - Multi-user workflows

## 📚 API Reference

### LangGraphWorkflowService

```python
class LangGraphWorkflowService:
    async def execute_workflow(
        self,
        message: str,
        media_files: List[Dict[str, str]],
        user_id: str,
        interaction_id: str
    ) -> Dict[str, Any]
```

### LangGraphIntegrationService

```python
class LangGraphIntegrationService:
    async def process_with_langgraph(
        self,
        user: User,
        interaction: Optional[Interaction],
        message: Optional[str],
        media_files: Optional[List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]

    async def stream_workflow_with_thinking(
        self,
        user: User,
        interaction: Optional[Interaction],
        message: Optional[str],
        media_files: Optional[List[Dict[str, str]]],
        websocket=None
    )
```

### ThinkingConfigManager

```python
class ThinkingConfigManager:
    @staticmethod
    def should_use_thinking(complexity: ComplexityLevel, task_type: str) -> bool

    @staticmethod
    def get_thinking_config(complexity: ComplexityLevel, task_type: str) -> Optional[Dict[str, Any]]
```

## 🎯 Best Practices

### 1. Workflow Design

- Keep workflows focused and specific
- Use clear state transitions
- Implement proper error handling
- Monitor performance metrics

### 2. ThinkingConfig Usage

- Use automatic detection when possible
- Override only when necessary
- Monitor thinking budget usage
- Test with different complexity levels

### 3. UI Integration

- Display thinking steps progressively
- Provide clear status updates
- Handle errors gracefully
- Optimize for user experience

### 4. Performance

- Use parallel processing where possible
- Implement proper caching
- Monitor resource usage
- Optimize workflow paths

## 🔧 Troubleshooting

### Common Issues

1. **LangGraph Not Triggered**

   - Check input analysis logic
   - Verify complexity detection
   - Review workflow conditions

2. **ThinkingConfig Not Working**

   - Verify model compatibility
   - Check API key configuration
   - Review thinking budget settings

3. **WebSocket Errors**

   - Check message format
   - Verify error handling
   - Review connection status

4. **Performance Issues**
   - Monitor workflow execution time
   - Check parallel processing
   - Review caching strategy

### Debug Commands

```bash
# Test workflow orchestration
python test_langgraph_workflow.py

# Check model configuration
python -c "from app.config.langchain_config import StudyGuruConfig; print(StudyGuruConfig.MODELS._is_gemini_model())"

# Verify thinking config
python -c "from app.services.langgraph_workflow_service import ThinkingConfigManager; print(ThinkingConfigManager.should_use_thinking('complex', 'analytical_reasoning'))"
```

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the test suite
3. Examine workflow logs
4. Contact the development team

---

**Implementation Status**: ✅ Complete
**Last Updated**: 2024-12-19
**Version**: 1.0.0
