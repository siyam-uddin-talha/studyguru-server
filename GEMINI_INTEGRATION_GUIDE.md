# Gemini Integration Guide for StudyGuru Pro

This guide explains how to use Google's Gemini models in StudyGuru Pro alongside the existing GPT models.

## Overview

StudyGuru Pro now supports both OpenAI GPT models and Google Gemini models. You can switch between them by changing a single environment variable.

## Model Mapping

### Gemini Models Used:

- **gemini-2.0-flash-exp**: Main chat and vision model (equivalent to GPT-4o)
- **gemini-1.5-flash**: Fast, cost-effective model for guardrails and titles (equivalent to GPT-4o-mini)
- **models/embedding-001**: Embeddings model (fast and cost-efficient)

### GPT Models (Fallback):

- **gpt-4.1**: Main chat and vision model
- **gpt-4.1-mini**: Fast, cost-effective model for guardrails and titles
- **text-embedding-3-small**: Embeddings model

## Configuration

### 1. Environment Variables

Add these to your `.env` file:

```bash
# Choose your LLM provider
LLM_MODEL=gemini  # or "gpt" for OpenAI models

# Google Gemini API Key (required for Gemini)
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API Key (required for GPT models)
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Getting Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" in the left sidebar
4. Create a new API key
5. Copy the key and add it to your `.env` file

## Usage

### Switching Between Models

Simply change the `LLM_MODEL` environment variable:

```bash
# Use Gemini models
LLM_MODEL=gemini

# Use GPT models (default)
LLM_MODEL=gpt
```

### Model Selection Logic

The system automatically selects the appropriate model based on the `LLM_MODEL` setting:

```python
# In StudyGuruModels class
@staticmethod
def _is_gemini_model() -> bool:
    return settings.LLM_MODEL.lower() == "gemini"
```

## Features Supported

### ✅ Fully Supported with Gemini:

- **Chat Conversations**: Natural language interactions
- **Document Analysis**: Image and text document processing
- **MCQ Generation**: Multiple choice question creation
- **Guardrail Checks**: Content safety validation
- **Title Generation**: Automatic conversation titling
- **Embeddings**: Vector embeddings for similarity search
- **Complex Reasoning**: Advanced problem solving

### 🔄 Automatic Model Selection:

- **Chat Model**: `gemini-2.0-flash-exp` (Gemini) vs `gpt-4.1` (GPT)
- **Vision Model**: `gemini-2.0-flash-exp` (Gemini) vs `gpt-4.1` (GPT)
- **Guardrail Model**: `gemini-1.5-flash` (Gemini) vs `gpt-4.1-mini` (GPT)
- **Embeddings**: `models/embedding-001` (Gemini) vs `text-embedding-3-small` (GPT)

## Vector Store Compatibility

### Embedding Dimensions:

- **Gemini**: 768 dimensions
- **OpenAI**: 1536 dimensions

### Collection Structure:

The system uses a **single collection** with dimension compatibility:

- **Single Collection**: `{ZILLIZ_COLLECTION}` (1536 dimensions)
- **GPT Embeddings**: Native 1536 dimensions
- **Gemini Embeddings**: 768 dimensions padded to 1536 dimensions

### Embedding Compatibility:

- **Automatic Padding**: Gemini embeddings (768D) are automatically padded to 1536D
- **Zero Padding**: Padding uses zeros to maintain semantic meaning
- **Single Search**: Fast single-collection search for all models
- **No Migration**: Existing data remains compatible

### Migration Notes:

- **No migration needed**: Single collection works with both models
- **Automatic compatibility**: Embeddings are padded automatically
- **Fast performance**: Single collection search is faster than cross-model search

## Testing

Run the integration test to verify everything works:

```bash
cd server
python test_gemini_integration.py
```

This test will:

1. Verify model instantiation
2. Test chat functionality
3. Test MCQ generation
4. Test title generation
5. Test GPT fallback

## Performance Comparison

### Gemini Advantages:

- **Cost**: Generally more cost-effective than GPT-4o
- **Speed**: Fast response times, especially with Flash models
- **Multimodal**: Excellent vision capabilities
- **Context**: Large context windows

### GPT Advantages:

- **Maturity**: More established and tested
- **JSON Output**: Better structured output formatting
- **Consistency**: More predictable responses

## Troubleshooting

### Common Issues:

1. **"No API key found"**

   - Ensure `GOOGLE_API_KEY` is set in your `.env` file
   - Verify the API key is valid and has proper permissions

2. **"Model not found"**

   - Check that you're using the correct model names
   - Ensure your Google AI Studio account has access to the models

3. **"Embedding dimension mismatch"**

   - This occurs when switching between GPT and Gemini
   - You may need to recreate your vector store collections

4. **"JSON parsing errors"**
   - Gemini models may have different JSON output formatting
   - The system includes robust JSON parsing to handle this

### Debug Mode:

Enable debug logging to see which models are being used:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Rate Limits

### Gemini Limits:

- **Free Tier**: 15 requests per minute
- **Paid Tier**: Higher limits available
- **Rate Limiting**: Built into the LangChain integration

### Best Practices:

- Use appropriate timeouts
- Implement retry logic for rate limit errors
- Monitor usage in Google AI Studio dashboard

## Security Considerations

1. **API Key Security**:

   - Never commit API keys to version control
   - Use environment variables or secure key management
   - Rotate keys regularly

2. **Content Safety**:
   - Both models include built-in safety filters
   - Additional guardrail checks are implemented
   - Monitor content for compliance

## Migration Guide

### From GPT to Gemini:

1. **Add Google API Key**:

   ```bash
   GOOGLE_API_KEY=your_key_here
   ```

2. **Update Environment**:

   ```bash
   LLM_MODEL=gemini
   ```

3. **Test Integration**:

   ```bash
   python test_gemini_integration.py
   ```

4. **Update Vector Store** (if needed):
   - Consider recreating collections for optimal performance
   - Existing embeddings may not be compatible

### From Gemini to GPT:

1. **Update Environment**:

   ```bash
   LLM_MODEL=gpt
   ```

2. **Ensure OpenAI Key**:

   ```bash
   OPENAI_API_KEY=your_key_here
   ```

3. **Test Integration**:
   ```bash
   python test_gemini_integration.py
   ```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the test output for specific errors
3. Verify your API keys and permissions
4. Check Google AI Studio dashboard for usage and limits

## Changelog

### Version 1.0.0

- Initial Gemini integration
- Support for gemini-2.0-flash-exp and gemini-1.5-flash
- Automatic model selection based on LLM_MODEL environment variable
- Vector store compatibility with different embedding dimensions
- Comprehensive testing suite
