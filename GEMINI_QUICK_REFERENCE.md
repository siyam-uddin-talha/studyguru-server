# Gemini Integration - Quick Reference

## 🚀 Quick Start

### 1. Set Environment Variables

```bash
# For Gemini
LLM_MODEL=gemini
GOOGLE_API_KEY=your_google_api_key_here

# For GPT (default)
LLM_MODEL=gpt
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Test Integration

```bash
cd server
source .venv/bin/activate
python test_gemini_integration.py
```

## 📋 Model Mapping

| Use Case              | Gemini Model         | GPT Model              |
| --------------------- | -------------------- | ---------------------- |
| **Main Chat**         | gemini-2.5-pro       | gpt-4o                 |
| **Vision**            | gemini-2.5-pro       | gpt-4o                 |
| **Guardrails**        | gemini-2.5-flash     | gpt-4o-mini            |
| **Complex Reasoning** | gemini-2.5-pro       | gpt-4o                 |
| **Embeddings**        | models/embedding-001 | text-embedding-3-small |
| **Titles**            | gemini-2.5-flash     | gpt-4o-mini            |

## 🔧 Configuration

### Environment Variables

- `LLM_MODEL`: "gemini" or "gpt"
- `GOOGLE_API_KEY`: Your Google AI Studio API key
- `OPENAI_API_KEY`: Your OpenAI API key

### Vector Store

- **Single Collection**: 1536 dimensions (common for both models)
- **Gemini**: 768D padded to 1536D automatically
- **GPT**: Native 1536 dimensions
- **Fast Search**: Single collection for optimal performance

## ✅ Features Supported

- ✅ Chat conversations
- ✅ Document analysis (images + text)
- ✅ MCQ generation
- ✅ Guardrail checks
- ✅ Title generation
- ✅ Embeddings & similarity search
- ✅ Complex reasoning
- ✅ Streaming responses

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_gemini_integration.py
```

Expected output:

```
🎉 All tests passed! Gemini integration is working correctly.
```

## 📚 Documentation

- **Full Guide**: `GEMINI_INTEGRATION_GUIDE.md`
- **Implementation**: `GEMINI_IMPLEMENTATION_SUMMARY.md`
- **Test Suite**: `test_gemini_integration.py`

## 🆘 Troubleshooting

### Common Issues:

1. **"No API key found"** → Set `GOOGLE_API_KEY` in `.env`
2. **"Model not found"** → Check Google AI Studio access
3. **"Dimension mismatch"** → Recreate vector collections when switching

### Get Google API Key:

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with Google account
3. Click "Get API Key"
4. Create new key
5. Add to `.env` file

## 🎯 Benefits

- **Cost-effective**: Gemini often cheaper than GPT-4o
- **Fast**: Flash models for quick responses
- **Flexible**: Easy switching between providers
- **Compatible**: Works with all existing features
