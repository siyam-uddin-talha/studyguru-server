# Zilliz Cloud Setup Guide

## 🚀 Quick Setup

### 1. Get Your Credentials from Zilliz Cloud

1. Go to [https://cloud.zilliz.com/](https://cloud.zilliz.com/)
2. Sign in to your account
3. Navigate to your cluster
4. Find your **URI** and **Token** (API Key)

### 2. Configure Your Environment

#### Option A: Interactive Setup (Recommended)

```bash
cd server
python3 setup_zilliz.py
```

#### Option B: Manual Setup

Add these to your `.env` file:

```bash
# Zilliz Vector Database Configuration
ZILLIZ_URI=https://your-cluster.zillizcloud.com
ZILLIZ_TOKEN=your-api-token-here
ZILLIZ_COLLECTION=document_embeddings
ZILLIZ_DIMENSION=1536
ZILLIZ_INDEX_METRIC=IP
ZILLIZ_CONSISTENCY_LEVEL=Bounded
```

### 3. Test Your Setup

```bash
# Test vector database functionality
python3 test_vector_db.py

# Test full LangChain implementation
python3 test_langchain.py
```

## 📋 What You Need from Zilliz Cloud

### URI Format

Your URI should look like:

```
https://your-cluster-id.zillizcloud.com
```

### Token

- Go to your cluster settings
- Find "API Keys" or "Tokens"
- Copy your API key/token

### Collection Name

- You can use the default: `document_embeddings`
- Or create a custom name like: `studyguru_embeddings`

## 🧪 Testing Your Setup

The `test_vector_db.py` script will test:

1. **✅ Connection** - Verify you can connect to Zilliz Cloud
2. **✅ Embeddings** - Test embedding generation and storage
3. **✅ Search** - Test similarity search functionality
4. **✅ User Isolation** - Verify users only see their own data
5. **✅ Performance** - Check response times and accuracy

## 🔧 Configuration Options

### Dimension

- **1536** - For `text-embedding-3-small` (recommended)
- **3072** - For `text-embedding-3-large`

### Index Metric

- **IP** - Inner Product (recommended for embeddings)
- **L2** - Euclidean distance
- **COSINE** - Cosine similarity

### Consistency Level

- **Bounded** - Good balance of performance and consistency
- **Strong** - Highest consistency, slower performance
- **Session** - Fastest, eventual consistency

## 🚨 Troubleshooting

### Connection Issues

```bash
# Check your credentials
echo $ZILLIZ_URI
echo $ZILLIZ_TOKEN
```

### Permission Issues

- Make sure your token has read/write permissions
- Check if your cluster is active and running

### Collection Issues

- The collection will be created automatically
- Make sure you have permission to create collections

## 📊 Expected Test Results

When everything is working, you should see:

```
🧪 Zilliz Vector Database Test Suite
============================================================

🔗 Testing Zilliz Cloud Connection...
📋 Configuration:
   URI: https://your-cluster...
   Token: ✅ Set
   Collection: document_embeddings
   Dimension: 1536
   Metric: IP

✅ Vector store initialized successfully!

🧮 Testing Embedding Operations...
📝 Testing with 3 documents...
   ✅ Document 'Introduction to Machine Learning' embedded successfully
   ✅ Document 'Deep Learning Fundamentals' embedded successfully
   ✅ Document 'Calculus Derivatives' embedded successfully

📊 Results: 3/3 documents embedded successfully

🔍 Testing Similarity Search...
🔎 Test 1: 'What is machine learning?'
   ✅ Found 2 results:
      1. Introduction to Machine Learning (score: 0.892)
      2. Deep Learning Fundamentals (score: 0.756)

🎉 Vector Database Test Complete!
```

## 🎯 Next Steps

Once your vector database is working:

1. **Deploy to Production** - Use the same credentials in your production environment
2. **Monitor Usage** - Check your Zilliz Cloud dashboard for usage metrics
3. **Scale as Needed** - Upgrade your Zilliz plan if you need more capacity
4. **Backup Strategy** - Consider setting up regular backups of your vector data

## 💡 Tips

- **Start Small** - Begin with a free tier to test functionality
- **Monitor Costs** - Vector databases can be expensive at scale
- **Use Indexes** - Proper indexing improves search performance
- **Batch Operations** - Insert multiple documents at once for better performance

Your vector database is now ready to power StudyGuru Pro's AI capabilities! 🚀
