# Enhanced RAG System Test Suite

This directory contains comprehensive tests for the enhanced RAG (Retrieval Augmented Generation) system.

## Test Files

### Core System Tests

- **`test_basic.py`** - Basic API and health check tests
- **`test_langchain.py`** - LangChain service integration tests
- **`test_vector_db.py`** - Vector database connection and operations tests

### Enhanced RAG System Tests

- **`test_vector_optimization.py`** - Hybrid search and query expansion tests
- **`test_document_integration.py`** - Document processing and analysis tests
- **`test_real_time_context.py`** - Real-time context update system tests
- **`test_context_testing_framework.py`** - Context usage validation framework tests
- **`test_performance_monitoring.py`** - Performance monitoring and optimization tests

### Test Runner

- **`run_all_tests.py`** - Executable script to run all tests

## Running Tests

### Run All Tests

```bash
# From the server directory
python3 test/run_all_tests.py

# Or make it executable and run directly
chmod +x test/run_all_tests.py
./test/run_all_tests.py
```

### Run Individual Tests

```bash
# From the server directory
python3 test/test_basic.py
python3 test/test_langchain.py
python3 test/test_vector_db.py
python3 test/test_vector_optimization.py
python3 test/test_document_integration.py
python3 test/test_real_time_context.py
python3 test/test_context_testing_framework.py
python3 test/test_performance_monitoring.py
```

### Run Tests from Test Directory

```bash
# From the test directory
cd test/
python3 test_basic.py
python3 test_langchain.py
# ... etc
```

## Test Categories

### 1. **Basic System Tests**

- API endpoint functionality
- Database connectivity
- Basic service initialization

### 2. **LangChain Integration Tests**

- Model configuration
- Guardrail functionality
- Document analysis
- Vector store operations
- Points calculation

### 3. **Vector Database Tests**

- Connection validation
- Embedding generation
- Similarity search
- User isolation
- Data cleanup

### 4. **Enhanced RAG System Tests**

- **Vector Optimization**: Hybrid search, query expansion, metadata enhancement
- **Document Integration**: Comprehensive document processing, chunking, analysis
- **Real-time Context**: Queue-based updates, consistency checks, background processing
- **Context Testing**: Validation framework, performance metrics, quality assessment
- **Performance Monitoring**: Metrics collection, alerting, optimization recommendations

## Prerequisites

Before running tests, ensure:

1. **Database Setup**: MySQL database is running and accessible
2. **Vector Database**: Zilliz/Milvus is configured and running
3. **Environment Variables**: All required API keys and configuration are set
4. **Dependencies**: All Python packages are installed

## Test Data

Tests use mock data and test users to avoid affecting production data:

- Test user IDs: `test_user_123`, `test_user_456`
- Test interaction IDs: `test_interaction_456`, etc.
- Test media IDs: `test_media_789`, etc.

## Expected Results

All tests should pass for the enhanced RAG system to be considered fully functional. The test suite validates:

- ✅ **Context Retrieval**: Multi-level context retrieval works correctly
- ✅ **Semantic Summarization**: Enhanced summarization with validation
- ✅ **Vector Search**: Hybrid search with query expansion
- ✅ **Document Processing**: Comprehensive document analysis
- ✅ **Real-time Updates**: Background processing and consistency
- ✅ **Performance Monitoring**: Metrics collection and optimization
- ✅ **Testing Framework**: Context usage validation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
2. **Database Connection**: Check database credentials and connectivity
3. **Vector Database**: Verify Zilliz/Milvus is running and accessible
4. **API Keys**: Ensure all required API keys are configured

### Debug Mode

For detailed debugging, you can modify test files to include more verbose output or run individual test functions.

## Contributing

When adding new tests:

1. Follow the existing naming convention: `test_[service_name].py`
2. Include comprehensive docstrings
3. Use mock data to avoid affecting production
4. Add the new test to `run_all_tests.py`
5. Update this README with test descriptions
