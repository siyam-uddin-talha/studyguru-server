# 🚀 Server Configuration Guide for 1K Concurrent Users

## 📊 Recommended Server Specifications

### **Minimum Configuration (Budget-Friendly)**

```
CPU: 4 cores (8 vCPUs)
RAM: 16 GB
Storage: 100 GB SSD
Network: 1 Gbps
```

### **Recommended Configuration (Production-Ready)**

```
CPU: 8 cores (16 vCPUs)
RAM: 32 GB
Storage: 200 GB SSD
Network: 1 Gbps+
```

### **High-Performance Configuration (Peak Load)**

```
CPU: 16 cores (32 vCPUs)
RAM: 64 GB
Storage: 500 GB SSD
Network: 10 Gbps
```

---

## 🔍 Resource Breakdown

### **Application Server (FastAPI)**

- **CPU**: 4-8 cores dedicated
  - FastAPI/Uvicorn is async and efficient
  - AI processing (LangChain, OpenAI, Gemini) is CPU-intensive
  - Vector database queries require CPU for embeddings
- **RAM**: 8-16 GB
  - Base application: ~2-3 GB
  - AI model processing: ~4-8 GB (per concurrent request)
  - Background tasks: ~1-2 GB
  - WebSocket/SSE connections: ~50-100 MB per 100 connections
  - Caching: ~1-2 GB

### **Database (MySQL)**

- **CPU**: 2-4 cores
- **RAM**: 4-8 GB
  - Connection pool: ~100-200 connections
  - Query cache: ~1-2 GB
  - Buffer pool: ~2-4 GB

### **Redis (Caching & Rate Limiting)**

- **CPU**: 1-2 cores
- **RAM**: 2-4 GB
  - Rate limiting data: ~100-200 MB
  - Cache storage: ~1-2 GB
  - Session data: ~500 MB

### **Vector Database (Zilliz/Milvus)**

- **CPU**: 2-4 cores
- **RAM**: 4-8 GB
  - Embedding storage: ~2-4 GB
  - Query processing: ~1-2 GB

---

## 📈 Concurrent User Capacity Analysis

### **Assumptions for 1K Users:**

- **Peak concurrent active**: 200-300 users (20-30% of total)
- **Simultaneous AI requests**: 50-100 (5-10% of total)
- **WebSocket/SSE connections**: 100-200 (10-20% of total)
- **Average request rate**: 2-5 requests per user per minute
- **Peak request rate**: 10-20 requests per user per minute

### **Resource Calculation:**

```
Active Users: 300
├── Regular API requests: 200 users × 3 req/min = 600 req/min
├── AI processing requests: 50 users × 1 req/min = 50 req/min
├── WebSocket connections: 150 active connections
└── Background tasks: 5-10 concurrent

Total Requests/Second: ~11 req/s (peak: ~50 req/s)
```

---

## ⚙️ Application Configuration

### **Uvicorn/Gunicorn Settings**

#### **Option 1: Gunicorn with Uvicorn Workers (Recommended)**

```bash
# Production deployment
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --log-level info \
  --access-logfile - \
  --error-logfile -
```

**Worker Calculation:**

```
Workers = (CPU cores × 2) + 1
For 8 cores: (8 × 2) + 1 = 17 workers (use 8-12 for stability)
For 4 cores: (4 × 2) + 1 = 9 workers (use 4-6 for stability)
```

#### **Option 2: Uvicorn Direct (Development/Testing)**

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop \
  --timeout-keep-alive 5 \
  --log-level info
```

### **Environment Variables (.env.production)**

```env
# Server
ENVIRONMENT=production
PORT=8000

# Database Connection Pool
DB_POOL_SIZE=50
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30

# Redis
REDIS_URL=redis://localhost:6379/0
ENABLE_RATE_LIMITING=true
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_WINDOW=60

# Background Tasks
MAX_CONCURRENT_BACKGROUND_TASKS=10
MAX_CONCURRENT_MESSAGE_TASKS=20
BACKGROUND_TASK_TIMEOUT=600

# Caching
ENABLE_MODEL_CACHING=true
ENABLE_CONTEXT_CACHING=true
CACHE_TTL=3600
CACHE_MAX_SIZE=5000

# AI Processing
# Adjust based on your AI provider rate limits
```

---

## 🗄️ Database Configuration

### **MySQL Connection Pool**

```python
# In your database config
pool_size = 50          # Base connections
max_overflow = 20       # Additional connections
pool_timeout = 30       # Seconds to wait for connection
pool_recycle = 3600     # Recycle connections after 1 hour
```

### **MySQL Server Settings (my.cnf)**

```ini
[mysqld]
max_connections = 200
innodb_buffer_pool_size = 4G
innodb_log_file_size = 512M
query_cache_size = 256M
query_cache_type = 1
thread_cache_size = 50
table_open_cache = 2000
```

---

## 🔴 Redis Configuration

### **Redis Server Settings (redis.conf)**

```conf
# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence (optional for cache)
save ""  # Disable RDB snapshots for cache-only use

# Performance
tcp-backlog 511
timeout 0
tcp-keepalive 300

# Threads (Redis 6+)
io-threads 4
io-threads-do-reads yes
```

### **Redis Memory Calculation**

```
Rate Limiting: ~50 MB (1000 users × 50 bytes per entry)
Cache: ~1-2 GB (model responses, context data)
Sessions: ~200 MB (active sessions)
Total: ~2-3 GB (with 20% headroom)
```

---

## 📦 Deployment Architecture Options

### **Option 1: Single Server (Budget)**

```
┌─────────────────────────────────┐
│   Application Server            │
│   - FastAPI (4 workers)         │
│   - MySQL                       │
│   - Redis                       │
│   - Vector DB (Zilliz Cloud)    │
└─────────────────────────────────┘
```

**Specs**: 8 CPU, 32 GB RAM, 200 GB SSD
**Cost**: ~$100-200/month
**Capacity**: 500-800 concurrent users

### **Option 2: Separated Services (Recommended)**

```
┌─────────────────┐     ┌──────────────┐
│  App Server     │────▶│   MySQL      │
│  (FastAPI)      │     │   (RDS)      │
│  8 CPU, 16 GB   │     │   4 CPU, 8GB │
└─────────────────┘     └──────────────┘
        │
        ▼
┌─────────────────┐     ┌──────────────┐
│   Redis         │     │  Vector DB   │
│   (ElastiCache) │     │  (Zilliz)    │
│   2 CPU, 4 GB   │     │   Cloud      │
└─────────────────┘     └──────────────┘
```

**Specs**:

- App: 8 CPU, 16 GB RAM
- DB: 4 CPU, 8 GB RAM (managed)
- Redis: 2 CPU, 4 GB (managed)
  **Cost**: ~$200-400/month
  **Capacity**: 1000+ concurrent users

### **Option 3: Load Balanced (High Availability)**

```
        ┌──────────────┐
        │ Load Balancer│
        └──────┬───────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌─────▼───┐
│ App 1  │          │ App 2   │
│ 4 CPU  │          │ 4 CPU   │
│ 16 GB  │          │ 16 GB   │
└───┬────┘          └─────┬───┘
    │                     │
    └──────────┬──────────┘
               │
        ┌──────▼──────┐
        │  MySQL      │
        │  (Primary)  │
        └─────────────┘
```

**Specs**: 2× App servers (4 CPU, 16 GB each)
**Cost**: ~$300-600/month
**Capacity**: 2000+ concurrent users

---

## 🔧 System Optimization

### **Linux System Settings**

```bash
# Increase file descriptor limits
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# Network optimizations
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 2048" >> /etc/sysctl.conf
echo "net.ipv4.ip_local_port_range = 10000 65535" >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

### **Process Management (systemd)**

```ini
# /etc/systemd/system/studyguru.service
[Unit]
Description=StudyGuru Pro API
After=network.target mysql.service redis.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/studyguru/server
Environment="PATH=/opt/studyguru/venv/bin"
ExecStart=/opt/studyguru/venv/bin/gunicorn app.main:app \
  --workers 8 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 📊 Monitoring & Scaling

### **Key Metrics to Monitor:**

1. **CPU Usage**: Should stay below 70% average
2. **RAM Usage**: Should stay below 80%
3. **Request Rate**: Requests per second
4. **Response Time**: P50, P95, P99 latencies
5. **Database Connections**: Active connections
6. **WebSocket Connections**: Active connections
7. **Error Rate**: Should be < 1%

### **Scaling Triggers:**

- **CPU > 70%** for 5+ minutes → Add workers or scale horizontally
- **RAM > 80%** → Increase server size or optimize memory usage
- **Response time P95 > 1s** → Optimize queries or add caching
- **Database connections > 80%** → Increase pool size or add read replicas

---

## 💰 Cost Estimates (Monthly)

### **AWS Example:**

- **EC2 (8 CPU, 32 GB)**: ~$150-200/month
- **RDS MySQL (db.t3.xlarge)**: ~$150/month
- **ElastiCache Redis (cache.t3.medium)**: ~$50/month
- **Zilliz Cloud**: ~$50-100/month
- **Total**: ~$400-500/month

### **DigitalOcean Example:**

- **Droplet (8 CPU, 32 GB)**: ~$160/month
- **Managed MySQL**: ~$90/month
- **Managed Redis**: ~$60/month
- **Zilliz Cloud**: ~$50-100/month
- **Total**: ~$360-410/month

### **VPS (Hetzner/OVH) Example:**

- **Dedicated Server (8 CPU, 32 GB)**: ~$50-80/month
- **Self-hosted MySQL/Redis**: Included
- **Zilliz Cloud**: ~$50-100/month
- **Total**: ~$100-180/month

---

## ✅ Quick Start Checklist

- [ ] Choose server size (8 CPU, 32 GB recommended)
- [ ] Set up MySQL with connection pooling
- [ ] Configure Redis for caching/rate limiting
- [ ] Set up Zilliz/Milvus vector database
- [ ] Configure Gunicorn with 8 workers
- [ ] Set environment variables
- [ ] Configure system limits (file descriptors)
- [ ] Set up process manager (systemd)
- [ ] Configure monitoring (CPU, RAM, requests)
- [ ] Set up logging and error tracking
- [ ] Test load with 100-200 concurrent users
- [ ] Monitor and adjust based on metrics

---

## 🎯 Final Recommendation

For **1K concurrent users**, I recommend:

**Server Configuration:**

- **CPU**: 8 cores (16 vCPUs)
- **RAM**: 32 GB
- **Storage**: 200 GB SSD
- **Network**: 1 Gbps

**Application Setup:**

- **Gunicorn workers**: 8-12 workers
- **Database pool**: 50 connections
- **Redis**: 2-4 GB memory
- **Background tasks**: 10-20 concurrent

This configuration provides:

- ✅ Comfortable headroom for 1K users
- ✅ Room for growth to 1.5K-2K users
- ✅ Good performance for AI processing
- ✅ Stable WebSocket/SSE connections
- ✅ Cost-effective (~$200-400/month)

---

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/settings.html)
- [MySQL Performance Tuning](https://dev.mysql.com/doc/refman/8.0/en/optimization.html)
- [Redis Memory Optimization](https://redis.io/docs/management/optimization/memory-optimization/)
