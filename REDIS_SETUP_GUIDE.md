# 🔴 Redis Setup Guide for StudyGuru Pro

## Quick Answer: Where to Get Redis?

**Development (Your Machine):**

- Install Redis locally (like your local MySQL)
- Or skip it and use in-memory backend

**Production (Cloud):**

- AWS ElastiCache (like RDS for MySQL)
- Redis Cloud (easiest)
- Upstash (serverless, free tier)
- DigitalOcean, Heroku, etc.

---

## 🖥️ Local Setup (Development)

### macOS

```bash
# Install Redis
brew install redis

# Start Redis (runs in background)
brew services start redis

# Test connection
redis-cli ping
# Should return: PONG

# Your Redis URL:
# redis://localhost:6379/0
```

### Ubuntu/Debian

```bash
# Install Redis
sudo apt-get update
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Test connection
redis-cli ping
# Should return: PONG

# Your Redis URL:
# redis://localhost:6379/0
```

### Docker (Any OS)

```bash
# Run Redis container
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Test connection
docker exec -it redis redis-cli ping
# Should return: PONG

# Your Redis URL:
# redis://localhost:6379/0
```

### Your `.env` File

```env
ENABLE_RATE_LIMITING=true
REDIS_URL=redis://localhost:6379/0
```

---

## ☁️ Cloud Setup (Production)

### 1. AWS ElastiCache (Recommended - Like RDS)

**Why Choose This?**

- ✅ Same ecosystem as RDS
- ✅ Integrated with AWS VPC
- ✅ Automatic backups and failover
- ✅ High availability

**Setup Steps:**

1. **Go to AWS Console**

   - Navigate to ElastiCache
   - Click "Create Redis cluster"

2. **Configuration**

   ```
   Cluster engine: Redis
   Location: AWS Cloud

   Multi-AZ: Enabled (for production)
   Engine version: 7.x
   Port: 6379
   Node type: cache.t3.micro (for testing)
            cache.t3.medium (for production)

   Number of replicas: 1 (for HA)
   ```

3. **Security**

   - VPC: Same as your application
   - Subnet: Private subnets
   - Security Group: Allow port 6379 from your app servers

4. **Get Connection Details**

   - Go to cluster details
   - Copy "Primary Endpoint"
   - Format: `your-cluster.abc123.0001.use1.cache.amazonaws.com:6379`

5. **Your `.env` File**

   ```env
   ENABLE_RATE_LIMITING=true

   # Without password (default)
   REDIS_URL=redis://your-cluster.abc123.0001.use1.cache.amazonaws.com:6379/0

   # With AUTH enabled (recommended)
   REDIS_URL=redis://:your-auth-token@your-cluster.abc123.0001.use1.cache.amazonaws.com:6379/0
   ```

**Pricing:**

- cache.t3.micro: ~$12/month
- cache.t3.small: ~$25/month
- cache.t3.medium: ~$50/month

---

### 2. Redis Cloud (Easiest Setup)

**Why Choose This?**

- ✅ Easiest setup (5 minutes)
- ✅ Free tier available (30MB)
- ✅ Managed by Redis Labs
- ✅ Works from anywhere

**Setup Steps:**

1. **Sign Up**

   - Go to https://redis.com/try-free/
   - Create account (free)

2. **Create Database**

   - Click "New Database"
   - Choose region (closest to your app)
   - Select plan:
     - Free: 30MB, 30 connections
     - Paid: From $5/month

3. **Get Connection Details**

   - Click on your database
   - Go to "Configuration"
   - Copy "Public endpoint"

4. **Your `.env` File**
   ```env
   ENABLE_RATE_LIMITING=true
   REDIS_URL=redis://default:your-password@redis-12345.c123.us-east-1-1.ec2.cloud.redislabs.com:12345/0
   ```

**Pricing:**

- Free: 30MB, perfect for rate limiting
- Paid: $5/month for 100MB

---

### 3. Upstash (Serverless - Best Free Tier)

**Why Choose This?**

- ✅ Best free tier (10,000 commands/day)
- ✅ Serverless (pay per use)
- ✅ No credit card for free tier
- ✅ Global edge caching

**Setup Steps:**

1. **Sign Up**

   - Go to https://upstash.com/
   - Create account (no credit card needed)

2. **Create Database**

   - Click "Create Database"
   - Choose region
   - Select "Free" plan

3. **Get Connection Details**

   - Click "Details"
   - Copy "Redis URL"

4. **Your `.env` File**
   ```env
   ENABLE_RATE_LIMITING=true
   REDIS_URL=rediss://:your-password@us1-charming-owl-12345.upstash.io:6379
   ```

**Pricing:**

- Free: 10,000 commands/day (enough for ~1000 users)
- Pay-as-you-go: $0.20 per 100K commands

---

### 4. DigitalOcean Managed Redis

**Why Choose This?**

- ✅ Simple pricing
- ✅ Good for DigitalOcean users
- ✅ Easy setup

**Setup Steps:**

1. **Go to DigitalOcean Console**

   - Navigate to Databases
   - Click "Create Database"

2. **Configuration**

   - Engine: Redis
   - Plan: Starting at $15/month (1GB)
   - Region: Closest to your app

3. **Get Connection Details**

   - Go to database overview
   - Copy "Public connection string"

4. **Your `.env` File**
   ```env
   ENABLE_RATE_LIMITING=true
   REDIS_URL=redis://default:your-password@db-redis-nyc1-12345-do-user-123456-0.b.db.ondigitalocean.com:25061/0
   ```

**Pricing:**

- Starts at $15/month (1GB RAM)

---

### 5. Railway (Dev-Friendly)

**Why Choose This?**

- ✅ Super easy deployment
- ✅ Free $5/month credit
- ✅ Deploy with one click

**Setup Steps:**

1. **Sign Up**

   - Go to https://railway.app/
   - Sign up with GitHub

2. **Create Redis Service**

   - New Project → Add Service → Redis
   - Deploys instantly

3. **Get Connection Details**

   - Click on Redis service
   - Copy "REDIS_URL" from variables

4. **Your `.env` File**
   ```env
   ENABLE_RATE_LIMITING=true
   REDIS_URL=redis://default:password@redis.railway.internal:6379
   ```

**Pricing:**

- Free: $5 credit/month
- Paid: Usage-based

---

## 🆚 Comparison Table

| Provider            | Free Tier       | Pricing   | Setup Difficulty | Best For               |
| ------------------- | --------------- | --------- | ---------------- | ---------------------- |
| **Local Redis**     | ✅ Free         | Free      | Easy             | Development            |
| **AWS ElastiCache** | ❌ No           | $12+/mo   | Medium           | Production + AWS users |
| **Redis Cloud**     | ✅ 30MB         | $5+/mo    | Very Easy        | Quick start            |
| **Upstash**         | ✅ 10K cmds/day | Pay-as-go | Very Easy        | Serverless apps        |
| **DigitalOcean**    | ❌ No           | $15+/mo   | Easy             | DO users               |
| **Railway**         | ✅ $5 credit    | Usage     | Very Easy        | Small projects         |

---

## 🎯 My Recommendation

### For You (StudyGuru Pro):

#### **Development:**

```bash
# Option 1: Local Redis (Recommended)
brew install redis
brew services start redis

# .env
REDIS_URL=redis://localhost:6379/0
```

```bash
# Option 2: Docker (Alternative)
docker run -d -p 6379:6379 redis:7-alpine

# .env
REDIS_URL=redis://localhost:6379/0
```

```env
# Option 3: Skip Redis (Simplest)
ENABLE_RATE_LIMITING=true
# No REDIS_URL - uses in-memory automatically
```

#### **Production:**

**If using AWS (same as your RDS):**

```
Use AWS ElastiCache
- Same VPC as your app
- Integrated with your infrastructure
- Reliable and scalable
```

**If want simplest setup:**

```
Use Upstash
- Generous free tier
- Serverless (no management)
- Pay only for what you use
```

**If on a budget:**

```
Use In-Memory Backend
- No Redis needed
- Works perfectly for single server
- Zero cost
```

---

## 🔧 Configuration Examples

### With Redis (Distributed)

```env
# .env.production
ENABLE_RATE_LIMITING=true
REDIS_URL=redis://:password@your-redis-host:6379/0

# Other settings
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_WINDOW=60
```

### Without Redis (In-Memory)

```env
# .env.production
ENABLE_RATE_LIMITING=true
# No REDIS_URL - automatically uses in-memory

# Other settings
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_WINDOW=60
```

---

## 🧪 Testing Your Setup

### Test Local Redis:

```bash
# Ping test
redis-cli ping

# Set a value
redis-cli SET mykey "Hello"

# Get the value
redis-cli GET mykey

# Check info
redis-cli INFO
```

### Test Cloud Redis:

```bash
# Using redis-cli
redis-cli -h your-redis-host -p 6379 -a your-password ping

# Using Python
python -c "
import redis
r = redis.from_url('redis://:password@host:6379/0')
print(r.ping())  # Should print True
"
```

### Test Rate Limiting:

```bash
# Start server
python -m app.main

# Check logs for:
# [INFO] Rate limiting initialized with backend: Redis
# or
# [INFO] Rate limiting initialized with backend: In-Memory
```

---

## 💡 Pro Tips

1. **Development**: Use local Redis or skip it (in-memory)
2. **Single Server Production**: In-memory works great, no Redis needed
3. **Multiple Servers**: Must use Redis for shared state
4. **AWS Users**: Use ElastiCache (same ecosystem)
5. **Quick Start**: Use Upstash (best free tier)
6. **Cost-Conscious**: Use in-memory (zero cost)

---

## ❓ FAQ

**Q: Do I need Redis?**
A: No! In-memory backend works great for single-server deployments.

**Q: When should I use Redis?**
A: When you have multiple server instances and need shared rate limits.

**Q: Is Redis expensive?**
A: Not at all. For rate limiting, even the smallest plan works (30MB is plenty).

**Q: Can I switch between Redis and in-memory?**
A: Yes! Just change `REDIS_URL` in `.env` and restart.

**Q: What if Redis goes down?**
A: The system automatically falls back to allowing requests (fail-open design).

---

## 🚀 Quick Start

**Want to start now? Do this:**

```bash
# 1. Skip Redis for now (use in-memory)
echo "ENABLE_RATE_LIMITING=true" >> .env

# 2. Start your server
python -m app.main

# 3. Check logs
# Should see: "Rate limiting initialized with backend: In-Memory"

# 4. Later, when you need Redis:
# Install locally: brew install redis
# Or use Upstash: https://upstash.com (free tier)
# Then add: REDIS_URL=redis://...
```

---

**You're all set! Rate limiting works great with or without Redis.** 🎉
