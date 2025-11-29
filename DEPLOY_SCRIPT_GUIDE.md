# 📖 Deploy Script Guide - What `deploy.sh` Does

## 🎯 Overview

The `deploy.sh` script is an **interactive setup tool** that automates the initial server configuration for deploying StudyGuru Pro in production. It helps you prepare your Linux server with all the necessary system-level configurations.

---

## 🔍 What It Does (Step by Step)

### **1. Initial Checks**

```bash
# Checks if script is run as root (required for system changes)
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root or with sudo"
    exit 1
fi
```

- **Purpose**: Ensures you have administrator privileges
- **Why**: Installing packages and modifying system files requires root access

### **2. OS Detection**

```bash
# Detects your Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi
```

- **Purpose**: Identifies if you're using Ubuntu/Debian or CentOS/RHEL
- **Why**: Different Linux distributions use different package managers (`apt` vs `yum`)

---

## 🛠️ Functions Explained

### **Function 1: `install_dependencies()`**

**What it does:**

- Installs all required system packages for your application

**For Ubuntu/Debian:**

```bash
apt-get install -y \
    python3.11          # Python runtime
    python3.11-venv     # Virtual environment support
    python3-pip         # Python package manager
    mysql-server        # MySQL database server
    redis-server        # Redis cache/rate limiting
    nginx               # Web server (reverse proxy)
    supervisor          # Process manager (alternative to systemd)
    build-essential    # Compiler tools for building Python packages
    libssl-dev          # SSL library (for secure connections)
    libffi-dev          # Foreign function interface (for Python packages)
    python3-dev         # Python development headers
```

**For CentOS/RHEL:**

```bash
yum install -y \
    python3.11
    mariadb-server      # MySQL alternative
    redis
    nginx
    # ... similar packages
```

**When to use:** First time setting up a fresh server

---

### **Function 2: `configure_system_limits()`**

**What it does:**

- Increases system resource limits to handle high traffic

**File Descriptor Limits** (`/etc/security/limits.conf`):

```conf
* soft nofile 65535      # Max open files per process (soft limit)
* hard nofile 65535      # Max open files per process (hard limit)
www-data soft nofile 65535  # For web server user
www-data hard nofile 65535
```

- **Why**: Your app needs to handle many concurrent connections (WebSocket, HTTP, database)
- **Default**: Usually 1024, which is too low for 1K users

**Network Optimizations** (`/etc/sysctl.conf`):

```conf
net.core.somaxconn = 1024              # Max pending connections
net.ipv4.tcp_max_syn_backlog = 2048    # Max SYN backlog
net.ipv4.ip_local_port_range = 10000 65535  # Available ports
net.ipv4.tcp_tw_reuse = 1              # Reuse TIME_WAIT sockets
net.ipv4.tcp_fin_timeout = 30          # Faster connection cleanup
```

- **Why**: Optimizes network performance for high concurrent connections
- **Impact**: Prevents "too many open files" errors and improves connection handling

**When to use:** Always recommended for production servers

---

### **Function 3: `configure_mysql()`**

**What it does:**

- Creates optimized MySQL configuration for your application

**Creates file:** `/etc/mysql/conf.d/studyguru.cnf`

```ini
[mysqld]
max_connections = 200              # Allow 200 concurrent connections
innodb_buffer_pool_size = 4G      # Memory for InnoDB (4GB RAM for MySQL)
innodb_log_file_size = 512M       # Transaction log size
query_cache_size = 256M           # Cache query results
query_cache_type = 1              # Enable query cache
thread_cache_size = 50            # Reuse threads
table_open_cache = 2000           # Cache open tables
```

**What each setting does:**

- `max_connections`: Allows your app to have 200 database connections simultaneously
- `innodb_buffer_pool_size`: Uses 4GB RAM to cache database data (faster queries)
- `query_cache_size`: Caches frequently used queries (reduces database load)

**When to use:** If you're running MySQL on the same server

**Note:** After running, you need to restart MySQL:

```bash
systemctl restart mysql
```

---

### **Function 4: `configure_redis()`**

**What it does:**

- Optimizes Redis configuration for caching and rate limiting

**Modifies:** `/etc/redis/redis.conf`

```conf
maxmemory 2gb                      # Use max 2GB RAM
maxmemory-policy allkeys-lru       # Evict least recently used keys when full
io-threads 4                      # Use 4 threads for I/O (Redis 6+)
io-threads-do-reads yes           # Enable threaded reads
```

**What each setting does:**

- `maxmemory 2gb`: Limits Redis to 2GB (prevents it from using all RAM)
- `maxmemory-policy allkeys-lru`: Automatically removes old cache when full
- `io-threads 4`: Uses multiple threads for better performance

**Safety:** Creates a backup of original config before modifying

**When to use:** If you're running Redis on the same server

**Note:** After running, you need to restart Redis:

```bash
systemctl restart redis
```

---

### **Function 5: `create_app_user()`**

**What it does:**

- Creates a dedicated system user for running your application

```bash
useradd -r -s /bin/false www-data
```

**What this means:**

- `-r`: Creates a system user (no home directory)
- `-s /bin/false`: User cannot login (security best practice)
- `www-data`: Standard name for web applications

**Why:**

- Security: Don't run your app as root
- Isolation: Separate user for your application
- Permissions: Easier to manage file permissions

**When to use:** If the `www-data` user doesn't exist (common on CentOS)

---

## 🎮 How to Use the Script

### **Step 1: Make it executable**

```bash
chmod +x deploy.sh
```

### **Step 2: Run with sudo**

```bash
sudo ./deploy.sh
```

### **Step 3: Interactive Prompts**

The script will ask you questions:

```
🚀 StudyGuru Pro - Production Deployment Setup
==============================================
📦 Detected OS: ubuntu

Install system dependencies? (y/n) y
📦 Installing system dependencies...
✅ Dependencies installed

Configure system limits? (y/n) y
⚙️  Configuring system limits...
✅ System limits configured

Configure MySQL? (y/n) y
🗄️  Configuring MySQL...
✅ MySQL configuration created at /etc/mysql/conf.d/studyguru.cnf
⚠️  Please restart MySQL: systemctl restart mysql

Configure Redis? (y/n) y
🔴 Configuring Redis...
✅ Redis configuration updated
⚠️  Please restart Redis: systemctl restart redis

Create application user? (y/n) y
👤 Creating application user...
✅ Application user ready

✅ Setup complete!
```

### **Step 4: Follow Next Steps**

After the script completes, it shows you what to do next:

1. Configure environment variables
2. Install Python dependencies
3. Set up systemd service
4. Start your application

---

## ⚠️ Important Notes

### **What the Script DOES:**

✅ Installs system packages  
✅ Configures system limits  
✅ Sets up MySQL/Redis configs  
✅ Creates application user  
✅ Makes your server ready for deployment

### **What the Script DOES NOT:**

❌ Install your Python application code  
❌ Install Python packages (pip install)  
❌ Set up your database schema  
❌ Configure environment variables  
❌ Deploy your application code  
❌ Set up SSL certificates  
❌ Configure Nginx reverse proxy

**Why:** These are application-specific and require your code/configurations.

---

## 🔄 Typical Workflow

```bash
# 1. Run the deployment script (one-time setup)
sudo ./deploy.sh

# 2. Restart services if configured
sudo systemctl restart mysql
sudo systemctl restart redis

# 3. Set up your application
cd /opt/studyguru/server
cp env.production.example .env.production
# Edit .env.production with your settings

# 4. Install Python dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Set up systemd service
sudo cp systemd/studyguru.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable studyguru
sudo systemctl start studyguru

# 6. Check if it's running
sudo systemctl status studyguru
```

---

## 🎯 When to Use This Script

### **✅ Use it when:**

- Setting up a fresh server for the first time
- You want to automate system configuration
- You're not familiar with Linux server setup
- You want consistent configuration across servers

### **❌ Skip it if:**

- You're using managed services (AWS RDS, ElastiCache)
- You prefer manual configuration
- You're using Docker/containers
- You're deploying to a platform (Heroku, Railway, etc.)

---

## 🔧 Customization

You can modify the script to:

- Add more packages
- Change MySQL/Redis settings
- Add additional system configurations
- Support more Linux distributions

Just edit the functions in `deploy.sh` before running it.

---

## 📊 Summary

| Function                    | What It Does                               | When Needed             |
| --------------------------- | ------------------------------------------ | ----------------------- |
| `install_dependencies()`    | Installs Python, MySQL, Redis, etc.        | Fresh server setup      |
| `configure_system_limits()` | Increases file descriptors, network limits | Always for production   |
| `configure_mysql()`         | Optimizes MySQL for high traffic           | If MySQL on same server |
| `configure_redis()`         | Optimizes Redis for caching                | If Redis on same server |
| `create_app_user()`         | Creates www-data user                      | If user doesn't exist   |

---

## 🚀 Quick Start

```bash
# Run the script
sudo ./deploy.sh

# Answer 'y' to all prompts for full setup
# Or answer selectively based on your needs
```

The script is **safe to run multiple times** - it checks if configurations already exist before modifying them.
