#!/bin/bash
# StudyGuru Pro - Production Deployment Script
# This script helps set up the server for production deployment

set -e  # Exit on error

echo "🚀 StudyGuru Pro - Production Deployment Setup"
echo "=============================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root or with sudo"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "❌ Cannot detect OS"
    exit 1
fi

echo "📦 Detected OS: $OS"
echo ""

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing system dependencies..."
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        apt-get update
        apt-get install -y \
            python3.11 \
            python3.11-venv \
            python3-pip \
            mysql-server \
            redis-server \
            nginx \
            supervisor \
            build-essential \
            libssl-dev \
            libffi-dev \
            python3-dev
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
        yum update -y
        yum install -y \
            python3.11 \
            python3-pip \
            mariadb-server \
            redis \
            nginx \
            supervisor \
            gcc \
            openssl-devel \
            libffi-devel \
            python3-devel
    else
        echo "❌ Unsupported OS: $OS"
        exit 1
    fi
    
    echo "✅ Dependencies installed"
}

# Function to configure system limits
configure_system_limits() {
    echo "⚙️  Configuring system limits..."
    
    # File descriptor limits
    if ! grep -q "studyguru" /etc/security/limits.conf; then
        cat >> /etc/security/limits.conf << EOF

# StudyGuru Pro limits
* soft nofile 65535
* hard nofile 65535
www-data soft nofile 65535
www-data hard nofile 65535
EOF
    fi
    
    # Network optimizations
    if ! grep -q "studyguru" /etc/sysctl.conf; then
        cat >> /etc/sysctl.conf << EOF

# StudyGuru Pro network optimizations
net.core.somaxconn = 1024
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.ip_local_port_range = 10000 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30
EOF
    fi
    
    sysctl -p
    
    echo "✅ System limits configured"
}

# Function to configure MySQL
configure_mysql() {
    echo "🗄️  Configuring MySQL..."
    
    # Check if MySQL is installed
    if ! command -v mysql &> /dev/null; then
        echo "⚠️  MySQL not found. Please install MySQL first."
        return
    fi
    
    # Create MySQL config snippet
    MYSQL_CONFIG="/etc/mysql/conf.d/studyguru.cnf"
    
    if [ ! -f "$MYSQL_CONFIG" ]; then
        cat > "$MYSQL_CONFIG" << EOF
[mysqld]
# StudyGuru Pro MySQL Configuration
max_connections = 200
innodb_buffer_pool_size = 4G
innodb_log_file_size = 512M
query_cache_size = 256M
query_cache_type = 1
thread_cache_size = 50
table_open_cache = 2000
EOF
        echo "✅ MySQL configuration created at $MYSQL_CONFIG"
        echo "⚠️  Please restart MySQL: systemctl restart mysql"
    else
        echo "✅ MySQL configuration already exists"
    fi
}

# Function to configure Redis
configure_redis() {
    echo "🔴 Configuring Redis..."
    
    # Check if Redis is installed
    if ! command -v redis-server &> /dev/null; then
        echo "⚠️  Redis not found. Please install Redis first."
        return
    fi
    
    REDIS_CONFIG="/etc/redis/redis.conf"
    
    if [ -f "$REDIS_CONFIG" ]; then
        # Backup original config
        if [ ! -f "${REDIS_CONFIG}.backup" ]; then
            cp "$REDIS_CONFIG" "${REDIS_CONFIG}.backup"
        fi
        
        # Update Redis config
        sed -i 's/^# maxmemory .*/maxmemory 2gb/' "$REDIS_CONFIG"
        sed -i 's/^# maxmemory-policy .*/maxmemory-policy allkeys-lru/' "$REDIS_CONFIG"
        sed -i 's/^# io-threads .*/io-threads 4/' "$REDIS_CONFIG"
        sed -i 's/^# io-threads-do-reads .*/io-threads-do-reads yes/' "$REDIS_CONFIG"
        
        echo "✅ Redis configuration updated"
        echo "⚠️  Please restart Redis: systemctl restart redis"
    else
        echo "⚠️  Redis config file not found at $REDIS_CONFIG"
    fi
}

# Function to create application user
create_app_user() {
    echo "👤 Creating application user..."
    
    if ! id "www-data" &>/dev/null; then
        useradd -r -s /bin/false www-data
    fi
    
    echo "✅ Application user ready"
}

# Main execution
main() {
    echo ""
    read -p "Install system dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_dependencies
    fi
    
    echo ""
    read -p "Configure system limits? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        configure_system_limits
    fi
    
    echo ""
    read -p "Configure MySQL? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        configure_mysql
    fi
    
    echo ""
    read -p "Configure Redis? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        configure_redis
    fi
    
    echo ""
    read -p "Create application user? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_app_user
    fi
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Copy env.production.example to .env.production and configure"
    echo "2. Install Python dependencies: pip install -r requirements.txt"
    echo "3. Copy systemd service file: cp systemd/studyguru.service /etc/systemd/system/"
    echo "4. Enable and start service: systemctl enable studyguru && systemctl start studyguru"
    echo "5. Check status: systemctl status studyguru"
    echo ""
}

# Run main function
main

