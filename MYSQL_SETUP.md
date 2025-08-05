# MySQL Setup for StudyGuru Pro

This guide will help you set up MySQL for the StudyGuru Pro backend.

## Prerequisites

1. MySQL Server (8.0 or higher recommended)
2. Python 3.11 or higher
3. pip or uv package manager

## Installation

### 1. Install MySQL Dependencies

The project now uses `asyncmy` for MySQL async support. Install the dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Create MySQL Database

```sql
CREATE DATABASE studyguru_pro CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'studyguru_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON studyguru_pro.* TO 'studyguru_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Environment Configuration

Create a `.env` file in the server directory with the following variables:

```env
# Server settings
PORT=5000
ENVIRONMENT=development

# Database - MySQL
DATABASE_URL=mysql+asyncmy://studyguru_user:your_password@localhost/studyguru_pro

# JWT
JWT_SECRET_KEY=your-secret-key-here

# Client
CLIENT_ORIGIN=http://localhost:3000
SERVER_URL=http://localhost:5000

# Email SMTP
SMTP_EMAIL_HOST=smtp.gmail.com
SMTP_EMAIL_PORT=587
SMTP_EMAIL_ADDRESS=your-email@gmail.com
SMTP_EMAIL_PASSWORD=your-app-password

# AWS S3
AWS_ACCESS_KEY=your-aws-access-key
SECRET_ACCESS_KEY=your-aws-secret-key
AWS_S3_BUCKET=studyguru-pro

# Paddle Configuration
PADDLE_API_KEY=your-paddle-api-key
PADDLE_WEBHOOK_SECRET=your-paddle-webhook-secret
PADDLE_ENVIRONMENT=sandbox

# OpenAI
OPENAI_API_KEY=your-openai-api-key
```

### 4. Run Database Migrations

```bash
# Navigate to server directory
cd server

# Run migrations
alembic upgrade head
```

### 5. Start the Server

```bash
# Using uv
uv run python -m app.main

# Or using python directly
python -m app.main
```

## Key Changes from PostgreSQL

1. **Database Driver**: Changed from `asyncpg` to `asyncmy`
2. **Connection String**: Uses `mysql+asyncmy://` prefix
3. **Default Values**: Boolean defaults use `0`/`1` instead of `true`/`false`
4. **Timestamps**: Uses `CURRENT_TIMESTAMP` and `ON UPDATE CURRENT_TIMESTAMP`
5. **String Length**: Limited to 191 characters for indexes (MySQL limitation)

## Troubleshooting

### Common Issues

1. **Connection Error**: Make sure MySQL server is running and accessible
2. **Authentication Error**: Verify username, password, and database name
3. **Character Set**: Ensure database uses `utf8mb4` character set
4. **Port Issues**: Default MySQL port is 3306, make sure it's not blocked

### Useful MySQL Commands

```sql
-- Check database status
SHOW DATABASES;
USE studyguru_pro;
SHOW TABLES;

-- Check table structure
DESCRIBE user;
DESCRIBE subscriptions;

-- Check migration status
SELECT * FROM alembic_version;
```

## Production Considerations

1. **Security**: Use strong passwords and limit database access
2. **Backup**: Set up regular database backups
3. **Performance**: Configure MySQL for production workloads
4. **SSL**: Enable SSL connections for production environments
