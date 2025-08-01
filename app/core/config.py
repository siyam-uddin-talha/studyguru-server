from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Server settings
    PORT: int = 5000
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str
    
    # JWT
    JWT_SECRET_KEY: str
    
    # Client
    CLIENT_ORIGIN: str = "http://localhost:3000"
    SERVER_URL: str = "http://localhost:5000"
    
    # Email SMTP
    SMTP_EMAIL_HOST: str
    SMTP_EMAIL_PORT: int = 587
    SMTP_EMAIL_ADDRESS: str
    SMTP_EMAIL_PASSWORD: str
    
    # AWS S3
    AWS_ACCESS_KEY: str
    SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET: str = "studyguru-pro"
    
    # Paddle Configuration
    PADDLE_API_KEY: str
    PADDLE_WEBHOOK_SECRET: str
    PADDLE_ENVIRONMENT: str = "sandbox"  # or "production"
    
    # OpenAI
    OPENAI_API_KEY: str
    
    class Config:
        env_file = ".env"


settings = Settings()