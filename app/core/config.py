from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Server settings
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = (
        "mysql+asyncmy://root:qSZyxkNekYRimpvKnVoD@localhost:3306/study_guru_pro"
    )

    # JWT
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"

    # Client
    CLIENT_ORIGIN: str = "http://localhost:3000"
    SERVER_URL: str = "http://localhost:5000"

    # Email SMTP
    SMTP_EMAIL_HOST: str = "smtp.gmail.com"
    SMTP_EMAIL_PORT: int = 587
    SMTP_EMAIL_ADDRESS: str = "test@example.com"
    SMTP_EMAIL_PASSWORD: str = "password"

    # AWS S3
    AWS_ACCESS_KEY: str = "test-key"
    SECRET_ACCESS_KEY: str = "test-secret"
    AWS_S3_BUCKET: str = "studyguru-pro"

    # Paddle Configuration
    PADDLE_API_KEY: str = "test-key"
    PADDLE_WEBHOOK_SECRET: str = "test-secret"
    PADDLE_ENVIRONMENT: str = "sandbox"  # or "production"

    # OpenAI
    OPENAI_API_KEY: str = "test-key"

    class Config:
        env_file = ".env"


settings = Settings()
