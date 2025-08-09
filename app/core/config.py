from pydantic_settings import BaseSettings
from typing import Optional
import os
from os import getenv


class Settings(BaseSettings):
    # Server settings
    PORT: int = int(getenv("PORT", 8000))
    ENVIRONMENT: str = getenv("ENVIRONMENT", "development")

    # Database
    DATABASE_URL: str = getenv(
        "DATABASE_URL",
        "mysql+asyncmy://root:qSZyxkNekYRimpvKnVoD@localhost:3306/study_guru_pro",
    )

    # JWT
    JWT_SECRET_KEY: str = getenv(
        "JWT_SECRET_KEY", "dev-secret-key-change-in-production"
    )

    # Client
    CLIENT_ORIGIN: str = getenv("CLIENT_ORIGIN", "http://localhost:3000")
    SERVER_URL: str = getenv("SERVER_URL", "http://localhost:5000")

    # Email SMTP
    SMTP_EMAIL_HOST: str = getenv("SMTP_EMAIL_HOST", "smtp.gmail.com")
    SMTP_EMAIL_PORT: int = int(getenv("SMTP_EMAIL_PORT", 587))
    SMTP_EMAIL_ADDRESS: str = getenv("SMTP_EMAIL_ADDRESS", "test@example.com")
    SMTP_EMAIL_PASSWORD: str = getenv("SMTP_EMAIL_PASSWORD", "password")

    # AWS S3
    AWS_ACCESS_KEY: str = getenv("AWS_ACCESS_KEY", "test-key")
    SECRET_ACCESS_KEY: str = getenv("SECRET_ACCESS_KEY", "test-secret")
    AWS_S3_BUCKET: str = getenv("AWS_S3_BUCKET", "studyguru-pro")

    # Paddle Configuration
    PADDLE_API_KEY: str = getenv("PADDLE_API_KEY", "test-key")
    PADDLE_WEBHOOK_SECRET: str = getenv("PADDLE_WEBHOOK_SECRET", "test-secret")
    PADDLE_ENVIRONMENT: str = getenv("PADDLE_ENVIRONMENT", "sandbox")  # or "production"

    # OpenAI
    OPENAI_API_KEY: str = getenv("OPENAI_API_KEY", "test-key")

    GOOGLE_CLIENT_ID: str = getenv("GOOGLE_CLIENT_ID", "")

    class Config:
        env_file = ".env"


settings = Settings()
