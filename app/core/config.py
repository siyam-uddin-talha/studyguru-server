from pydantic_settings import BaseSettings
from typing import Optional
import os
from os import getenv
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Server settings
    APP_NAME: str = "Study Guru - Pro"
    APP_LINK: str = getenv(
        "APP_LINK",
        "studygurupro://open/",
    )
    PORT: int = int(getenv("PORT", 8000))
    ENVIRONMENT: str = getenv("ENVIRONMENT", "development")
    DISABLE_LOGS: bool = getenv("DISABLE_LOGS", "true").lower() == "false"

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
    AWS_ORIGIN: str = getenv("AWS_ORIGIN", "ap-southeast-1")

    # Paddle Configuration
    PADDLE_API_KEY: str = getenv("PADDLE_API_KEY", "test-key")
    PADDLE_WEBHOOK_SECRET: str = getenv("PADDLE_WEBHOOK_SECRET", "test-secret")
    PADDLE_ENVIRONMENT: str = getenv("PADDLE_ENVIRONMENT", "sandbox")  # or "production"

    # Guardrail Configuration
    DISABLE_GUARDRAIL: bool = getenv("DISABLE_GUARDRAIL", "true").lower() == "true"

    # Logging Configuration
    ENABLE_LOGS: bool = getenv("ENABLE_LOGS", "false").lower() == "true"

    # OpenAI
    OPENAI_API_KEY: str = getenv("OPENAI_API_KEY", "")

    GOOGLE_CLIENT_ID: str = getenv("GOOGLE_CLIENT_ID", "google-id")

    # Zilliz / Milvus Vector DB
    ZILLIZ_URI: str = getenv("ZILLIZ_URI", "")
    ZILLIZ_TOKEN: str = getenv("ZILLIZ_TOKEN", "")
    ZILLIZ_DB_NAME: str = getenv("ZILLIZ_DB_NAME", "studyguru")
    ZILLIZ_COLLECTION: str = getenv("ZILLIZ_COLLECTION", "document_embeddings")
    ZILLIZ_DIMENSION: int = int(getenv("ZILLIZ_DIMENSION", 1536))
    ZILLIZ_INDEX_METRIC: str = getenv("ZILLIZ_INDEX_METRIC", "IP")  # or "L2", "COSINE"
    ZILLIZ_CONSISTENCY_LEVEL: str = getenv("ZILLIZ_CONSISTENCY_LEVEL", "Bounded")
    ADD_UNIT_ID_1: str = getenv(
        "ADD_UNIT_ID_1", "ca-app-pub-2962676217775659/5723259375"
    )

    class Config:
        env_file = ".env"


settings = Settings()
