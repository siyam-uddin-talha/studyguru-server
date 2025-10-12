import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from google.auth.transport import requests
from google.oauth2 import id_token
import secrets

from app.core.config import settings
from app.models.user import User

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        return None

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )

async def verify_google_token(token: str) -> Optional[dict]:
    try:
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), settings.GOOGLE_CLIENT_ID
        )
        return idinfo
    except ValueError:
        return None

def generate_pin() -> int:
    return secrets.randbelow(900000) + 100000

async def get_current_user(request: Request, db: AsyncSession) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    authorization = request.headers.get("Authorization")
    if not authorization:
        raise credentials_exception

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise credentials_exception
    except ValueError:
        raise credentials_exception

    payload = verify_token(token)
    if payload is None:
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user

async def get_current_user_optional(
    request: Request, db: AsyncSession
) -> Optional[User]:
    try:
        return await get_current_user(request, db)
    except HTTPException:
        return None
