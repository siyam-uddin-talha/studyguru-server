#!/usr/bin/env python3
"""
Simple test script to check if the basic FastAPI setup works
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="StudyGuru Pro API",
    description="FastAPI GraphQL API for StudyGuru Pro",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "StudyGuru Pro API - Basic Test"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "test_basic:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
    )
