"""
Model configuration for helper models (background tasks and embeddings)

This module provides mappings for selecting appropriate helper models
based on the primary model's group category.
"""

from app.models.subscription import ModelGroup


# Background task model mappings
# Used for tasks like title generation, summary, context extraction, etc.
BACKGROUND_TASK_MODELS = {
    ModelGroup.GEMINI: "gemini-2.5-flash",
    ModelGroup.GPT: "gpt-5-nano",
    ModelGroup.KIMI: "gpt-5-nano",
}

# Embedding model mappings
# Used for vector embeddings
EMBEDDING_MODELS = {
    ModelGroup.GEMINI: "gemini-embedding-001",
    ModelGroup.GPT: "text-embedding-3-small",
    ModelGroup.KIMI: "text-embedding-3-small",
}


def get_background_task_model(model_group: ModelGroup) -> str:
    """
    Get the background task model for a given model group.

    Args:
        model_group: The ModelGroup enum value

    Returns:
        The model name string for background tasks
    """
    return BACKGROUND_TASK_MODELS.get(model_group, "gpt-5-nano")


def get_embedding_model(model_group: ModelGroup) -> str:
    """
    Get the embedding model for a given model group.

    Args:
        model_group: The ModelGroup enum value

    Returns:
        The model name string for embeddings
    """
    return EMBEDDING_MODELS.get(model_group, "text-embedding-3-small")
