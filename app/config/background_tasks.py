"""
Configuration for background task processing
"""

from app.core.config import settings

# Background task configuration
MAX_CONCURRENT_BACKGROUND_TASKS = getattr(
    settings, "MAX_CONCURRENT_BACKGROUND_TASKS", 5
)
BACKGROUND_TASK_TIMEOUT = getattr(settings, "BACKGROUND_TASK_TIMEOUT", 300)  # 5 minutes
BACKGROUND_TASK_RETRY_DELAY = getattr(
    settings, "BACKGROUND_TASK_RETRY_DELAY", 30
)  # 30 seconds
BACKGROUND_TASK_CLEANUP_HOURS = getattr(
    settings, "BACKGROUND_TASK_CLEANUP_HOURS", 24
)  # 24 hours


# Task priority levels
class TaskPriority:
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


# Task types
class TaskType:
    DELETE_INTERACTION = "delete_interaction"
    DELETE_INTERACTIONS = "delete_interactions"
    DELETE_MEDIA_FILES = "delete_media_files"
    DELETE_VECTOR_EMBEDDINGS = "delete_vector_embeddings"
    CLEANUP_ORPHANED_DATA = "cleanup_orphaned_data"
