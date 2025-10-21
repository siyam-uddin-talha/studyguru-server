import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BackgroundTask:
    id: str
    task_type: str
    user_id: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None


class BackgroundTaskService:
    """Service for managing background tasks without blocking API requests"""

    def __init__(self):
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = getattr(
            settings, "MAX_CONCURRENT_BACKGROUND_TASKS", 5
        )
        self.task_timeout = getattr(
            settings, "BACKGROUND_TASK_TIMEOUT", 300
        )  # 5 minutes
        self.retry_delay = getattr(
            settings, "BACKGROUND_TASK_RETRY_DELAY", 30
        )  # 30 seconds

    async def submit_task(
        self,
        task_type: str,
        user_id: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> str:
        """Submit a new background task"""
        task_id = str(uuid.uuid4())

        task = BackgroundTask(
            id=task_id,
            task_type=task_type,
            user_id=user_id,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
        )

        self.tasks[task_id] = task
        self._add_to_queue(task_id)

        # Start processing if not already running
        if len(self.processing_tasks) < self.max_concurrent_tasks:
            asyncio.create_task(self._process_task_queue())

        print(f"📋 Submitted background task: {task_type} (ID: {task_id})")
        return task_id

    def _add_to_queue(self, task_id: str):
        """Add task to queue based on priority"""
        task = self.tasks[task_id]

        # Insert based on priority (higher priority = lower number)
        insert_index = 0
        for i, queued_task_id in enumerate(self.task_queue):
            queued_task = self.tasks[queued_task_id]
            if queued_task.priority.value < task.priority.value:
                insert_index = i + 1
            else:
                break

        self.task_queue.insert(insert_index, task_id)

    async def _process_task_queue(self):
        """Process tasks from the queue"""
        while (
            self.task_queue and len(self.processing_tasks) < self.max_concurrent_tasks
        ):
            task_id = self.task_queue.pop(0)
            task = self.tasks[task_id]

            if task.status != TaskStatus.PENDING:
                continue

            # Create processing task
            processing_task = asyncio.create_task(self._execute_task(task))
            self.processing_tasks[task_id] = processing_task

            # Add completion callback
            processing_task.add_done_callback(
                lambda t, tid=task_id: self._on_task_complete(tid, t)
            )

    async def _execute_task(self, task: BackgroundTask) -> bool:
        """Execute a single background task"""
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()

            print(f"🔄 Executing background task: {task.task_type} (ID: {task.id})")

            # Route to appropriate handler
            if task.task_type == "delete_interaction":
                success = await self._handle_delete_interaction(task)
            elif task.task_type == "delete_interactions":
                success = await self._handle_delete_interactions(task)
            elif task.task_type == "delete_media_files":
                success = await self._handle_delete_media_files(task)
            elif task.task_type == "delete_vector_embeddings":
                success = await self._handle_delete_vector_embeddings(task)
            else:
                print(f"⚠️ Unknown task type: {task.task_type}")
                return False

            if success:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                print(f"✅ Completed background task: {task.task_type} (ID: {task.id})")
            else:
                task.status = TaskStatus.FAILED
                task.error_message = "Task execution failed"
                print(f"❌ Failed background task: {task.task_type} (ID: {task.id})")

            return success

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            print(f"❌ Error in background task {task.task_type} (ID: {task.id}): {e}")
            return False

    async def _handle_delete_interaction(self, task: BackgroundTask) -> bool:
        """Handle single interaction deletion"""
        try:
            interaction_id = task.payload.get("interaction_id")
            user_id = task.user_id

            if not interaction_id:
                return False

            # Import here to avoid circular imports
            from app.core.database import AsyncSessionLocal
            from app.models.interaction import Interaction, Conversation
            from app.models.context import (
                ContextUsageLog,
                ConversationContext,
                DocumentContext,
            )
            from app.models.interaction import InteractionShare, InteractionShareVisitor
            from app.services.file_service import FileService
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with AsyncSessionLocal() as db:
                # Get the interaction with all related data
                result = await db.execute(
                    select(Interaction)
                    .options(
                        selectinload(Interaction.conversations).selectinload(
                            Conversation.files
                        )
                    )
                    .where(
                        Interaction.id == interaction_id,
                        Interaction.user_id == user_id,
                    )
                )
                interaction = result.scalar_one_or_none()

                if not interaction:
                    print(
                        f"⚠️ Interaction {interaction_id} not found for user {user_id}"
                    )
                    return False

                # Collect all media files to delete from S3
                media_files_to_delete = set()
                for conversation in interaction.conversations:
                    if conversation.files:
                        for media_file in conversation.files:
                            media_files_to_delete.add(media_file.s3_key)

                # Delete media files from S3 (non-blocking)
                for s3_key in media_files_to_delete:
                    try:
                        await FileService.delete_file_from_s3(s3_key)
                    except Exception as e:
                        print(f"⚠️ Failed to delete S3 file {s3_key}: {e}")
                        # Continue even if S3 deletion fails

                # Delete from vector database if configured
                try:
                    from app.services.langchain_service import langchain_service

                    if langchain_service.vector_store:
                        await langchain_service.delete_embeddings_by_interaction(
                            interaction_id
                        )
                except Exception as e:
                    print(f"⚠️ Failed to delete vector embeddings: {e}")

                # Delete context-related records
                await self._delete_context_records(db, interaction_id)

                # Delete conversations
                for conversation in interaction.conversations:
                    await db.delete(conversation)

                # Delete the interaction
                await db.delete(interaction)
                await db.commit()

                print(f"✅ Successfully deleted interaction {interaction_id}")
                return True

        except Exception as e:
            print(f"❌ Error deleting interaction {interaction_id}: {e}")
            return False

    async def _handle_delete_interactions(self, task: BackgroundTask) -> bool:
        """Handle multiple interactions deletion"""
        try:
            interaction_ids = task.payload.get("interaction_ids", [])
            user_id = task.user_id

            if not interaction_ids:
                return False

            from app.core.database import AsyncSessionLocal
            from app.models.interaction import Interaction, Conversation
            from app.models.context import (
                ContextUsageLog,
                ConversationContext,
                DocumentContext,
            )
            from app.models.interaction import InteractionShare, InteractionShareVisitor
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with AsyncSessionLocal() as db:
                deleted_count = 0

                for interaction_id in interaction_ids:
                    try:
                        # Get the interaction
                        result = await db.execute(
                            select(Interaction)
                            .options(
                                selectinload(Interaction.conversations).selectinload(
                                    Conversation.files
                                )
                            )
                            .where(
                                Interaction.id == interaction_id,
                                Interaction.user_id == user_id,
                            )
                        )
                        interaction = result.scalar_one_or_none()

                        if not interaction:
                            continue

                        # Delete context records
                        await self._delete_context_records(db, interaction_id)

                        # Delete conversations
                        for conversation in interaction.conversations:
                            await db.delete(conversation)

                        # Delete the interaction
                        await db.delete(interaction)
                        deleted_count += 1

                    except Exception as e:
                        print(f"⚠️ Error deleting interaction {interaction_id}: {e}")
                        continue

                await db.commit()
                print(f"✅ Successfully deleted {deleted_count} interactions")
                return deleted_count > 0

        except Exception as e:
            print(f"❌ Error deleting interactions: {e}")
            return False

    async def _delete_context_records(self, db: AsyncSession, interaction_id: str):
        """Delete all context-related records for an interaction"""
        try:
            from app.models.context import (
                ContextUsageLog,
                ConversationContext,
                DocumentContext,
            )
            from app.models.interaction import InteractionShare, InteractionShareVisitor
            from sqlalchemy import select

            # Delete context usage logs
            context_logs = await db.execute(
                select(ContextUsageLog).where(
                    ContextUsageLog.interaction_id == interaction_id
                )
            )
            for log in context_logs.scalars():
                await db.delete(log)

            # Delete conversation contexts
            conversation_contexts = await db.execute(
                select(ConversationContext).where(
                    ConversationContext.interaction_id == interaction_id
                )
            )
            for context in conversation_contexts.scalars():
                await db.delete(context)

            # Delete document contexts
            document_contexts = await db.execute(
                select(DocumentContext).where(
                    DocumentContext.interaction_id == interaction_id
                )
            )
            for doc_context in document_contexts.scalars():
                await db.delete(doc_context)

            # Delete interaction share visitors and shares
            interaction_shares = await db.execute(
                select(InteractionShare).where(
                    InteractionShare.original_interaction_id == interaction_id
                )
            )
            for share in interaction_shares.scalars():
                # Delete visitors for this share
                visitors = await db.execute(
                    select(InteractionShareVisitor).where(
                        InteractionShareVisitor.interaction_share_id == share.id
                    )
                )
                for visitor in visitors.scalars():
                    await db.delete(visitor)
                # Delete the share itself
                await db.delete(share)

        except Exception as e:
            print(f"⚠️ Error deleting context records for {interaction_id}: {e}")

    async def _handle_delete_media_files(self, task: BackgroundTask) -> bool:
        """Handle media files deletion"""
        # Implementation for media file deletion
        return True

    async def _handle_delete_vector_embeddings(self, task: BackgroundTask) -> bool:
        """Handle vector embeddings deletion"""
        # Implementation for vector embeddings deletion
        return True

    def _on_task_complete(self, task_id: str, task: asyncio.Task):
        """Handle task completion"""
        if task_id in self.processing_tasks:
            del self.processing_tasks[task_id]

        # Continue processing if there are more tasks
        if self.task_queue:
            asyncio.create_task(self._process_task_queue())

    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task status by ID"""
        return self.tasks.get(task_id)

    def get_user_tasks(self, user_id: str) -> List[BackgroundTask]:
        """Get all tasks for a specific user"""
        return [task for task in self.tasks.values() if task.user_id == user_id]

    def get_active_tasks(self) -> List[BackgroundTask]:
        """Get all active tasks"""
        return [
            task
            for task in self.tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
        ]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.processing_tasks:
            self.processing_tasks[task_id].cancel()
            del self.processing_tasks[task_id]

        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.CANCELLED
            self.tasks[task_id].completed_at = datetime.utcnow()

            # Remove from queue if present
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)

            return True

        return False

    async def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """Clean up completed tasks older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (
                task.status
                in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.completed_at
                and task.completed_at < cutoff_time
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.tasks[task_id]

        print(f"🧹 Cleaned up {len(tasks_to_remove)} completed tasks")


# Global instance
background_task_service = BackgroundTaskService()
