import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings


class MessageTaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageTaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BackgroundMessageTask:
    id: str
    user_id: str
    interaction_id: Optional[str]
    message: str
    media_files: List[Dict[str, Any]]
    max_tokens: int
    status: MessageTaskStatus = MessageTaskStatus.PENDING
    priority: MessageTaskPriority = MessageTaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None
    streaming_content: str = ""


class BackgroundMessageService:
    """Service for managing background message processing without blocking API requests"""

    def __init__(self):
        self.tasks: Dict[str, BackgroundMessageTask] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = getattr(
            settings, "MAX_CONCURRENT_MESSAGE_TASKS", 10
        )
        self.task_timeout = getattr(settings, "MESSAGE_TASK_TIMEOUT", 600)  # 10 minutes
        self.retry_delay = getattr(
            settings, "MESSAGE_TASK_RETRY_DELAY", 30
        )  # 30 seconds

    async def submit_message_task(
        self,
        user_id: str,
        interaction_id: Optional[str],
        message: str,
        media_files: List[Dict[str, Any]],
        max_tokens: int,
        priority: MessageTaskPriority = MessageTaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> str:
        """Submit a new background message task"""
        task_id = str(uuid.uuid4())

        task = BackgroundMessageTask(
            id=task_id,
            user_id=user_id,
            interaction_id=interaction_id,
            message=message,
            media_files=media_files,
            max_tokens=max_tokens,
            priority=priority,
            max_retries=max_retries,
        )

        self.tasks[task_id] = task
        self._add_to_queue(task_id)

        # Start processing if not already running
        if len(self.processing_tasks) < self.max_concurrent_tasks:
            asyncio.create_task(self._process_task_queue())

        print(f"📋 Submitted background message task: {task_id}")
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

            if task.status != MessageTaskStatus.PENDING:
                continue

            # Create processing task
            processing_task = asyncio.create_task(self._execute_message_task(task))
            self.processing_tasks[task_id] = processing_task

            # Add completion callback
            processing_task.add_done_callback(
                lambda t, tid=task_id: self._on_task_complete(tid, t)
            )

    async def _execute_message_task(self, task: BackgroundMessageTask) -> bool:
        """Execute a single background message task"""
        try:
            task.status = MessageTaskStatus.PROCESSING
            task.started_at = datetime.utcnow()

            print(f"🔄 Executing background message task: {task.id}")

            # Import here to avoid circular imports
            from app.core.database import AsyncSessionLocal
            from app.services.interaction import process_conversation_message
            from app.models.user import User
            from app.models.interaction import Interaction
            from app.models.subscription import PurchasedSubscription
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with AsyncSessionLocal() as db:
                # Get user with eager loading of purchased_subscription
                user_result = await db.execute(
                    select(User)
                    .options(
                        selectinload(User.purchased_subscription).selectinload(
                            PurchasedSubscription.subscription
                        )
                    )
                    .where(User.id == task.user_id)
                )
                user = user_result.scalar_one_or_none()

                if not user:
                    print(f"⚠️ User {task.user_id} not found")
                    return False

                # Get or create interaction
                interaction = None
                if task.interaction_id:
                    interaction_result = await db.execute(
                        select(Interaction).where(
                            Interaction.id == task.interaction_id,
                            Interaction.user_id == task.user_id,
                        )
                    )
                    interaction = interaction_result.scalar_one_or_none()

                # Process the conversation message
                task.status = MessageTaskStatus.STREAMING

                result = await process_conversation_message(
                    user=user,
                    interaction=interaction,
                    message=task.message,
                    media_files=task.media_files if task.media_files else None,
                    max_tokens=task.max_tokens,
                    db=db,
                )

                if result.get("success", False):
                    task.status = MessageTaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    task.result = result
                    print(f"✅ Completed background message task: {task.id}")
                else:
                    task.status = MessageTaskStatus.FAILED
                    task.error_message = result.get("message", "Unknown error")
                    task.completed_at = datetime.utcnow()
                    print(f"❌ Failed background message task: {task.id}")

                return result.get("success", False)

        except Exception as e:
            task.status = MessageTaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            print(f"❌ Error in background message task {task.id}: {e}")
            return False

    def _on_task_complete(self, task_id: str, task: asyncio.Task):
        """Handle task completion"""
        if task_id in self.processing_tasks:
            del self.processing_tasks[task_id]

        # Continue processing if there are more tasks
        if self.task_queue:
            asyncio.create_task(self._process_task_queue())

    def get_task_status(self, task_id: str) -> Optional[BackgroundMessageTask]:
        """Get task status by ID"""
        return self.tasks.get(task_id)

    def get_user_tasks(self, user_id: str) -> List[BackgroundMessageTask]:
        """Get all tasks for a specific user"""
        return [task for task in self.tasks.values() if task.user_id == user_id]

    def get_interaction_tasks(self, interaction_id: str) -> List[BackgroundMessageTask]:
        """Get all tasks for a specific interaction"""
        return [
            task
            for task in self.tasks.values()
            if task.interaction_id == interaction_id
        ]

    def get_active_tasks(self) -> List[BackgroundMessageTask]:
        """Get all active tasks"""
        return [
            task
            for task in self.tasks.values()
            if task.status
            in [
                MessageTaskStatus.PENDING,
                MessageTaskStatus.PROCESSING,
                MessageTaskStatus.STREAMING,
            ]
        ]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.processing_tasks:
            self.processing_tasks[task_id].cancel()
            del self.processing_tasks[task_id]

        if task_id in self.tasks:
            self.tasks[task_id].status = MessageTaskStatus.CANCELLED
            self.tasks[task_id].completed_at = datetime.utcnow()

            # Remove from queue if present
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)

            return True

        return False

    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        task = self.tasks.get(task_id)
        if not task or task.status != MessageTaskStatus.FAILED:
            return False

        if task.retry_count >= task.max_retries:
            print(f"❌ Task {task_id} has exceeded max retries")
            return False

        task.retry_count += 1
        task.status = MessageTaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error_message = None
        task.result = None

        # Add back to queue
        self._add_to_queue(task_id)

        # Start processing if not already running
        if len(self.processing_tasks) < self.max_concurrent_tasks:
            asyncio.create_task(self._process_task_queue())

        print(
            f"🔄 Retrying background message task: {task_id} (attempt {task.retry_count})"
        )
        return True

    async def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """Clean up completed tasks older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (
                task.status
                in [
                    MessageTaskStatus.COMPLETED,
                    MessageTaskStatus.FAILED,
                    MessageTaskStatus.CANCELLED,
                ]
                and task.completed_at
                and task.completed_at < cutoff_time
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.tasks[task_id]

        print(f"🧹 Cleaned up {len(tasks_to_remove)} completed message tasks")

    def get_task_statistics(self) -> Dict[str, int]:
        """Get task statistics"""
        tasks = list(self.tasks.values())

        return {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == MessageTaskStatus.PENDING]),
            "processing": len(
                [t for t in tasks if t.status == MessageTaskStatus.PROCESSING]
            ),
            "streaming": len(
                [t for t in tasks if t.status == MessageTaskStatus.STREAMING]
            ),
            "completed": len(
                [t for t in tasks if t.status == MessageTaskStatus.COMPLETED]
            ),
            "failed": len([t for t in tasks if t.status == MessageTaskStatus.FAILED]),
            "cancelled": len(
                [t for t in tasks if t.status == MessageTaskStatus.CANCELLED]
            ),
        }


# Global instance
background_message_service = BackgroundMessageService()
