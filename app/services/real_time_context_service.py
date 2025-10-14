"""
Real-time context update and consistency service
Ensures context updates happen reliably and consistently across the system
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.interaction import Interaction, Conversation
from app.models.context import (
    ConversationContext,
    UserLearningProfile,
    DocumentContext,
    ContextUsageLog,
)
from app.services.langchain_service import langchain_service
from app.services.semantic_summary_service import semantic_summary_service
from app.services.vector_optimization_service import vector_optimization_service


class UpdateStatus(Enum):
    """Status of context updates"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ContextUpdateTask:
    """Represents a context update task"""

    task_id: str
    user_id: str
    interaction_id: str
    conversation_id: Optional[str]
    update_type: str  # "semantic_summary", "embedding", "document_context", etc.
    payload: Dict[str, Any]
    status: UpdateStatus = UpdateStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    updated_at: datetime = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class RealTimeContextService:
    """Service for managing real-time context updates and consistency"""

    def __init__(self):
        self.update_queue: List[ContextUpdateTask] = []
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_updates = 5
        self.update_timeout = 30  # seconds
        self.retry_delay = 2  # seconds

    async def queue_context_update(
        self,
        user_id: str,
        interaction_id: str,
        conversation_id: Optional[str],
        update_type: str,
        payload: Dict[str, Any],
        priority: int = 1,
    ) -> str:
        """Queue a context update for processing"""
        task_id = f"{update_type}_{interaction_id}_{int(time.time())}"

        task = ContextUpdateTask(
            task_id=task_id,
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=conversation_id,
            update_type=update_type,
            payload=payload,
        )

        # Insert based on priority (lower number = higher priority)
        insert_index = 0
        for i, existing_task in enumerate(self.update_queue):
            if existing_task.update_type == "semantic_summary":
                insert_index = i + 1
            else:
                break

        self.update_queue.insert(insert_index, task)

        # Start processing if not already running
        if len(self.processing_tasks) < self.max_concurrent_updates:
            asyncio.create_task(self._process_update_queue())

        return task_id

    async def _process_update_queue(self):
        """Process the update queue"""
        while (
            self.update_queue
            and len(self.processing_tasks) < self.max_concurrent_updates
        ):
            task = self.update_queue.pop(0)

            # Create processing task
            processing_task = asyncio.create_task(self._execute_update_task(task))
            self.processing_tasks[task.task_id] = processing_task

            # Add completion callback
            processing_task.add_done_callback(
                lambda t, task_id=task.task_id: self._on_task_complete(task_id, t)
            )

    async def _execute_update_task(self, task: ContextUpdateTask) -> bool:
        """Execute a single update task"""
        try:
            task.status = UpdateStatus.IN_PROGRESS
            task.updated_at = datetime.now()

            print(
                f"🔄 Executing context update: {task.update_type} for interaction {task.interaction_id}"
            )

            # Route to appropriate handler
            if task.update_type == "semantic_summary":
                success = await self._update_semantic_summary(task)
            elif task.update_type == "embedding":
                success = await self._update_embedding(task)
            elif task.update_type == "document_context":
                success = await self._update_document_context(task)
            elif task.update_type == "user_learning_profile":
                success = await self._update_user_learning_profile(task)
            elif task.update_type == "conversation_context":
                success = await self._update_conversation_context(task)
            else:
                print(f"⚠️ Unknown update type: {task.update_type}")
                return False

            if success:
                task.status = UpdateStatus.COMPLETED
                task.updated_at = datetime.now()
                print(f"✅ Context update completed: {task.update_type}")
                return True
            else:
                raise Exception("Update handler returned False")

        except Exception as e:
            task.status = UpdateStatus.FAILED
            task.error_message = str(e)
            task.updated_at = datetime.now()
            print(f"❌ Context update failed: {task.update_type} - {e}")

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = UpdateStatus.RETRYING
                print(f"🔄 Retrying update {task.retry_count}/{task.max_retries}")

                # Add back to queue with delay
                await asyncio.sleep(self.retry_delay * task.retry_count)
                self.update_queue.insert(0, task)  # High priority for retries

            return False

    async def _update_semantic_summary(self, task: ContextUpdateTask) -> bool:
        """Update semantic summary with consistency checks"""
        try:
            payload = task.payload
            user_message = payload.get("user_message", "")
            ai_response = payload.get("ai_response", "")

            async with AsyncSessionLocal() as db:
                # Get current interaction
                result = await db.execute(
                    select(Interaction).where(Interaction.id == task.interaction_id)
                )
                interaction = result.scalar_one_or_none()

                if not interaction:
                    print(f"⚠️ Interaction not found: {task.interaction_id}")
                    return False

                # Get current summary
                current_summary = interaction.semantic_summary

                # Update using semantic summary service
                updated_summary = (
                    await semantic_summary_service.update_interaction_summary(
                        current_summary=current_summary,
                        new_user_message=user_message,
                        new_ai_response=ai_response,
                    )
                )

                # Validate summary before saving
                if not self._validate_semantic_summary(updated_summary):
                    print(f"⚠️ Invalid semantic summary generated")
                    return False

                # Save with transaction
                interaction.semantic_summary = updated_summary
                interaction.updated_at = datetime.now()

                await db.commit()

                # Verify the update was saved correctly
                await db.refresh(interaction)
                if interaction.semantic_summary != updated_summary:
                    print(f"⚠️ Semantic summary not saved correctly")
                    return False

                print(f"✅ Semantic summary updated successfully")
                return True

        except Exception as e:
            print(f"❌ Semantic summary update failed: {e}")
            return False

    async def _update_embedding(self, task: ContextUpdateTask) -> bool:
        """Update embeddings with enhanced metadata"""
        try:
            payload = task.payload
            conv_id = payload.get("conversation_id")
            text = payload.get("text", "")
            title = payload.get("title", "")
            metadata = payload.get("metadata", {})

            # Enhanced metadata with consistency information
            enhanced_metadata = {
                **metadata,
                "update_timestamp": datetime.now().isoformat(),
                "task_id": task.task_id,
                "consistency_version": "1.0",
            }

            # Create embedding using LangChain service
            result = await langchain_service.upsert_embedding(
                conv_id=conv_id,
                user_id=task.user_id,
                text=text,
                title=title,
                metadata=enhanced_metadata,
            )

            if result:
                print(f"✅ Embedding created successfully")
                return True
            else:
                print(f"⚠️ Embedding creation returned False")
                return False

        except Exception as e:
            print(f"❌ Embedding update failed: {e}")
            return False

    async def _update_document_context(self, task: ContextUpdateTask) -> bool:
        """Update document context with consistency checks"""
        try:
            payload = task.payload
            media_id = payload.get("media_id")
            document_analysis = payload.get("document_analysis")

            if not media_id or not document_analysis:
                print(f"⚠️ Missing required fields for document context update")
                return False

            async with AsyncSessionLocal() as db:
                # Check if document context exists
                result = await db.execute(
                    select(DocumentContext).where(
                        and_(
                            DocumentContext.media_id == media_id,
                            DocumentContext.interaction_id == task.interaction_id,
                        )
                    )
                )
                existing_doc = result.scalar_one_or_none()

                if existing_doc:
                    # Update existing
                    existing_doc.document_type = document_analysis.get("document_type")
                    existing_doc.total_questions = document_analysis.get(
                        "total_questions"
                    )
                    existing_doc.main_topics = document_analysis.get("main_topics")
                    existing_doc.difficulty_level = document_analysis.get(
                        "difficulty_level"
                    )
                    existing_doc.subject_area = document_analysis.get("subject_area")
                    existing_doc.updated_at = datetime.now()
                else:
                    # Create new
                    doc_context = DocumentContext(
                        media_id=media_id,
                        interaction_id=task.interaction_id,
                        user_id=task.user_id,
                        document_type=document_analysis.get("document_type"),
                        total_questions=document_analysis.get("total_questions"),
                        main_topics=document_analysis.get("main_topics"),
                        difficulty_level=document_analysis.get("difficulty_level"),
                        subject_area=document_analysis.get("subject_area"),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                    db.add(doc_context)

                await db.commit()
                print(f"✅ Document context updated successfully")
                return True

        except Exception as e:
            print(f"❌ Document context update failed: {e}")
            return False

    async def _update_user_learning_profile(self, task: ContextUpdateTask) -> bool:
        """Update user learning profile with consistency checks"""
        try:
            payload = task.payload
            learning_data = payload.get("learning_data", {})

            async with AsyncSessionLocal() as db:
                # Get or create learning profile
                result = await db.execute(
                    select(UserLearningProfile).where(
                        UserLearningProfile.user_id == task.user_id
                    )
                )
                profile = result.scalar_one_or_none()

                if not profile:
                    profile = UserLearningProfile(
                        user_id=task.user_id,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                    db.add(profile)

                # Update profile data
                if "preferred_topics" in learning_data:
                    profile.preferred_topics = learning_data["preferred_topics"]
                if "learning_style" in learning_data:
                    profile.learning_style = learning_data["learning_style"]
                if "mastered_concepts" in learning_data:
                    profile.mastered_concepts = learning_data["mastered_concepts"]
                if "struggling_areas" in learning_data:
                    profile.struggling_areas = learning_data["struggling_areas"]

                profile.updated_at = datetime.now()
                profile.last_activity = datetime.now()

                await db.commit()
                print(f"✅ User learning profile updated successfully")
                return True

        except Exception as e:
            print(f"❌ User learning profile update failed: {e}")
            return False

    async def _update_conversation_context(self, task: ContextUpdateTask) -> bool:
        """Update conversation context cache"""
        try:
            payload = task.payload
            context_data = payload.get("context_data", {})

            async with AsyncSessionLocal() as db:
                # Create or update conversation context
                context = ConversationContext(
                    interaction_id=task.interaction_id,
                    user_id=task.user_id,
                    context_type="conversation_cache",
                    context_data=context_data,
                    context_hash=self._calculate_context_hash(context_data),
                    relevance_score=context_data.get("relevance_score", 0.8),
                    recency_score=1.0,  # Fresh context
                    importance_score=context_data.get("importance_score", 0.7),
                    content_length=len(str(context_data)),
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=24),
                )

                db.add(context)
                await db.commit()

                print(f"✅ Conversation context updated successfully")
                return True

        except Exception as e:
            print(f"❌ Conversation context update failed: {e}")
            return False

    def _validate_semantic_summary(self, summary: Dict[str, Any]) -> bool:
        """Validate semantic summary structure and content"""
        try:
            required_fields = ["updated_summary", "key_topics", "version"]

            for field in required_fields:
                if field not in summary:
                    print(f"⚠️ Missing required field in semantic summary: {field}")
                    return False

            # Check summary content
            if (
                not summary.get("updated_summary")
                or len(summary["updated_summary"]) < 10
            ):
                print(f"⚠️ Semantic summary too short or empty")
                return False

            # Check version
            if not isinstance(summary.get("version"), int) or summary["version"] < 1:
                print(f"⚠️ Invalid version in semantic summary")
                return False

            return True

        except Exception as e:
            print(f"⚠️ Error validating semantic summary: {e}")
            return False

    def _calculate_context_hash(self, context_data: Dict[str, Any]) -> str:
        """Calculate hash for context data"""
        import hashlib

        data_string = json.dumps(context_data, sort_keys=True)
        return hashlib.md5(data_string.encode()).hexdigest()

    def _on_task_complete(self, task_id: str, task: asyncio.Task):
        """Handle task completion"""
        if task_id in self.processing_tasks:
            del self.processing_tasks[task_id]

        # Continue processing queue
        if self.update_queue:
            asyncio.create_task(self._process_update_queue())

    async def get_update_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific update task"""
        # Check processing tasks
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "in_progress",
                "created_at": None,
                "updated_at": None,
            }

        # Check queue
        for queued_task in self.update_queue:
            if queued_task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": queued_task.status.value,
                    "retry_count": queued_task.retry_count,
                    "created_at": queued_task.created_at.isoformat(),
                    "updated_at": queued_task.updated_at.isoformat(),
                    "error_message": queued_task.error_message,
                }

        return None

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        return {
            "queue_length": len(self.update_queue),
            "processing_tasks": len(self.processing_tasks),
            "max_concurrent": self.max_concurrent_updates,
            "queue_items": [
                {
                    "task_id": task.task_id,
                    "update_type": task.update_type,
                    "status": task.status.value,
                    "retry_count": task.retry_count,
                    "created_at": task.created_at.isoformat(),
                }
                for task in self.update_queue[:10]  # Show first 10 items
            ],
        }

    async def ensure_consistency(self, user_id: str, interaction_id: str) -> bool:
        """Ensure all context is consistent for a user/interaction"""
        try:
            print(
                f"🔍 Ensuring consistency for user {user_id}, interaction {interaction_id}"
            )

            async with AsyncSessionLocal() as db:
                # Check interaction exists
                result = await db.execute(
                    select(Interaction).where(Interaction.id == interaction_id)
                )
                interaction = result.scalar_one_or_none()

                if not interaction:
                    print(f"⚠️ Interaction not found: {interaction_id}")
                    return False

                # Check semantic summary consistency
                if not interaction.semantic_summary:
                    print(
                        f"⚠️ Missing semantic summary for interaction {interaction_id}"
                    )
                    return False

                # Check document context consistency
                doc_result = await db.execute(
                    select(DocumentContext).where(
                        and_(
                            DocumentContext.user_id == user_id,
                            DocumentContext.interaction_id == interaction_id,
                        )
                    )
                )
                doc_contexts = doc_result.scalars().all()

                # Check conversation context consistency
                conv_result = await db.execute(
                    select(ConversationContext).where(
                        and_(
                            ConversationContext.user_id == user_id,
                            ConversationContext.interaction_id == interaction_id,
                        )
                    )
                )
                conv_contexts = conv_result.scalars().all()

                print(f"✅ Consistency check completed:")
                print(
                    f"   Semantic summary: {'✓' if interaction.semantic_summary else '✗'}"
                )
                print(f"   Document contexts: {len(doc_contexts)}")
                print(f"   Conversation contexts: {len(conv_contexts)}")

                return True

        except Exception as e:
            print(f"❌ Consistency check failed: {e}")
            return False

    async def cleanup_expired_context(self, max_age_hours: int = 24) -> int:
        """Clean up expired context data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            async with AsyncSessionLocal() as db:
                # Clean up expired conversation contexts
                result = await db.execute(
                    select(ConversationContext).where(
                        ConversationContext.expires_at < datetime.now()
                    )
                )
                expired_contexts = result.scalars().all()

                for context in expired_contexts:
                    await db.delete(context)

                await db.commit()

                print(f"✅ Cleaned up {len(expired_contexts)} expired context entries")
                return len(expired_contexts)

        except Exception as e:
            print(f"❌ Context cleanup failed: {e}")
            return 0


# Global instance
real_time_context_service = RealTimeContextService()
