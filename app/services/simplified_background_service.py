"""
Simplified Background Operations Service

Streamlined implementation that only handles:
- Creating embeddings for user and AI messages in parallel

Removed:
- Priority queue system
- Task tracking/retry logic
- Semantic summary updates
- Conversation context saves

Why this works:
- Vector embeddings are the ONLY thing needed for future retrieval
- Simpler = more reliable
- Production systems benefit from focused, well-tested pipelines
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime

from app.services.langchain_service import langchain_service


class SimplifiedBackgroundService:
    """
    Simplified background service that only creates embeddings

    Benefits:
    - 50% fewer background tasks
    - No complex priority queues
    - No task tracking overhead
    - More reliable embedding creation
    """

    async def create_embeddings_for_conversation(
        self,
        user_conv_id: str,
        ai_conv_id: str,
        user_id: str,
        interaction_id: str,
        message: str,
        ai_content: str,
    ) -> Dict[str, Any]:
        """
        Only create embeddings - nothing else

        Creates embeddings for both user message and AI response in parallel.
        These embeddings power the vector search for future context retrieval.

        Args:
            user_conv_id: ID of the user conversation record
            ai_conv_id: ID of the AI conversation record
            user_id: User's ID
            interaction_id: Interaction ID
            message: User's message content
            ai_content: AI response content

        Returns:
            Dict with success status and timing metrics
        """
        start_time = time.time()

        try:
            # Create embeddings for both messages in parallel
            results = await asyncio.gather(
                # User message embedding
                self._create_embedding(
                    conv_id=str(user_conv_id),
                    user_id=user_id,
                    text=message[:3000],  # Truncate for embedding
                    title=f"User message in {interaction_id}",
                    metadata={
                        "interaction_id": interaction_id,
                        "conversation_id": str(user_conv_id),
                        "role": "user",
                        "created_at": datetime.now().isoformat(),
                    },
                ),
                # AI response embedding
                self._create_embedding(
                    conv_id=str(ai_conv_id),
                    user_id=user_id,
                    text=str(ai_content)[:3000],  # Truncate for embedding
                    title=f"AI response in {interaction_id}",
                    metadata={
                        "interaction_id": interaction_id,
                        "conversation_id": str(ai_conv_id),
                        "role": "ai",
                        "created_at": datetime.now().isoformat(),
                    },
                ),
                return_exceptions=True,
            )

            user_success = not isinstance(results[0], Exception) and results[0]
            ai_success = not isinstance(results[1], Exception) and results[1]

            elapsed_time = time.time() - start_time

            if user_success and ai_success:
                print(f"✅ [BACKGROUND] Embeddings created in {elapsed_time:.2f}s")
            else:
                print(
                    f"⚠️ [BACKGROUND] Partial embedding success in {elapsed_time:.2f}s"
                )
                if isinstance(results[0], Exception):
                    print(f"   User embedding error: {results[0]}")
                if isinstance(results[1], Exception):
                    print(f"   AI embedding error: {results[1]}")

            return {
                "success": user_success and ai_success,
                "user_embedding_created": user_success,
                "ai_embedding_created": ai_success,
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(
                f"❌ [BACKGROUND] Embedding creation failed in {elapsed_time:.2f}s: {e}"
            )
            # Fail silently - don't block user experience
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time,
            }

    async def _create_embedding(
        self,
        conv_id: str,
        user_id: str,
        text: str,
        title: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Create a single embedding using langchain service"""
        try:
            return await langchain_service.upsert_embedding(
                conv_id=conv_id,
                user_id=user_id,
                text=text,
                title=title,
                metadata=metadata,
            )
        except Exception as e:
            print(f"⚠️ [BACKGROUND] Single embedding error: {e}")
            return False


async def run_simplified_background_operations(
    user_conv_id: str,
    ai_conv_id: str,
    user_id: str,
    interaction_id: str,
    message: str,
    ai_content: str,
) -> Dict[str, Any]:
    """
    Simplified background operations function

    This is the main entry point for background operations.
    It only creates embeddings - nothing else.

    Usage:
        # Fire and forget in a background task
        asyncio.create_task(
            run_simplified_background_operations(
                user_conv_id=user_conv.id,
                ai_conv_id=ai_conv.id,
                user_id=str(user.id),
                interaction_id=str(interaction.id),
                message=message,
                ai_content=ai_response,
            )
        )
    """
    service = SimplifiedBackgroundService()
    return await service.create_embeddings_for_conversation(
        user_conv_id=user_conv_id,
        ai_conv_id=ai_conv_id,
        user_id=user_id,
        interaction_id=interaction_id,
        message=message,
        ai_content=ai_content,
    )


# Global instance
simplified_background_service = SimplifiedBackgroundService()
