from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict, Any
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.models.subscription import PointTransaction
from app.services.langchain_service import langchain_service
from app.services.cache_service import (
    cache_user_context,
    get_cached_user_context,
    cache_interaction_data,
    get_cached_interaction_data,
    invalidate_user_cache,
    invalidate_interaction_cache,
)
import json
from app.core.config import settings
from pydantic import BaseModel
from app.api.websocket_routes import notify_message_received
from app.api.sse_routes import notify_message_received_sse
from app.core.database import AsyncSessionLocal
import asyncio
from concurrent.futures import ThreadPoolExecutor


async def process_conversation_message(
    *,
    user: User,
    interaction: Optional[Interaction],
    message: Optional[str],
    media_files: Optional[List[Dict[str, str]]],
    max_tokens: int = 800,  # Reduced from 1000
) -> Dict[str, Any]:
    """
    Optimized conversation processor with parallel operations and aggressive caching.
    """

    # Use a single DB session for the entire operation
    async with AsyncSessionLocal() as db:
        user_id = user.id
        interaction_id = interaction.id if interaction else None

        # === PHASE 1: PARALLEL SETUP & MEDIA PROCESSING ===
        async def process_media_parallel():
            """Process media files in parallel"""
            if not media_files:
                return [], []

            media_objects = []
            media_urls = []

            # Process all media files concurrently
            media_tasks = []
            for media_file in media_files:
                media_id = media_file.get("id")
                media_url = media_file.get("url")

                if media_url:
                    # Direct URL - create mock media object
                    mock_media = Media(
                        id=media_id,
                        s3_key=media_url,
                        original_filename=f"media_{media_id}",
                        file_type="image/jpeg",
                        original_size=0,
                    )
                    media_objects.append(mock_media)
                    media_urls.append(media_url)
                else:
                    # Queue database lookup
                    media_tasks.append(
                        db.execute(select(Media).where(Media.id == media_id))
                    )

            # Execute all DB queries in parallel
            if media_tasks:
                media_results = await asyncio.gather(
                    *media_tasks, return_exceptions=True
                )
                for result in media_results:
                    if not isinstance(result, Exception):
                        media = result.scalar_one_or_none()
                        if media:
                            media_objects.append(media)
                            media_url = f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{media.s3_key}"
                            media_urls.append(media_url)

            return media_objects, media_urls

        async def create_user_conversation():
            """Create user conversation record"""
            user_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.USER,
                content={
                    "type": "text",
                    "result": {"content": message or ""},
                },
                status="processing",
            )
            db.add(user_conv)
            await db.flush()
            return user_conv

        # Run media processing and user conversation creation in parallel
        (media_objects, media_urls), user_conv = await asyncio.gather(
            process_media_parallel(), create_user_conversation()
        )

        # Associate media files if present (non-blocking)
        if media_objects:
            from app.models.interaction import conversation_files

            media_tasks = [
                db.execute(
                    conversation_files.insert().values(
                        conversation_id=str(user_conv.id), media_id=str(media_obj.id)
                    )
                )
                for media_obj in media_objects
            ]
            await asyncio.gather(*media_tasks, return_exceptions=True)

        # === PHASE 2: PARALLEL VALIDATION & CONTEXT RETRIEVAL ===
        async def run_guardrails():
            """Run guardrail checks"""
            if not (message or media_urls):
                return None
            try:
                return await langchain_service.check_guardrails(
                    message or "", media_urls
                )
            except Exception:
                return None

        async def get_context_fast():
            """Fast context retrieval with aggressive caching"""
            try:
                # Check cache first - use a hash of message + interaction info for better cache hits
                cache_key_components = [
                    str(user_id),
                    message or "",
                    interaction.title or "",
                    str(interaction.id) if interaction else "",
                ]
                cache_hash = str(hash("".join(cache_key_components)))

                cached_context = await get_cached_user_context(
                    f"{user_id}_{cache_hash}"
                )
                if cached_context:
                    return cached_context.get("context", "")

                # Build optimized query
                vector_query_parts = []
                if message:
                    vector_query_parts.append(message)
                if interaction and interaction.title:
                    vector_query_parts.append(f"Topic: {interaction.title}")

                vector_query = (
                    " ".join(vector_query_parts).strip() or "educational context"
                )

                # Parallel similarity searches with reduced top_k
                search_tasks = []

                # Interaction-specific search (highest priority)
                if interaction and interaction.id:
                    search_tasks.append(
                        langchain_service.similarity_search_by_interaction(
                            query=vector_query,
                            user_id=str(user_id),
                            interaction_id=str(interaction.id),
                            top_k=2,
                        )
                    )

                # General search (lower priority, smaller result set)
                search_tasks.append(
                    langchain_service.similarity_search(
                        query=vector_query,
                        user_id=str(user_id),
                        top_k=3,  # Reduced from 4
                    )
                )

                # Execute searches in parallel
                search_results = await asyncio.gather(
                    *search_tasks, return_exceptions=True
                )

                # Combine results efficiently
                similar = []
                for result in search_results:
                    if not isinstance(result, Exception) and result:
                        similar.extend(result)

                # Fast context building with aggressive limits
                context_parts = []
                seen_titles = set()

                for m in similar[:3]:  # Only process top 3 results
                    title = (m.get("title") or "").strip()
                    content = (m.get("content") or "").strip()
                    score = m.get("score", 0)
                    priority = m.get("priority", "normal")

                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # Faster score filtering
                    if score > (0.3 if priority == "high" else 0.4):
                        # Aggressive content truncation
                        if len(content) > 150:  # Very short content
                            content = content[:150] + "..."

                        if title not in ["User message", "AI response"]:
                            context_parts.append(f"**{title}**\n{content}")
                        else:
                            context_parts.append(content)

                context_text = "\n\n---\n\n".join(
                    context_parts[:2]
                )  # Max 2 context pieces

                # Cache aggressively
                await cache_user_context(
                    f"{user_id}_{cache_hash}",
                    {"context": context_text},
                    ttl=600,  # 10 minutes
                )

                return context_text
            except Exception as e:
                print(f"Fast context retrieval error: {e}")
                return ""

        # Send notification in background (non-blocking)
        asyncio.create_task(
            _send_notifications_background(
                str(user_id), str(interaction.id), str(user_conv.id)
            )
        )

        # Run guardrails and context retrieval in parallel
        guardrail_result, context_text = await asyncio.gather(
            run_guardrails(), get_context_fast(), return_exceptions=True
        )

        # Handle guardrail violations quickly
        if (
            not isinstance(guardrail_result, Exception)
            and guardrail_result
            and guardrail_result.is_violation
        ):
            user_conv.status = "failed"
            user_conv.error_message = (
                f"Request blocked: {guardrail_result.violation_type}"
            )
            await db.commit()
            return {
                "success": False,
                "message": f"Request blocked: {guardrail_result.violation_type}",
                "interaction_id": str(interaction.id),
                "ai_response": None,
            }

        # Handle context retrieval exceptions
        if isinstance(context_text, Exception):
            context_text = ""

        # === PHASE 3: OPTIMIZED AI GENERATION ===
        try:
            # Use streaming for better perceived performance
            content_text, input_tokens, output_tokens, tokens_used = (
                await langchain_service.generate_conversation_response(
                    message=message or "",
                    context=context_text,
                    image_urls=media_urls if media_urls else None,
                    interaction_title=interaction.title,
                    interaction_summary=interaction.summary_title,
                    max_tokens=max_tokens,
                )
            )

            # Fast title extraction for new interactions
            if not interaction.title and not interaction.summary_title:
                await _extract_interaction_metadata_fast(interaction, content_text)

        except Exception as e:
            ai_failed = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={"type": "other", "result": {"error": str(e)}},
                status="failed",
                error_message=str(e),
            )
            db.add(ai_failed)
            await db.commit()
            return {
                "success": False,
                "message": "AI generation failed",
                "ai_response": None,
            }

        # === PHASE 4: FAST RESPONSE PROCESSING ===
        points_cost = langchain_service.calculate_points_cost(tokens_used)

        # Quick points check
        if user.current_points < points_cost:
            ai_content_type, ai_result_content = _process_ai_content_fast(content_text)

            ai_failed = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={
                    "type": ai_content_type,
                    "result": {
                        "content": ai_result_content,
                        "error": "Insufficient points for AI response",
                    },
                },
                status="failed",
                error_message="Insufficient points",
                tokens_used=tokens_used,
                points_cost=points_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_hidden=True,
            )
            db.add(ai_failed)
            await db.commit()
            return {
                "success": False,
                "message": "Insufficient points",
                "interaction_id": str(interaction.id),
            }

        # Update user points
        user.current_points -= points_cost
        user.total_points_used += points_cost

        # Process AI content efficiently
        ai_content_type, ai_result_content = _process_ai_content_fast(content_text)

        # Create AI conversation
        ai_conv = Conversation(
            interaction_id=str(interaction.id),
            role=ConversationRole.AI,
            content={"type": ai_content_type, "result": {"content": ai_result_content}},
            status="completed",
            tokens_used=tokens_used,
            points_cost=points_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        db.add(ai_conv)

        # Create point transaction
        pt = PointTransaction(
            user_id=str(user_id),
            transaction_type="used",
            points=points_cost,
            description=f"Chat response - {tokens_used} tokens",
            conversation_id=str(ai_conv.id),
        )
        db.add(pt)

        # === PHASE 5: BACKGROUND OPERATIONS ===
        # Commit first for faster response
        await db.commit()

        # Run embeddings and cache invalidation in background
        asyncio.create_task(
            _background_operations(
                user_conv.id,
                ai_conv.id,
                user_id,
                interaction.id,
                message,
                ai_result_content,
            )
        )

        return {
            "success": True,
            "message": "Conversation updated",
            "interaction_id": str(interaction.id),
            "ai_response": ai_result_content,
        }


async def _send_notifications_background(
    user_id: str, interaction_id: str, conversation_id: str
):
    """Send notifications in background without blocking main flow"""
    try:
        await notify_message_received(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=conversation_id,
        )
    except Exception:
        try:
            await notify_message_received_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                conversation_id=conversation_id,
            )
        except Exception:
            pass


async def _extract_interaction_metadata_fast(
    interaction: Interaction, content_text: str
):
    """Fast metadata extraction with error handling"""
    try:
        if content_text.strip().startswith("{"):
            parsed_response = json.loads(content_text)
            if isinstance(parsed_response, dict):
                if "title" in parsed_response and not interaction.title:
                    interaction.title = parsed_response["title"][:100]  # Limit length
                if "summary_title" in parsed_response and not interaction.summary_title:
                    interaction.summary_title = parsed_response["summary_title"][:200]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass


def _process_ai_content_fast(content_text: str) -> tuple[str, str]:
    """Fast AI content processing with caching"""
    ai_content_type = "written"
    ai_result_content = content_text

    try:
        if content_text and content_text.strip().startswith("{"):
            parsed_response = json.loads(content_text)
            if isinstance(parsed_response, dict):
                ai_content_type = parsed_response.get("type", "written")
                result_content = parsed_response.get("result", {})

                if isinstance(result_content, dict):
                    if "content" in result_content:
                        ai_result_content = result_content["content"]
                    elif "questions" in result_content:
                        # Fast MCQ formatting
                        questions = result_content["questions"][
                            :3
                        ]  # Limit to 3 questions
                        formatted_questions = []

                        for q in questions:
                            question_text = q.get("question", "")[:200]  # Limit length
                            options = q.get("options", {})
                            answer = q.get("answer", "")
                            explanation = q.get("explanation", "")[
                                :100
                            ]  # Limit explanation

                            formatted_q = f"**{question_text}**\n"
                            for opt_key, opt_value in list(options.items())[
                                :4
                            ]:  # Max 4 options
                                formatted_q += f"{opt_key}. {str(opt_value)[:100]}\n"
                            if answer:
                                formatted_q += f"**Answer:** {answer}\n"
                            if explanation:
                                formatted_q += f"**Explanation:** {explanation}\n"
                            formatted_questions.append(formatted_q)

                        ai_result_content = "\n\n".join(formatted_questions)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ai_content_type, ai_result_content


async def _background_operations(
    user_conv_id: int,
    ai_conv_id: int,
    user_id: str,
    interaction_id: str,
    message: str,
    ai_content: str,
):
    """Run embeddings and cache operations in background"""
    try:
        # Create embeddings in parallel
        embedding_tasks = []

        if message:
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(user_conv_id),
                    user_id=str(user_id),
                    text=message[:500],  # Truncate for faster processing
                    title="User message",
                    metadata={
                        "interaction_id": str(interaction_id),
                        "type": "user_message",
                    },
                )
            )

        if ai_content:
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(ai_conv_id),
                    user_id=str(user_id),
                    text=ai_content[:500],  # Truncate for faster processing
                    title="AI response",
                    metadata={
                        "interaction_id": str(interaction_id),
                        "type": "ai_response",
                    },
                )
            )

        # Run embeddings in parallel
        if embedding_tasks:
            await asyncio.gather(*embedding_tasks, return_exceptions=True)

        # Invalidate cache
        await invalidate_user_cache(str(user_id))

    except Exception as e:
        print(f"Background operations error: {e}")
        pass
