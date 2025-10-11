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
from app.constants.constant import RESPONSE_STATUS
from pydantic import BaseModel
from app.api.websocket_routes import notify_message_received, notify_ai_response_ready
from app.api.sse_routes import notify_message_received_sse, notify_ai_response_ready_sse
from app.core.database import AsyncSessionLocal
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Track active AI generation tasks for cancellation
active_generation_tasks: Dict[str, asyncio.Task] = {}


async def cancel_ai_generation(
    user_id: str, interaction_id: str, db: AsyncSession
) -> Dict[str, Any]:
    """Cancel ongoing AI generation for a user's interaction"""
    print(f"\n{'='*60}")
    print(f"🛑 CANCELLATION REQUESTED")
    print(f"{'='*60}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Interaction ID: {interaction_id}")
    print(f"{'='*60}\n")

    # Check if there's an active task for this interaction
    task_key = f"{user_id}_{interaction_id}"
    if task_key in active_generation_tasks:
        task = active_generation_tasks[task_key]
        if not task.done():
            task.cancel()
            print(f"✅ Cancelled active AI generation task for {task_key}")
        del active_generation_tasks[task_key]

    # Find the most recent processing conversation for this interaction
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.interaction_id == interaction_id,
            Conversation.role == ConversationRole.AI,
            Conversation.status == "processing",
        )
        .order_by(Conversation.created_at.desc())
        .limit(1)
    )
    processing_conv = result.scalar_one_or_none()

    if processing_conv:
        # Mark as cancelled
        processing_conv.status = "cancelled"
        processing_conv.error_message = "Generation stopped by user"
        processing_conv.content = {
            "type": "text",
            "result": {"content": "Response generation was stopped by user."},
        }
        await db.commit()
        print(f"✅ Marked conversation {processing_conv.id} as cancelled")

        # Send notification that generation was cancelled
        try:
            await notify_ai_response_ready_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                ai_response="Response generation was stopped by user.",
            )
            print(f"✅ Sent cancellation notification via SSE")
        except Exception as e:
            print(f"⚠️  Failed to send cancellation notification: {e}")

        return {
            "success": True,
            "message": "Generation cancelled successfully",
            "interaction_id": interaction_id,
        }
    else:
        print(f"⚠️  No processing conversation found to cancel")
        return {
            "success": False,
            "message": "No active generation found",
            "interaction_id": interaction_id,
        }


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

    # Validate input - ensure user has provided some content
    if not message or not message.strip():
        if not media_files or len(media_files) == 0:
            return {
                "success": False,
                "message": "Please provide a message or attach a file",
                "interaction_id": str(interaction.id) if interaction else None,
                "ai_response": "Please provide a message or attach a file to get help.",
            }

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

        # Commit user message immediately for instant notification
        await db.commit()

        print(f"\n🚀 USER MESSAGE STORED - Sending immediate notification")
        print(f"   User ID: {user_id}")
        print(f"   Interaction ID: {interaction.id}")
        print(f"   Conversation ID: {user_conv.id}\n")

        # Send notification IMMEDIATELY after user message is stored
        # This gives instant feedback to the user with typing indicator
        notification_task = asyncio.create_task(
            _send_notifications_background(
                str(user_id), str(interaction.id), str(user_conv.id)
            )
        )
        # Don't await - let it run in background, but keep reference
        notification_task.add_done_callback(
            lambda t: (
                print(f"✅ Message notification task completed")
                if not t.exception()
                else print(f"❌ Message notification task failed: {t.exception()}")
            )
        )

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

                # Temporarily disable cache to test fresh context retrieval
                # cached_context = await get_cached_user_context(
                #     f"{user_id}_{cache_hash}"
                # )
                # if cached_context:
                #     return cached_context.get("context", "")

                # === STEP 1: Get interaction-level semantic summary ===
                semantic_context = ""
                if interaction and interaction.semantic_summary:
                    summary_data = interaction.semantic_summary
                    print(f"📚 Using interaction semantic summary:")
                    print(f"   Topics: {summary_data.get('key_topics', [])}")
                    print(f"   Facts: {len(summary_data.get('accumulated_facts', []))}")

                    # Build rich semantic context from the summary
                    semantic_parts = []

                    # Add the main summary
                    if summary_data.get("updated_summary"):
                        semantic_parts.append(
                            f"**Conversation Summary:**\n{summary_data['updated_summary']}"
                        )

                    # Add recent focus (most important for understanding current context)
                    if summary_data.get("recent_focus"):
                        semantic_parts.append(
                            f"**Recent Focus:**\n{summary_data['recent_focus']}"
                        )

                    # Add key topics
                    if summary_data.get("key_topics"):
                        topics_str = ", ".join(summary_data["key_topics"][:5])
                        semantic_parts.append(f"**Topics Covered:** {topics_str}")

                    # Add accumulated facts (critical for answering follow-up questions)
                    if summary_data.get("accumulated_facts"):
                        facts = summary_data["accumulated_facts"][
                            :3
                        ]  # Top 3 most important
                        if facts:
                            facts_str = "\n".join(f"- {fact}" for fact in facts)
                            semantic_parts.append(f"**Key Facts:**\n{facts_str}")

                    semantic_context = "\n\n".join(semantic_parts)
                    print(
                        f"   Semantic context length: {len(semantic_context)} characters"
                    )

                # === STEP 2: Build optimized vector search query ===
                vector_query_parts = []
                if message:
                    vector_query_parts.append(message)
                if interaction and interaction.title:
                    vector_query_parts.append(f"Topic: {interaction.title}")

                # Enhance query with key topics from semantic summary for better retrieval
                if interaction and interaction.semantic_summary:
                    topics = interaction.semantic_summary.get("key_topics", [])
                    if topics:
                        vector_query_parts.append(
                            " ".join(topics[:2])
                        )  # Add top 2 topics

                vector_query = (
                    " ".join(vector_query_parts).strip() or "educational context"
                )

                # === STEP 3: Parallel similarity searches ===
                search_tasks = []

                # Interaction-specific search (highest priority - includes recent conversations)
                if interaction and interaction.id:
                    search_tasks.append(
                        langchain_service.similarity_search_by_interaction(
                            query=vector_query,
                            user_id=str(user_id),
                            interaction_id=str(interaction.id),
                            top_k=5,  # Increased for better context retrieval
                        )
                    )

                # General search (lower priority, smaller result set - finds related past conversations)
                search_tasks.append(
                    langchain_service.similarity_search(
                        query=vector_query,
                        user_id=str(user_id),
                        top_k=3,
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

                # Debug: Print what we found
                print(f"🔍 Vector DB Search Results: Found {len(similar)} results")
                for idx, m in enumerate(similar[:5]):
                    print(
                        f"  Result {idx+1}: Title='{m.get('title', '')}', Score={m.get('score', 0)}, Content Length={len(m.get('content', ''))}"
                    )

                # === STEP 4: Fast context building with priority for summaries ===
                context_parts = []
                seen_titles = set()

                for m in similar[:5]:  # Process top 5 results for better context
                    title = (m.get("title") or "").strip()
                    content = (m.get("content") or "").strip()
                    score = m.get("score", 0)
                    priority = m.get("priority", "normal")
                    metadata = m.get("metadata", {})

                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # More lenient score filtering to include relevant context
                    # Accept scores > 0.0 for interaction-specific searches
                    if score >= 0.0 or priority == "high":
                        # Check if this result has a summary in metadata (prioritize it!)
                        has_summary = metadata.get("summary")

                        # For results with summaries, include the summary instead of truncated content
                        if has_summary:
                            context_parts.append(
                                f"**{title} (Summary):**\n{has_summary}"
                            )
                        else:
                            # Moderate content truncation to preserve context
                            if len(content) > 500:
                                content = content[:500] + "..."

                            if title not in ["User message", "AI response"]:
                                context_parts.append(f"**{title}:**\n{content}")
                            else:
                                context_parts.append(content)

                # === STEP 5: Combine semantic summary with vector search results ===
                final_context_parts = []

                # Semantic summary goes FIRST (most important for understanding overall context)
                if semantic_context:
                    final_context_parts.append(semantic_context)

                # Then add specific conversation context from vector search
                if context_parts:
                    final_context_parts.append(
                        "\n\n---\n\n**Related Conversations:**\n"
                        + "\n\n---\n\n".join(context_parts[:3])
                    )

                context_text = "\n\n---\n\n".join(final_context_parts)

                # Debug: Print final context
                print(f"🔍 Final Context Length: {len(context_text)} characters")
                print(f"   Includes semantic summary: {bool(semantic_context)}")
                print(f"   Includes {len(context_parts[:3])} conversation snippets")
                if context_text:
                    print(f"🔍 Context Preview: {context_text[:300]}...")

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

        # Run guardrails and context retrieval in parallel
        guardrail_result, context_text = await asyncio.gather(
            run_guardrails(), get_context_fast(), return_exceptions=True
        )

        # Debug guardrail result
        if not isinstance(guardrail_result, Exception) and guardrail_result:
            print(f"🛡️ GUARDRAIL DEBUG - Violation: {guardrail_result.is_violation}")
            print(f"🛡️ GUARDRAIL DEBUG - Type: {guardrail_result.violation_type}")
            print(f"🛡️ GUARDRAIL DEBUG - Reasoning: {guardrail_result.reasoning}")
            print(f"🛡️ GUARDRAIL DEBUG - Message: '{message}'")
            print(f"🛡️ GUARDRAIL DEBUG - Media URLs: {media_urls}")

        # Check if guardrail is disabled via environment variable
        from app.core.config import settings

        if settings.DISABLE_GUARDRAIL:
            print(f"🛡️ GUARDRAIL DISABLED via DISABLE_GUARDRAIL environment variable")
            guardrail_result = None

        # Check if this is clearly educational content that might be incorrectly flagged
        is_clearly_educational = False
        if (
            not isinstance(guardrail_result, Exception)
            and guardrail_result
            and guardrail_result.is_violation
            and media_urls
        ):
            # Check if the message or image description suggests educational content
            educational_keywords = [
                "exercise",
                "problem",
                "solve",
                "equation",
                "math",
                "mathematics",
                "worksheet",
                "practice",
                "study",
                "homework",
                "assignment",
                "textbook",
                "notes",
                "diagram",
                "chart",
                "formula",
                "absolute value",
                "inequality",
                "algebra",
                "geometry",
                "calculus",
                "trigonometry",
            ]
            message_lower = (message or "").lower()

            # Check message text for educational keywords
            if any(keyword in message_lower for keyword in educational_keywords):
                is_clearly_educational = True
                print(
                    f"🎓 CLEARLY EDUCATIONAL CONTENT DETECTED (message) - Bypassing guardrail"
                )

            # If no message or no keywords in message, check if it's an image upload
            # For image uploads without text, we'll be more permissive if the guardrail
            # is being overly strict (which seems to be the case)
            elif not message or message.strip() == "":
                # For image-only uploads, if guardrail flags as non-educational but
                # the user is uploading to an educational platform, assume it's educational
                is_clearly_educational = True
                print(
                    f"🎓 IMAGE-ONLY UPLOAD - Assuming educational content - Bypassing guardrail"
                )

            # Additional check: if guardrail is being overly strict with educational content
            # and this is clearly a study/educational platform, be more permissive
            elif guardrail_result.violation_type == "non_educational_content":
                # If the guardrail thinks it's non-educational but we're on an educational platform,
                # and the user is uploading content (not just chatting), assume it's educational
                is_clearly_educational = True
                print(
                    f"🎓 EDUCATIONAL PLATFORM BYPASS - Assuming educational content - Bypassing guardrail"
                )

        # Handle guardrail violations quickly
        if (
            not isinstance(guardrail_result, Exception)
            and guardrail_result
            and guardrail_result.is_violation
            and not is_clearly_educational
        ):
            # Update user conversation status (already committed above)
            user_conv.status = "failed"
            user_conv.error_message = (
                f"Request blocked: {guardrail_result.violation_type}"
            )

            # Generate title for this interaction if not present
            if not interaction.title or not interaction.summary_title:
                try:
                    await _extract_interaction_metadata_fast(
                        interaction,
                        f"Content blocked: {guardrail_result.violation_type}",
                        message or "Content review",
                    )
                    # Merge and flush the interaction
                    try:
                        interaction = await db.merge(interaction)
                        await db.flush()
                        await db.refresh(interaction)
                    except Exception:
                        fresh_result = await db.execute(
                            select(Interaction).where(Interaction.id == interaction.id)
                        )
                        fresh_interaction = fresh_result.scalar_one_or_none()
                        if fresh_interaction:
                            fresh_interaction.title = interaction.title
                            fresh_interaction.summary_title = interaction.summary_title
                            await db.flush()
                            await db.refresh(fresh_interaction)
                            interaction = fresh_interaction
                except Exception as title_error:
                    print(f"Title generation error for blocked content: {title_error}")
                    pass

            # Create AI response for blocked content with user-friendly message
            violation_messages = {
                "non_educational_content": "I can only help with educational content like textbooks, notes, worksheets, and study materials. Please upload educational documents without personal photos or portraits.",
                "inappropriate_content": "This content violates our guidelines. Please ensure your uploads are appropriate and educational.",
                "code_generation": "I cannot generate code directly. However, I can help explain code concepts or analyze educational programming problems.",
            }
            user_friendly_message = violation_messages.get(
                guardrail_result.violation_type,
                "I cannot process this request. Please ensure your content is educational and appropriate.",
            )

            ai_blocked = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={
                    "type": "text",
                    "result": {"content": user_friendly_message},
                },
                status="completed",
            )
            db.add(ai_blocked)
            await db.commit()

            # Send AI response notification for blocked content
            asyncio.create_task(
                _send_ai_response_notification(
                    str(user_id),
                    str(interaction.id),
                    user_friendly_message,
                )
            )

            return {
                "success": False,
                "message": f"Request blocked: {guardrail_result.violation_type}",
                "interaction_id": str(interaction.id),
                "ai_response": user_friendly_message,
            }

        # Handle context retrieval exceptions
        if isinstance(context_text, Exception):
            context_text = ""

        # === PHASE 3: OPTIMIZED AI GENERATION ===
        print(f"🎛️ DYNAMIC TOKEN SYSTEM")
        print(f"📊 Max Tokens: {max_tokens}")
        print(f"📎 Attachments: {len(media_urls) if media_urls else 0}")
        if media_urls:
            tokens_per_attachment = (
                (max_tokens - 2000) // len(media_urls) if len(media_urls) > 0 else 0
            )
            print(f"💡 Calculated tokens per attachment: {tokens_per_attachment}")
        print("-" * 40)

        try:
            # Check if we have images that need document analysis
            if media_urls and len(media_urls) > 0:
                print("📋 DOCUMENT ANALYSIS MODE - Processing uploaded images")
                # Use document analysis for images
                analysis_results = []
                total_tokens = 0

                # Calculate tokens per attachment more accurately
                base_tokens = 2000
                attachment_tokens = (
                    max(500, (max_tokens - base_tokens) // len(media_urls))
                    if len(media_urls) > 0
                    else 1000
                )

                for media_url in media_urls:
                    print(f"🔍 Analyzing image: {media_url}")
                    print(f"📊 Using {attachment_tokens} tokens for this image")
                    analysis_result = await langchain_service.analyze_document(
                        file_url=media_url, max_tokens=attachment_tokens
                    )

                    # Check if analysis failed
                    if analysis_result.get("type") == "error":
                        error_msg = analysis_result.get("_result", {}).get(
                            "error", "Failed to analyze the uploaded content"
                        )
                        print(f"❌ Analysis failed: {error_msg}")

                        # Generate title for this interaction if not present
                        if not interaction.title or not interaction.summary_title:
                            try:
                                await _extract_interaction_metadata_fast(
                                    interaction,
                                    f"Analysis failed: {error_msg}",
                                    message or "Document upload",
                                )
                                # Merge and flush the interaction
                                try:
                                    interaction = await db.merge(interaction)
                                    await db.flush()
                                    await db.refresh(interaction)
                                except Exception:
                                    fresh_result = await db.execute(
                                        select(Interaction).where(
                                            Interaction.id == interaction.id
                                        )
                                    )
                                    fresh_interaction = (
                                        fresh_result.scalar_one_or_none()
                                    )
                                    if fresh_interaction:
                                        fresh_interaction.title = interaction.title
                                        fresh_interaction.summary_title = (
                                            interaction.summary_title
                                        )
                                        await db.flush()
                                        await db.refresh(fresh_interaction)
                                        interaction = fresh_interaction
                            except Exception as title_error:
                                print(
                                    f"Title generation error for failed analysis: {title_error}"
                                )
                                pass

                        # Create AI response for analysis failure
                        ai_error_response = Conversation(
                            interaction_id=str(interaction.id),
                            role=ConversationRole.AI,
                            content={
                                "type": "text",
                                "result": {
                                    "content": f"Sorry, I couldn't analyze the uploaded content. {error_msg}"
                                },
                            },
                            status="failed",
                            error_message=error_msg,
                        )
                        db.add(ai_error_response)
                        await db.commit()

                        # Send error notification
                        asyncio.create_task(
                            _send_ai_response_notification(
                                str(user_id),
                                str(interaction.id),
                                f"Sorry, I couldn't analyze the uploaded content. {error_msg}",
                            )
                        )

                        return {
                            "success": False,
                            "message": error_msg,
                            "interaction_id": str(interaction.id),
                            "ai_response": f"Sorry, I couldn't analyze the uploaded content. {error_msg}",
                        }

                    analysis_results.append(analysis_result)
                    total_tokens += analysis_result.get("token", 0)
                    print(f"📊 Analysis result type: {analysis_result.get('type')}")
                    print(f"📊 Analysis tokens: {analysis_result.get('token', 0)}")

                # Extract title and summary from analysis results if available
                analysis_title = None
                analysis_summary = None

                # For single image, use the analysis title/summary directly
                if len(analysis_results) == 1:
                    analysis = analysis_results[0]
                    if analysis.get("title") and not interaction.title:
                        analysis_title = analysis["title"][:50]
                    if analysis.get("summary_title") and not interaction.summary_title:
                        analysis_summary = analysis["summary_title"][:100]
                else:
                    # For multiple images, use the first analysis that has title/summary
                    for analysis in analysis_results:
                        if analysis.get("title") and not analysis_title:
                            analysis_title = analysis["title"][:50]
                        if analysis.get("summary_title") and not analysis_summary:
                            analysis_summary = analysis["summary_title"][:100]
                        if analysis_title and analysis_summary:
                            break

                # Set the extracted title and summary
                if analysis_title:
                    interaction.title = analysis_title
                    print(f"📝 Using analysis title: {analysis_title}")
                if analysis_summary:
                    interaction.summary_title = analysis_summary
                    print(f"📝 Using analysis summary: {analysis_summary}")

                # Combine analysis results
                if len(analysis_results) == 1:
                    # Single image - use the analysis directly
                    analysis = analysis_results[0]
                    if analysis.get("type") == "mcq" and analysis.get(
                        "_result", {}
                    ).get("questions"):
                        # Format MCQ content
                        questions = analysis["_result"]["questions"]
                        content_parts = []
                        for i, q in enumerate(questions, 1):
                            question_text = f"{i}. {q.get('question', '')}\n\n"
                            if q.get("options"):
                                for opt_key, opt_value in q["options"].items():
                                    question_text += f"{opt_key.upper()}. {opt_value}\n"
                            question_text += "\n"
                            if q.get("answer"):
                                question_text += f"Answer: {q['answer']}\n\n"
                            if q.get("explanation"):
                                question_text += f"Explanation: {q['explanation']}\n"
                            content_parts.append(question_text)
                        content_text = "\n".join(content_parts)
                    else:
                        # Use content from analysis
                        content_text = analysis.get("_result", {}).get(
                            "content", str(analysis)
                        )
                else:
                    # Multiple images - combine summaries
                    content_parts = []
                    for i, analysis in enumerate(analysis_results, 1):
                        content_parts.append(
                            f"Document {i}: {analysis.get('_result', {}).get('content', str(analysis))}"
                        )
                    content_text = "\n\n".join(content_parts)

                # Set token counts from analysis
                input_tokens = total_tokens // 2  # Estimate
                output_tokens = total_tokens // 2  # Estimate
                tokens_used = total_tokens

                print(f"📄 Final content type: Document Analysis")
                print(f"📄 Final content length: {len(content_text)}")

            else:
                print("💬 CONVERSATION MODE - No images, using text generation")
                # Use streaming for better perceived performance
                content_text, input_tokens, output_tokens, tokens_used = (
                    await langchain_service.generate_conversation_response(
                        message=message or "",
                        context=context_text,
                        image_urls=None,  # No images in conversation mode
                        interaction_title=interaction.title,
                        interaction_summary=interaction.summary_title,
                        max_tokens=max_tokens,
                    )
                )

            # Log raw AI generation response
            print("🚀 RAW AI GENERATION RESPONSE")
            print("-" * 50)
            print(
                f"📊 Tokens Used: {tokens_used} (Input: {input_tokens}, Output: {output_tokens})"
            )
            print(f"📝 Content Type: {type(content_text)}")
            print(f"📝 Content Length: {len(str(content_text)) if content_text else 0}")
            print("📄 Raw Content:")

            print(content_text)

            print("-" * 50)

            # Fast title extraction for new interactions (with error handling)
            try:

                if not interaction.title or not interaction.summary_title:

                    await _extract_interaction_metadata_fast(
                        interaction, content_text, message or ""
                    )
                    # Handle session attachment issue - merge the interaction into current session
                    try:
                        # If interaction is from another session, merge it into current session
                        interaction = await db.merge(interaction)
                        await db.flush()  # Force the session to register the changes
                        await db.refresh(
                            interaction
                        )  # Refresh to ensure changes are reflected
                    except Exception as session_error:

                        # Alternative: Query the interaction fresh from current session

                        fresh_result = await db.execute(
                            select(Interaction).where(Interaction.id == interaction.id)
                        )
                        fresh_interaction = fresh_result.scalar_one_or_none()
                        if fresh_interaction:
                            fresh_interaction.title = interaction.title
                            fresh_interaction.summary_title = interaction.summary_title
                            await db.flush()
                            await db.refresh(fresh_interaction)
                            # Update the reference
                            interaction = fresh_interaction

            except Exception as title_error:
                pass
                # Don't let title generation failure break the main AI response

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
            # Create a proper error message for insufficient points
            insufficient_points_message = f"You need {points_cost} coins to get an AI response, but you only have {user.current_points} coins. Please earn more coins or upgrade your plan."

            ai_failed = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={
                    "type": "text",
                    "result": {
                        "content": insufficient_points_message,
                        "error": "Insufficient points for AI response",
                    },
                },
                status="failed",
                error_message=RESPONSE_STATUS.INSUFFICIENT_BALANCE,
                tokens_used=tokens_used,
                points_cost=points_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_hidden=False,  # Make it visible so user can see the error
            )
            db.add(ai_failed)
            await db.commit()

            # Send AI response notification for insufficient points
            asyncio.create_task(
                _send_ai_response_notification(
                    str(user_id),
                    str(interaction.id),
                    insufficient_points_message,
                )
            )

            return {
                "success": False,
                "message": RESPONSE_STATUS.INSUFFICIENT_BALANCE,
                "interaction_id": str(interaction.id),
                "ai_response": insufficient_points_message,  # Include the error message
            }

        # Update user points
        user.current_points -= points_cost
        user.total_points_used += points_cost

        # Process AI content efficiently
        ai_content_type, ai_result_content = _process_ai_content_fast(content_text)

        # Log processed AI content
        print("⚙️ PROCESSED AI CONTENT")
        print("-" * 40)
        print(f"📊 Content Type: {ai_content_type}")
        print(
            f"📝 Processed Length: {len(str(ai_result_content)) if ai_result_content else 0}"
        )
        print("📄 Processed Content:")
        if ai_result_content:
            if len(str(ai_result_content)) > 300:
                print(f"{str(ai_result_content)[:300]}...")
            else:
                print(ai_result_content)
        else:
            print("No processed content")
        print("-" * 40)

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

        # Flush to get the conversation ID
        await db.flush()

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

        # Test: Query the interaction directly from DB to verify persistence

        # Send AI response notification in background
        # Create task and store reference to prevent garbage collection
        ai_notification_task = asyncio.create_task(
            _send_ai_response_notification(
                str(user_id), str(interaction.id), ai_result_content
            )
        )
        # Don't await - let it run in background, but keep reference
        ai_notification_task.add_done_callback(
            lambda t: (
                print(f"✅ AI response notification task completed")
                if not t.exception()
                else print(f"❌ AI response notification task failed: {t.exception()}")
            )
        )

        # Run embeddings and cache invalidation in background
        print(f"🔄 Starting background embeddings for interaction {interaction.id}")
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
    print(f"\n{'='*60}")
    print(f"📤 SENDING MESSAGE_RECEIVED NOTIFICATION")
    print(f"{'='*60}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Interaction ID: {interaction_id}")
    print(f"🆔 Conversation ID: {conversation_id}")
    print(f"{'='*60}\n")

    # Try WebSocket first
    try:
        await notify_message_received(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=conversation_id,
        )
        print(f"✅ WebSocket notification sent successfully for user {user_id}")
    except Exception as ws_error:
        print(f"⚠️  WebSocket notification failed: {ws_error}")
        # Fallback to SSE
        try:
            await notify_message_received_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                conversation_id=conversation_id,
            )
            print(f"✅ SSE notification sent successfully for user {user_id}")
        except Exception as sse_error:
            print(f"❌ SSE notification failed: {sse_error}")
            import traceback

            traceback.print_exc()
            print(f"❌ All notification methods failed for user {user_id}")


async def _send_ai_response_notification(
    user_id: str, interaction_id: str, ai_response: str
):
    """Send AI response notification in background without blocking main flow"""
    print(f"\n{'='*60}")
    print(f"📤 SENDING AI_RESPONSE_READY NOTIFICATION")
    print(f"{'='*60}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Interaction ID: {interaction_id}")
    print(f"📝 Response Length: {len(ai_response)} characters")
    print(f"📄 Response Preview: {ai_response[:100]}...")
    print(f"{'='*60}\n")

    # Try WebSocket first
    try:
        await notify_ai_response_ready(
            user_id=user_id,
            interaction_id=interaction_id,
            ai_response=ai_response,
        )
        print(
            f"✅ WebSocket AI response notification sent successfully for user {user_id}"
        )
    except Exception as ws_error:
        print(f"⚠️  WebSocket AI response notification failed: {ws_error}")
        # Fallback to SSE
        try:
            await notify_ai_response_ready_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                ai_response=ai_response,
            )
            print(
                f"✅ SSE AI response notification sent successfully for user {user_id}"
            )
        except Exception as sse_error:
            print(f"❌ SSE AI response notification failed: {sse_error}")
            import traceback

            traceback.print_exc()
            print(f"❌ All AI response notification methods failed for user {user_id}")


async def _extract_interaction_metadata_fast(
    interaction: Interaction, content_text: str, original_message: str = ""
):
    """Fast metadata extraction with dedicated title generation"""
    try:
        # First try to extract from JSON response (existing logic)
        if content_text.strip().startswith("{"):
            parsed_response = json.loads(content_text)
            if isinstance(parsed_response, dict):
                if "title" in parsed_response and not interaction.title:
                    interaction.title = parsed_response["title"][:50]
                if "summary_title" in parsed_response and not interaction.summary_title:
                    interaction.summary_title = parsed_response["summary_title"][:100]
                return  # If we got titles from JSON, we're done

        # If no title from JSON or plain text response, generate one using AI
        if not interaction.title or not interaction.summary_title:

            # First, try AI generation
            try:
                title, summary_title = (
                    await langchain_service.generate_interaction_title(
                        message=original_message,
                        response_preview=content_text[
                            :300
                        ],  # First 300 chars of response
                    )
                )

            except Exception as ai_title_error:

                # Fallback to simple title generation
                if original_message:
                    title = original_message[:40].strip()
                    summary_title = f"Help with {title.lower()}"
                else:
                    title = "Study Session"
                    summary_title = "Educational assistance"

            if title and not interaction.title:
                interaction.title = title

            if summary_title and not interaction.summary_title:
                interaction.summary_title = summary_title

    except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
        print(f"Metadata extraction error: {e}")
        # Final fallback: create basic title from message
        if original_message and not interaction.title:
            interaction.title = original_message[:40].strip()


def _process_ai_content_fast(content_text: str) -> tuple[str, str]:
    """Fast AI content processing with enhanced formatting and escaped character handling"""
    ai_content_type = "written"
    ai_result_content = content_text

    # Clean up escaped characters and LaTeX formatting
    if content_text:
        # Remove escaped characters that show as backslashes
        ai_result_content = content_text.replace("\\(", "(").replace("\\)", ")")
        ai_result_content = ai_result_content.replace("\\[", "[").replace("\\]", "]")
        ai_result_content = ai_result_content.replace("\\\\", "")
        ai_result_content = ai_result_content.replace("\\ ", " ")

        # Clean up common LaTeX patterns
        ai_result_content = ai_result_content.replace("\\geq", "≥")
        ai_result_content = ai_result_content.replace("\\leq", "≤")
        ai_result_content = ai_result_content.replace("\\times", "×")
        ai_result_content = ai_result_content.replace("\\div", "÷")

    # Detect MCQ patterns in plain text content
    if content_text and not content_text.strip().startswith("{"):
        # Check for MCQ patterns: numbered questions with Answer: sections
        import re

        # Pattern 1: Numbered questions with "Answer:" pattern
        answer_pattern = r"\d+\.\s+.+?Answer:\s*[A-Za-z]"
        if re.search(answer_pattern, content_text, re.MULTILINE | re.DOTALL):
            ai_content_type = "mcq"
            print(f"🎯 MCQ DETECTED: Answer pattern found")
        else:
            # Pattern 2: Multiple numbered questions with options A, B, C, D
            option_pattern = (
                r"\d+\.\s+.+?[A-D]\.\s+.+?[A-D]\.\s+.+?[A-D]\.\s+.+?[A-D]\."
            )
            if re.search(option_pattern, content_text, re.MULTILINE | re.DOTALL):
                ai_content_type = "mcq"
                print(f"🎯 MCQ DETECTED: Multiple choice options found")
            else:
                # Pattern 3: Explanation sections (common in MCQ)
                explanation_pattern = r"Explanation:\s+.+"
                if re.search(explanation_pattern, content_text, re.MULTILINE):
                    ai_content_type = "mcq"
                    print(f"🎯 MCQ DETECTED: Explanation sections found")

        print(f"🔍 MCQ Detection result: {ai_content_type}")

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
                        # Enhanced MCQ formatting
                        questions = result_content["questions"][
                            :3
                        ]  # Limit to 3 questions
                        formatted_questions = []

                        for i, q in enumerate(questions, 1):
                            question_text = q.get("question", "")
                            options = q.get("options", {})
                            answer = q.get("answer", "")
                            explanation = q.get("explanation", "")

                            # Format question with better structure
                            formatted_q = f"{i}. {question_text}\n\n"

                            # Format options
                            for opt_key, opt_value in list(options.items())[:4]:
                                formatted_q += f"{opt_key.upper()}. {str(opt_value)}\n"

                            formatted_q += "\n"

                            # Add answer with clear formatting
                            if answer:
                                formatted_q += f"Answer: {answer.upper()}\n\n"

                            # Add explanation with clear formatting
                            if explanation:
                                formatted_q += f"Explanation: {explanation}\n"

                            formatted_questions.append(formatted_q)

                        ai_result_content = "\n".join(formatted_questions)

        # Apply character cleaning to any processed content
        if ai_result_content:
            ai_result_content = ai_result_content.replace("\\(", "(").replace(
                "\\)", ")"
            )
            ai_result_content = ai_result_content.replace("\\[", "[").replace(
                "\\]", "]"
            )
            ai_result_content = ai_result_content.replace("\\\\", "")
            ai_result_content = ai_result_content.replace("\\ ", " ")

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
    """Run embeddings, summarization, and cache operations in background"""
    try:
        print(f"📦 Background operations started for interaction {interaction_id}")
        print(f"   User message length: {len(message) if message else 0}")
        print(f"   AI content length: {len(ai_content) if ai_content else 0}")

        # === STEP 1: Create semantic summary ===
        conversation_summary = None
        if message and ai_content:
            try:
                print(f"   📊 Creating semantic summary...")
                conversation_summary = await langchain_service.summarize_conversation(
                    user_message=message, ai_response=ai_content
                )
                print(f"   ✅ Semantic summary created")
            except Exception as sum_error:
                print(f"   ⚠️  Semantic summary creation failed: {sum_error}")

        # === STEP 2: Update interaction-level running summary ===
        try:
            print(f"   🔄 Updating interaction-level summary...")
            async with AsyncSessionLocal() as db:
                # Get current interaction
                result = await db.execute(
                    select(Interaction).where(Interaction.id == str(interaction_id))
                )
                interaction = result.scalar_one_or_none()

                if interaction:
                    current_summary = interaction.semantic_summary

                    # Update the running summary
                    updated_summary = (
                        await langchain_service.update_interaction_summary(
                            current_summary=current_summary,
                            new_user_message=message,
                            new_ai_response=ai_content,
                        )
                    )

                    # Save updated summary to database
                    interaction.semantic_summary = updated_summary
                    await db.commit()

                    print(f"   ✅ Interaction summary updated and saved")
                    print(f"      Topics: {len(updated_summary.get('key_topics', []))}")
                    print(
                        f"      Facts: {len(updated_summary.get('accumulated_facts', []))}"
                    )
        except Exception as sum_update_error:
            print(f"   ⚠️  Interaction summary update failed: {sum_update_error}")

        # === STEP 3: Create embeddings with enhanced metadata ===
        embedding_tasks = []

        # Prepare metadata with summary information
        enhanced_metadata = {
            "interaction_id": str(interaction_id),
        }

        if conversation_summary:
            enhanced_metadata["summary"] = conversation_summary.get(
                "semantic_summary", ""
            )
            enhanced_metadata["topics"] = conversation_summary.get("main_topics", [])
            enhanced_metadata["key_facts"] = conversation_summary.get("key_facts", [])

        if message:
            print(
                f"   ✅ Creating embedding for user message (conv_id: {user_conv_id})"
            )
            user_metadata = {**enhanced_metadata, "type": "user_message"}
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(user_conv_id),
                    user_id=str(user_id),
                    text=message,  # Full message content
                    title="User message",
                    metadata=user_metadata,
                )
            )

        if ai_content:
            print(f"   ✅ Creating embedding for AI response (conv_id: {ai_conv_id})")
            ai_metadata = {**enhanced_metadata, "type": "ai_response"}
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(ai_conv_id),
                    user_id=str(user_id),
                    text=ai_content,  # Full AI content
                    title="AI response",
                    metadata=ai_metadata,
                )
            )

        # Run embeddings in parallel
        if embedding_tasks:
            print(
                f"   🚀 Running {len(embedding_tasks)} embedding tasks in parallel..."
            )
            results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

            # Check for errors
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"   ❌ Embedding task {idx+1} failed: {result}")
                else:
                    print(f"   ✅ Embedding task {idx+1} completed successfully")
        else:
            print(f"   ⚠️  No embedding tasks created (empty message/content)")

        # === STEP 4: Invalidate cache ===
        print(f"   🔄 Invalidating user cache for user {user_id}")
        await invalidate_user_cache(str(user_id))

        print(
            f"✅ Background operations completed successfully for interaction {interaction_id}"
        )

    except Exception as e:
        print(f"❌ Background operations error for interaction {interaction_id}: {e}")
        import traceback

        traceback.print_exc()
        pass
