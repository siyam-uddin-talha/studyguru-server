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
from app.api.websocket_routes import notify_message_received, notify_ai_response_ready
from app.api.sse_routes import notify_message_received_sse, notify_ai_response_ready_sse
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
        asyncio.create_task(
            _send_ai_response_notification(
                str(user_id), str(interaction.id), ai_result_content
            )
        )

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


async def _send_ai_response_notification(
    user_id: str, interaction_id: str, ai_response: str
):
    """Send AI response notification in background without blocking main flow"""
    try:
        await notify_ai_response_ready(
            user_id=user_id,
            interaction_id=interaction_id,
            ai_response=ai_response,
        )
    except Exception:
        try:
            await notify_ai_response_ready_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                ai_response=ai_response,
            )
        except Exception:
            pass


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
