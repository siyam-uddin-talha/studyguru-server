from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import re
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.models.subscription import PointTransaction
from app.services.langchain_service import langchain_service
from app.services.document_integration_service import DocumentIntegrationService
from app.config.langchain_config import StudyGuruConfig

from app.core.database import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
import asyncio


def normalize_markdown_format(content: str) -> str:
    """
    Normalize markdown format from different LLMs to a consistent format.
    Converts formats like:
    - "1) Title" or "1. Title" → "### 1. **Title**"
    - Ensures consistent header and list formatting
    """
    if not content or not isinstance(content, str):
        return content

    lines = content.split("\n")
    normalized_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            normalized_lines.append("")
            i += 1
            continue

        # Check for numbered items with parenthesis: "1) Title" (Gemini format)
        paren_match = re.match(r"^(\d+)\)\s*(.+)$", line)
        if paren_match:
            number = paren_match.group(1)
            title = paren_match.group(2).strip()
            # Remove existing markdown formatting to avoid double bold
            title = title.replace("**", "").strip()
            # Add bold formatting
            normalized_lines.append(f"### {number}. **{title}**")
            i += 1
            continue

        # Check for numbered items with period at start of line: "1. Title" (but not headers)
        # Only convert if it's not already a header (doesn't start with #)
        period_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if period_match and not line.startswith("#"):
            number = period_match.group(1)
            title = period_match.group(2).strip()

            # Check if this looks like a main item (starts with capital, has substantial content)
            # and the next few lines contain details (like **Date:**, **Agency:**, etc.)
            is_main_item = False
            if i + 1 < len(lines):
                next_lines = "\n".join(lines[i + 1 : i + 4]).lower()
                if any(
                    keyword in next_lines
                    for keyword in [
                        "**date:**",
                        "**agency:**",
                        "**details:**",
                        "**what's new:**",
                        "**official",
                        "**discovery date:**",
                    ]
                ):
                    is_main_item = True

            # Convert to header format if it's a main item
            if is_main_item:
                # Clean up existing markdown in title
                clean_title = title.replace("**", "").strip()
                normalized_lines.append(f"### {number}. **{clean_title}**")
            else:
                # Keep as is for sub-items
                normalized_lines.append(lines[i])
            i += 1
            continue

        # Keep other lines as is
        normalized_lines.append(lines[i])
        i += 1

    return "\n".join(normalized_lines)


# Global task tracking for AI generation cancellation
active_generation_tasks: Dict[str, asyncio.Task] = {}

# Initialize document integration service
document_integration_service = DocumentIntegrationService()


def detect_mindmap_request(message: str) -> bool:
    """Detect if user wants a mindmap"""
    if not message:
        return False

    message_lower = message.lower()
    mindmap_keywords = [
        "mindmap",
        "mind map",
        "concept map",
        "visualize",
        "diagram of",
        "map out",
        "visual representation",
    ]
    return any(keyword in message_lower for keyword in mindmap_keywords)


def extract_topic_from_message(message: str) -> str:
    """Extract topic from mindmap request"""
    # Simple extraction - can be enhanced with NLP
    patterns = [
        r"mindmap (?:of |about |for )?(.+)",
        r"(?:create|make|generate) (?:a )?mindmap (?:of |about |for )?(.+)",
        r"visualize (.+)",
        r"(?:map out|diagram of) (.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return message  # Fallback to full message


def serialize_mindmap_tree(node) -> Dict[str, Any]:
    """Convert MindmapNode to serializable dict"""
    return {
        "id": node.id,
        "content": node.content,
        "level": node.level,
        "parent_id": node.parent_id,
        "color": node.color,
        "children": [serialize_mindmap_tree(child) for child in node.children],
    }


async def process_conversation_message(
    *,
    user: User,
    interaction: Optional[Interaction],
    message: Optional[str],
    media_files: Optional[List[Dict[str, str]]],
    max_tokens: Optional[int] = None,  # Will be calculated dynamically if not provided
    db: Optional[AsyncSession] = None,  # Database session parameter
    visualize_model: Optional[str] = None,  # Model for image/document analysis
    assistant_model: Optional[str] = None,  # Model for text conversation
) -> Dict[str, Any]:
    """
    Optimized conversation processor with parallel operations and aggressive caching.
    """

    # Get user's subscription plan for model validation
    subscription_plan = None
    if user.purchased_subscription:
        subscription_plan = user.purchased_subscription.subscription.subscription_plan

    print(f"🔐 User subscription plan: {subscription_plan}")
    print(f"🤖 Visualize model: {visualize_model or 'default'}")
    print(f"💬 Assistant model: {assistant_model or 'default'}")

    # Validate input - ensure user has provided some content
    if not message or not message.strip():
        if not media_files or len(media_files) == 0:
            return {
                "success": False,
                "message": "Please provide a message or attach a file",
                "interaction_id": str(interaction.id) if interaction else None,
                "ai_response": "Please provide a message or attach a file to get help.",
            }

    # Calculate dynamic token limit based on file count
    if max_tokens is None:
        file_count = len(media_files) if media_files else 0
        max_tokens = StudyGuruConfig.calculate_dynamic_tokens(file_count)
        print(f"🔢 Dynamic token calculation: {file_count} files = {max_tokens} tokens")

    # Use provided DB session or create a new one
    should_close_db = False
    if db is None:
        db = AsyncSessionLocal()
        should_close_db = True

    try:
        user_id = user.id
        interaction_id = interaction.id if interaction else None

        # If interaction was passed from resolver, merge it into this session
        if interaction:
            interaction = await db.merge(interaction)

            # === PHASE 1: PARALLEL SETUP & MEDIA PROCESSING ===
            async def process_media_parallel():
                """Process media files in parallel"""
                if not media_files:
                    return [], []

                media_objects = []
                media_urls = []

                # Process all media files concurrently
                media_ids_to_fetch = []

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
                        # Collect media IDs for batch fetch
                        media_ids_to_fetch.append(media_id)

                # Single optimized query for all media files by ID
                if media_ids_to_fetch:
                    result = await db.execute(
                        select(Media).where(Media.id.in_(media_ids_to_fetch))
                    )
                    fetched_media = result.scalars().all()

                    for media in fetched_media:
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
                            conversation_id=str(user_conv.id),
                            media_id=str(media_obj.id),
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
            import time

            phase2_start = time.time()
            print(
                f"🚀 PHASE 2 START: Guardrails + Context Retrieval at {time.time():.3f}"
            )

            async def run_guardrails():
                """Run guardrail checks with timeout"""
                guardrail_start = time.time()
                print(f"🛡️ GUARDRAIL START: {guardrail_start:.3f}")

                if not (message or media_urls):
                    print(f"🛡️ GUARDRAIL SKIP: No content to check")
                    return None
                try:
                    # OPTIMIZATION: Add timeout for guardrail check (increased for reliability)
                    result = await asyncio.wait_for(
                        langchain_service.check_guardrails(
                            message or "",
                            media_urls,
                            assistant_model=assistant_model,
                            subscription_plan=subscription_plan,
                        ),
                        timeout=3.0,  # Increased from 2.0 to 3.0 seconds
                    )
                    guardrail_end = time.time()
                    print(
                        f"🛡️ GUARDRAIL COMPLETE: {guardrail_end:.3f} (took {guardrail_end - guardrail_start:.3f}s)"
                    )
                    return result
                except asyncio.TimeoutError:
                    guardrail_end = time.time()
                    print(
                        f"🛡️ GUARDRAIL TIMEOUT: {guardrail_end:.3f} (took {guardrail_end - guardrail_start:.3f}s) - continuing"
                    )
                    return None
                except Exception as e:
                    guardrail_end = time.time()
                    print(
                        f"🛡️ GUARDRAIL ERROR: {guardrail_end:.3f} (took {guardrail_end - guardrail_start:.3f}s) - {e}"
                    )
                    return None

            async def get_context_fast():
                """
                Streamlined context retrieval with only 2 sources

                Uses simplified_context_service for ~60% faster retrieval:
                - Only 2 sources: document content + vector search
                - No query expansion (semantic embeddings already capture meaning)
                - Reduced top_k from 10 to 5
                - Max context 4000 chars (was 8000)
                """
                context_start = time.time()
                print(f"🔍 CONTEXT START (SIMPLIFIED): {context_start:.3f}")

                try:
                    from app.services.simplified_context_service import (
                        simplified_context_service,
                    )

                    # SIMPLIFIED: Only 2 sources, 4000 char limit, faster retrieval
                    context_result = await asyncio.wait_for(
                        simplified_context_service.get_simplified_context(
                            user_id=str(user_id),
                            interaction_id=str(interaction.id) if interaction else None,
                            message=message or "",
                            max_context_length=4000,  # Reduced from 8000 for focus
                        ),
                        timeout=2.0,  # Reduced from 3.5 - simplified is faster
                    )

                    context_text = context_result.get("context", "")
                    metadata = context_result.get("metadata", {})

                    context_end = time.time()
                    print(
                        f"🔍 CONTEXT COMPLETE (SIMPLIFIED): {context_end:.3f} (took {context_end - context_start:.3f}s)"
                    )

                    print(f"🔍 Simplified Context Retrieval Results:")
                    print(f"   Sources used: {metadata.get('sources_used', [])}")
                    print(f"   Context length: {len(context_text)} characters")
                    if metadata.get("question_numbers_detected"):
                        print(
                            f"   Question numbers detected: {metadata['question_numbers_detected']}"
                        )

                    return context_text
                except asyncio.TimeoutError:
                    context_end = time.time()
                    print(
                        f"🔍 CONTEXT TIMEOUT: {context_end:.3f} (took {context_end - context_start:.3f}s) - continuing without context"
                    )
                    return ""
                except Exception as e:
                    context_end = time.time()
                    print(
                        f"🔍 CONTEXT ERROR: {context_end:.3f} (took {context_end - context_start:.3f}s) - {e}"
                    )
                    return ""

            # Run guardrails and context retrieval in parallel with overall timeout
            print(f"🔄 Starting parallel guardrails + context retrieval...")
            try:
                guardrail_result, context_text = await asyncio.wait_for(
                    asyncio.gather(
                        run_guardrails(), get_context_fast(), return_exceptions=True
                    ),
                    timeout=5.0,  # Increased from 2.5 to 5.0 seconds for reliability
                )
            except asyncio.TimeoutError:
                print(f"⏱️ Phase 2 timed out - continuing with available results")
                guardrail_result = None
                context_text = ""

            phase2_end = time.time()
            print(
                f"✅ PHASE 2 COMPLETE: {phase2_end:.3f} (took {phase2_end - phase2_start:.3f}s)"
            )

            print(f"🔍 Context text: {context_text}")

            # Debug guardrail result
            if not isinstance(guardrail_result, Exception) and guardrail_result:
                print(f"🛡️ GUARDRAIL DEBUG - Violation: {guardrail_result.is_violation}")
                print(f"🛡️ GUARDRAIL DEBUG - Type: {guardrail_result.violation_type}")
                print(f"🛡️ GUARDRAIL DEBUG - Reasoning: {guardrail_result.reasoning}")
            else:
                print(f"🛡️ GUARDRAIL DEBUG - No violation detected")

            # Check if context retrieval failed
            if isinstance(context_text, Exception):
                print(f"⚠️ Context retrieval failed: {context_text}")
                context_text = ""
            elif not context_text:
                print(f"⚠️ No context retrieved")
            else:
                print(
                    f"✅ Context retrieved successfully: {len(context_text)} characters"
                )

            # Check if guardrail violation occurred
            if (
                not isinstance(guardrail_result, Exception)
                and guardrail_result
                and guardrail_result.is_violation
            ):
                print(f"🛡️ GUARDRAIL VIOLATION - Blocking request")
                return {
                    "success": False,
                    "message": "Content violates platform policies",
                    "violation_type": guardrail_result.violation_type,
                    "reasoning": guardrail_result.reasoning,
                }

            # Generate AI response
            print(f"🤖 Generating AI response...")
            print(f"   Message: {message[:100]}..." if message else "   No message")
            print(f"   Context length: {len(context_text)} characters")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Media files: {len(media_urls) if media_urls else 0}")

            # Check if we have images that need document analysis
            if (
                media_urls
                and len(media_urls) > 0
                and settings.ENABLE_ENHANCED_PROCESSING
            ):
                doc_processing_start = time.time()
                print(f"📋 DOCUMENT PROCESSING START: {doc_processing_start:.3f}")
                print(
                    "📋 ENHANCED DOCUMENT PROCESSING MODE - Processing uploaded images"
                )

                # Use the new document integration service for comprehensive processing
                from app.services.document_integration_service import (
                    document_integration_service,
                )

                analysis_results = []
                total_tokens = 0
                processed_documents = []

                # Calculate tokens per attachment more accurately
                base_tokens = 2000
                attachment_tokens = (
                    max(500, (max_tokens - base_tokens) // len(media_urls))
                    if len(media_urls) > 0
                    else 1000
                )

                print(f"📊 Document processing config:")
                print(f"   Media URLs: {len(media_urls)}")
                print(f"   Attachment tokens: {attachment_tokens}")
                print(f"   Base tokens: {base_tokens}")
                print(f"   Max tokens: {max_tokens}")

                # Process all documents in parallel for much faster processing
                async def process_single_document(
                    i: int, media_url: str
                ) -> Dict[str, Any]:
                    """Process a single document with error handling"""
                    doc_start = time.time()
                    print(f"🔍 DOC {i+1} START: {doc_start:.3f} - {media_url}")
                    print(f"📊 Using {attachment_tokens} tokens for this document")

                    try:
                        # Get media object for this URL
                        media_obj = media_objects[i] if i < len(media_objects) else None
                        media_id = media_obj.id if media_obj else f"media_{i}"

                        # Use comprehensive document processing
                        analysis_start = time.time()
                        print(f"🔍 DOC {i+1} ANALYSIS START: {analysis_start:.3f}")

                        document_analysis = await document_integration_service.process_document_comprehensive(
                            media_id=media_id,
                            interaction_id=str(interaction.id),
                            user_id=str(user_id),
                            file_url=media_url,
                            max_tokens=attachment_tokens,
                            visualize_model=visualize_model,
                            subscription_plan=subscription_plan,
                        )

                        analysis_end = time.time()
                        print(
                            f"🔍 DOC {i+1} ANALYSIS COMPLETE: {analysis_end:.3f} (took {analysis_end - analysis_start:.3f}s)"
                        )

                        # Convert to expected format for backward compatibility
                        analysis_result = {
                            "type": document_analysis.document_type,
                            "language": document_analysis.metadata.get(
                                "language", "english"
                            ),
                            "title": document_analysis.metadata.get(
                                "title", f"Document {i+1}"
                            ),
                            "summary_title": document_analysis.content_summary,
                            "token": attachment_tokens,
                            "_result": {
                                "content": document_analysis.full_content,
                                "questions": [
                                    {
                                        "question": q.question_text,
                                        "options": (
                                            q.metadata.get("options", {})
                                            if q.metadata
                                            else {}
                                        ),
                                        "answer": (
                                            q.metadata.get("answer", "")
                                            if q.metadata
                                            else ""
                                        ),
                                        "explanation": (
                                            q.metadata.get("explanation", "")
                                            if q.metadata
                                            else ""
                                        ),
                                    }
                                    for q in document_analysis.chunks
                                    if q.chunk_type == "question"
                                ],
                                "document_analysis": {
                                    "total_questions": document_analysis.total_questions,
                                    "main_topics": document_analysis.main_topics,
                                    "difficulty_level": document_analysis.difficulty_level,
                                    "subject_area": document_analysis.subject_area,
                                    "key_concepts": document_analysis.key_concepts,
                                    "chunk_count": len(document_analysis.chunks),
                                },
                            },
                        }

                        doc_end = time.time()
                        print(
                            f"✅ DOC {i+1} COMPLETE: {doc_end:.3f} (took {doc_end - doc_start:.3f}s)"
                        )
                        print(f"   Type: {document_analysis.document_type}")
                        print(f"   Questions: {document_analysis.total_questions}")
                        print(f"   Topics: {document_analysis.main_topics}")
                        print(f"   Chunks: {len(document_analysis.chunks)}")

                        return analysis_result, document_analysis

                    except Exception as doc_error:
                        print(
                            f"❌ Document processing failed for {media_url}: {doc_error}"
                        )
                        # Fallback to basic analysis
                        try:
                            basic_analysis = await langchain_service.analyze_document(
                                file_url=media_url,
                                max_tokens=attachment_tokens,
                                visualize_model=visualize_model,
                                subscription_plan=subscription_plan,
                            )
                            if basic_analysis.get("type") != "error":
                                return basic_analysis, None
                        except Exception as basic_error:
                            print(f"❌ Basic analysis also failed: {basic_error}")

                        # Return error result
                        return {
                            "type": "error",
                            "language": "unknown",
                            "title": f"Document {i+1}",
                            "summary_title": "Processing failed",
                            "token": 0,
                            "_result": {
                                "error": f"Unable to process document: {str(doc_error)}",
                                "details": (
                                    str(basic_error)
                                    if "basic_error" in locals()
                                    else str(doc_error)
                                ),
                            },
                        }, None

                # Process all documents in parallel
                parallel_start = time.time()
                print(f"🚀 PARALLEL PROCESSING START: {parallel_start:.3f}")
                print(f"🚀 Processing {len(media_urls)} documents in parallel...")
                document_tasks = [
                    process_single_document(i, media_url)
                    for i, media_url in enumerate(media_urls)
                ]

                # Wait for all documents to complete
                print(f"⏳ Waiting for all documents to complete...")
                document_results = await asyncio.gather(
                    *document_tasks, return_exceptions=True
                )

                parallel_end = time.time()
                print(
                    f"✅ PARALLEL PROCESSING COMPLETE: {parallel_end:.3f} (took {parallel_end - parallel_start:.3f}s)"
                )

                # Process results
                for i, result in enumerate(document_results):
                    if isinstance(result, Exception):
                        print(f"❌ Document {i+1} failed with exception: {result}")
                        analysis_results.append(
                            {
                                "type": "error",
                                "language": "unknown",
                                "title": f"Document {i+1}",
                                "summary_title": "Processing failed",
                                "token": 0,
                                "_result": {"error": f"Exception: {str(result)}"},
                            }
                        )
                    else:
                        analysis_result, document_analysis = result
                        analysis_results.append(analysis_result)
                        if document_analysis:
                            processed_documents.append(document_analysis)
                        total_tokens += attachment_tokens

                doc_processing_end = time.time()
                print(
                    f"✅ DOCUMENT PROCESSING COMPLETE: {doc_processing_end:.3f} (took {doc_processing_end - doc_processing_start:.3f}s)"
                )

                if analysis_results:

                    print(f"📊 Total tokens used: {total_tokens}")

                    # Create conversation entry for document analysis
                    analysis_content = {
                        "type": "document_analysis",
                        "_result": {
                            "note": f"Processed {len(analysis_results)} document(s) with enhanced analysis",
                            "results": analysis_results,
                            "total_tokens": total_tokens,
                            "enhanced_processing": True,
                            "processed_documents": [
                                {
                                    "document_type": doc.document_type,
                                    "total_questions": doc.total_questions,
                                    "main_topics": doc.main_topics,
                                    "difficulty_level": doc.difficulty_level,
                                    "subject_area": doc.subject_area,
                                    "chunk_count": len(doc.chunks),
                                }
                                for doc in processed_documents
                            ],
                        },
                    }

                    # Create AI conversation entry
                    ai_conv = Conversation(
                        interaction_id=str(interaction.id),
                        role=ConversationRole.AI,
                        content=analysis_content,
                        input_tokens=total_tokens,
                        output_tokens=0,
                        tokens_used=total_tokens,
                        points_cost=langchain_service.calculate_points_cost(
                            total_tokens
                        ),
                        status="completed",
                    )

                    db.add(ai_conv)
                    await db.flush()

                    return {
                        "success": True,
                        "message": "Enhanced document processing completed",
                        "conversation_id": str(ai_conv.id),
                        "analysis_results": analysis_results,
                        "total_tokens": total_tokens,
                        "points_cost": ai_conv.points_cost,
                        "enhanced_processing": True,
                        "processed_documents": len(processed_documents),
                    }
                else:
                    return {
                        "success": False,
                        "message": "Document processing failed for all uploaded files",
                    }

            # === PHASE 3: DETECT SPECIAL REQUEST TYPES ===
            # Check if this is a mindmap request
            is_mindmap_request = detect_mindmap_request(message or "")

            print(f"🔍 Request type detection:")
            print(f"   Mindmap request: {is_mindmap_request}")

            # Handle mindmap generation requests
            if is_mindmap_request:
                print(f"🗺️ MINDMAP GENERATOR MODE ACTIVATED")
                try:
                    # Extract topic from message
                    topic = extract_topic_from_message(message or "")
                    print(f"   Topic: {topic}")

                    # Generate mindmap WITH CONTEXT
                    mindmap = await langchain_service.generate_mindmap(
                        topic=topic,
                        context=context_text,  # NEW: Pass context
                        max_tokens=max_tokens,
                        assistant_model=assistant_model,
                        subscription_plan=subscription_plan,
                    )

                    # Format as conversation content
                    ai_content = {
                        "type": "mindmap",
                        "_result": {
                            "topic": mindmap.topic,
                            "nodes": serialize_mindmap_tree(mindmap.root_node),
                            "total_nodes": mindmap.total_nodes,
                        },
                    }

                    # Create AI conversation entry
                    ai_conv = Conversation(
                        interaction_id=str(interaction.id),
                        role=ConversationRole.AI,
                        content=ai_content,
                        question_type="mindmap",
                        input_tokens=0,
                        output_tokens=0,
                        tokens_used=0,
                        points_cost=0,
                        status="completed",
                    )

                    db.add(ai_conv)
                    await db.flush()

                    print(f"✅ Mindmap generated successfully")
                    print(f"   Topic: {mindmap.topic}")
                    print(f"   Total nodes: {mindmap.total_nodes}")

                    # NEW: Background title generation
                    asyncio.create_task(
                        _extract_interaction_metadata_fast(
                            interaction=interaction,
                            content_text=f"Mindmap: {topic}",
                            original_message=message or "",
                            assistant_model=assistant_model,
                            subscription_plan=subscription_plan,
                        )
                    )

                    # NEW: Background vector DB storage
                    def format_mindmap_for_embedding(node, depth=0):
                        indent = "  " * depth
                        text = f"{indent}- {node.content}\n"
                        for child in node.children:
                            text += format_mindmap_for_embedding(child, depth + 1)
                        return text

                    formatted_text = f"Mindmap: {topic}\n\n{format_mindmap_for_embedding(mindmap.root_node)}"

                    asyncio.create_task(
                        langchain_service.upsert_embedding(
                            conv_id=str(ai_conv.id),
                            user_id=str(user_id),
                            text=formatted_text,
                            title=f"Mindmap: {topic}",
                            metadata={
                                "interaction_id": str(interaction.id),
                                "type": "mindmap",
                                "total_nodes": mindmap.total_nodes,
                            },
                        )
                    )

                    return {
                        "success": True,
                        "message": "Mindmap generated",
                        "conversation_id": str(ai_conv.id),
                        "solution_type": "mindmap",
                    }

                except Exception as e:
                    print(f"❌ Mindmap generator error: {e}")
                    # Fall through to regular conversation generation
                    is_mindmap_request = False

            # Generate conversation response using LangChain
            ai_generation_start = time.time()
            print(f"🤖 AI GENERATION START: {ai_generation_start:.3f}")

            # Create a task for AI generation that can be cancelled
            task_key = f"{user_id}_{interaction.id}"

            async def generate_ai_response():
                ai_start = time.time()
                print(f"🤖 AI SERVICE CALL START: {ai_start:.3f}")

                # Note: interaction_summary no longer uses semantic_summary field
                # Vector search context now provides semantic context
                result = await langchain_service.generate_conversation_response(
                    message=message or "",
                    context=context_text,
                    media_urls=media_urls,
                    interaction_title=interaction.title if interaction else None,
                    interaction_summary=None,  # Removed: semantic_summary field no longer exists
                    max_tokens=max_tokens,
                    visualize_model=visualize_model,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
                )

                ai_end = time.time()
                print(
                    f"🤖 AI SERVICE CALL COMPLETE: {ai_end:.3f} (took {ai_end - ai_start:.3f}s)"
                )
                return result

            # Create and track the task
            generation_task = asyncio.create_task(generate_ai_response())
            active_generation_tasks[task_key] = generation_task

            try:
                print(f"⏳ Waiting for AI response...")
                ai_response, input_tokens, output_tokens, total_tokens = (
                    await generation_task
                )

                ai_generation_end = time.time()
                print(
                    f"🤖 AI GENERATION COMPLETE: {ai_generation_end:.3f} (took {ai_generation_end - ai_generation_start:.3f}s)"
                )
                print("=" * 200)
                print(f"🔄 AI RESPONSE TYPE: {type(ai_response)}")
                print(
                    f"🔄 AI RESPONSE LENGTH: {len(str(ai_response)) if ai_response else 0}"
                )
                print(f"🔄 AI RESPONSE: {ai_response}")
                print("=" * 200)

                # Check if AI response is empty or None
                if not ai_response or (
                    isinstance(ai_response, str) and ai_response.strip() == ""
                ):
                    print(f"❌ ERROR: AI response is empty or None!")
                    print(f"   Input tokens: {input_tokens}")
                    print(f"   Output tokens: {output_tokens}")
                    print(f"   Total tokens: {total_tokens}")
                    return {
                        "success": False,
                        "message": "AI response was empty",
                        "interaction_id": str(interaction.id),
                        "ai_response": "Sorry, I couldn't generate a response. Please try again.",
                    }

            except asyncio.CancelledError:

                # Clean up the task from tracking
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]
                return {
                    "success": False,
                    "message": "AI generation was cancelled",
                    "interaction_id": str(interaction.id),
                }
            except Exception as e:

                # Clean up the task from tracking
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]
                return {
                    "success": False,
                    "message": f"Failed to generate response: {str(e)}",
                }
            finally:
                # Clean up the task from tracking when done
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]

            # Create user conversation entry (store media IDs instead of URLs)
            user_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.USER,
                content={
                    "type": "text",
                    "_result": {
                        "note": message or "",
                        "media_ids": (
                            [media.id for media in media_objects]
                            if media_objects
                            else []
                        ),
                    },
                },
                status="completed",
            )

            db.add(user_conv)
            await db.flush()

            # Send SSE notification that user message was received
            try:
                from app.api.sse_routes import notify_message_received_sse

                await notify_message_received_sse(
                    user_id=str(user_id),
                    interaction_id=str(interaction.id),
                    conversation_id=str(user_conv.id),
                )

            except Exception as e:
                print(f"⚠️  Failed to send message received notification: {e}")

            # Process AI content with enhanced formatting - with robust error handling
            print(
                f"🔧 CALLING _process_ai_content_fast with ai_response: {ai_response}"
            )

            # Initialize fallback values
            ai_content_type = "written"
            processed_ai_response = (
                str(ai_response) if ai_response else "No response generated"
            )

            try:
                ai_content_type, processed_ai_response = _process_ai_content_fast(
                    ai_response
                )
                print(
                    f"🔧 _process_ai_content_fast returned: type={ai_content_type}, content={processed_ai_response}"
                )
            except Exception as processing_error:
                print(f"⚠️ Error processing AI content: {processing_error}")
                print(f"🔧 Using fallback processing for AI response")
                # Use fallback processing - store raw response
                if isinstance(ai_response, dict):
                    ai_content_type = "structured"
                    processed_ai_response = str(ai_response)
                else:
                    ai_content_type = "written"
                    processed_ai_response = (
                        str(ai_response) if ai_response else "No response generated"
                    )
                print(
                    f"🔧 Fallback result: type={ai_content_type}, content={processed_ai_response}"
                )

            # Ensure we have valid content to store
            if not processed_ai_response or (
                isinstance(processed_ai_response, str)
                and processed_ai_response.strip() == ""
            ):
                processed_ai_response = (
                    "AI response was generated but could not be processed"
                )
                ai_content_type = "error"
                print(f"🔧 No valid content, using error fallback")

            # Create AI conversation entry with enhanced content
            # Handle MCQ responses differently - store structured data directly
            print(
                f"🔧 Database storage check: ai_content_type={ai_content_type}, processed_ai_response_type={type(processed_ai_response)}"
            )
            if ai_content_type == "mcq" and isinstance(processed_ai_response, dict):
                print(f"🔧 Storing MCQ data directly in database")
                ai_conv = Conversation(
                    interaction_id=str(interaction.id),
                    role=ConversationRole.AI,
                    content=processed_ai_response,  # Store MCQ data directly
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_used=total_tokens,
                    points_cost=langchain_service.calculate_points_cost(total_tokens),
                    status="completed",
                )
            else:
                print(f"🔧 Using old structure for content type: {ai_content_type}")
                # For other content types, use the existing structure
                ai_conv = Conversation(
                    interaction_id=str(interaction.id),
                    role=ConversationRole.AI,
                    content={
                        "type": ai_content_type,
                        "_result": {"content": processed_ai_response},
                    },
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_used=total_tokens,
                    points_cost=langchain_service.calculate_points_cost(total_tokens),
                    status="completed",
                )

            # CRITICAL: Always store the AI response in database, regardless of other errors
            try:
                db.add(ai_conv)
                await db.flush()
                print(f"✅ AI conversation entry added to database successfully")
            except Exception as db_error:
                print(
                    f"❌ CRITICAL: Failed to add AI conversation to database: {db_error}"
                )
                # This is a critical error - we must not continue without storing the response
                raise Exception(f"Failed to store AI response in database: {db_error}")

            # CRITICAL: Always commit the AI response FIRST (non-blocking title generation)
            try:
                await db.commit()
                print(f"✅ AI response committed to database successfully")
            except Exception as commit_error:
                print(
                    f"❌ CRITICAL: Failed to commit AI response to database: {commit_error}"
                )
                # This is a critical error - we must not continue without committing the response
                raise Exception(
                    f"Failed to commit AI response to database: {commit_error}"
                )

            # OPTIMIZATION: Update interaction title in background (non-blocking)
            asyncio.create_task(
                _update_title_background_task(
                    interaction_id=str(interaction.id),
                    content_text=ai_response,
                    original_message=message or "",
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
                )
            )

            # Refresh the interaction to verify the changes were saved
            try:
                await db.refresh(interaction)
                print(f"✅ Interaction refreshed successfully")
            except Exception as refresh_error:
                print(f"⚠️ Failed to refresh interaction: {refresh_error}")
                # This is not critical - the AI response is already committed

            # Start simplified background operations (only embeddings, no complex queues)
            from app.services.simplified_background_service import (
                run_simplified_background_operations,
            )

            asyncio.create_task(
                run_simplified_background_operations(
                    user_conv_id=str(user_conv.id),
                    ai_conv_id=str(ai_conv.id),
                    user_id=str(user_id),
                    interaction_id=str(interaction.id),
                    message=message or "",
                    ai_content=ai_response,
                )
            )

            # Send AI response notification using enhanced function
            try:
                await _send_ai_response_notification(
                    user_id=str(user_id),
                    interaction_id=str(interaction.id),
                    ai_response=ai_response,
                )
                print(f"✅ Sent AI response notification")
            except Exception as e:
                print(f"⚠️  Failed to send AI response notification: {e}")

            result = {
                "success": True,
                "message": "Response generated successfully",
                "conversation_id": str(ai_conv.id),
                "ai_response": processed_ai_response,
                "ai_content_type": ai_content_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "points_cost": ai_conv.points_cost,
            }
            print(f"🔧 SERVICE RETURNING RESULT: {result}")
            return result

    except Exception as e:
        print(f"❌ Error in process_conversation_message: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "message": f"An error occurred: {str(e)}"}
    finally:
        # Close the database session if we created it
        if should_close_db and db:
            await db.close()


async def generate_mcq_questions(
    *,
    user: User,
    interaction: Optional[Interaction],
    topic_or_content: str,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """
    Generate MCQ questions for a given topic or content
    """
    try:
        # Use a single DB session for the entire operation
        async with AsyncSessionLocal() as db:
            # Get or create interaction
            if not interaction:
                interaction = Interaction(
                    user_id=user.id,
                    title="MCQ Generation",
                    summary="Generated multiple choice questions",
                )
                db.add(interaction)
                await db.flush()

            # Generate MCQ questions using LangChain
            mcq_result = await langchain_service.generate_mcq_questions(
                topic_or_content=topic_or_content, max_tokens=max_tokens
            )

            if mcq_result.get("type") == "error":
                return {
                    "success": False,
                    "message": "Failed to generate MCQs",
                    "error": mcq_result.get("_result", {}).get(
                        "error", "Unknown error"
                    ),
                }

            # Create conversation entries
            user_conv = Conversation(
                interaction_id=interaction.id,
                role=ConversationRole.USER,
                content=f"Generate MCQs for: {topic_or_content}",
                points_cost=0,
            )
            db.add(user_conv)

            # Store MCQ data in structured format for proper frontend rendering
            questions = mcq_result.get("_result", {}).get("questions", [])

            # Create structured MCQ content for frontend
            structured_mcq_content = {
                "type": "mcq",
                "language": mcq_result.get("language", "english"),
                "_result": {"questions": questions},
            }

            ai_conv = Conversation(
                interaction_id=interaction.id,
                role=ConversationRole.AI,
                content=structured_mcq_content,  # Store structured data directly
                points_cost=StudyGuruConfig.calculate_points_cost(
                    mcq_result.get("token", 0)
                ),
            )
            db.add(ai_conv)

            # Update interaction title and summary
            interaction.title = mcq_result.get("title", "MCQ Generation")
            interaction.summary = mcq_result.get(
                "summary_title", "Generated multiple choice questions"
            )

            # Commit all changes
            await db.commit()

            return {
                "success": True,
                "message": "MCQ questions generated successfully",
                "conversation_id": str(ai_conv.id),
                "ai_response": structured_mcq_content,
                "mcq_data": mcq_result,
                "total_questions": len(questions),
                "points_cost": ai_conv.points_cost,
            }

    except Exception as e:
        print(f"❌ Error in generate_mcq_questions: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "message": f"An error occurred: {str(e)}"}


async def _background_operations_enhanced(
    user_conv_id: int,
    ai_conv_id: int,
    user_id: str,
    interaction_id: str,
    message: str,
    ai_content: str,
):
    """Enhanced background operations using real-time context service"""
    import json  # Import json at the top of the function to avoid scope issues

    try:
        print(
            f"🔄 Starting enhanced background operations for interaction {interaction_id}"
        )

        # Import the real-time context service
        from app.services.real_time_context_service import real_time_context_service

        # === STEP 1: Queue semantic summary update ===
        semantic_task_id = await real_time_context_service.queue_context_update(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=str(ai_conv_id),
            update_type="semantic_summary",
            payload={"user_message": message, "ai_response": ai_content},
            priority=1,  # High priority
        )
        print(f"   📝 Queued semantic summary update: {semantic_task_id}")

        # === STEP 2: Queue embedding updates ===
        embedding_tasks = []

        # Prepare enhanced metadata
        enhanced_metadata = {
            "interaction_id": interaction_id,
            "conversation_id": str(ai_conv_id),
            "update_timestamp": datetime.now().isoformat(),
            "context_version": "2.0",
        }

        # Queue user message embedding
        if message:
            user_embedding_task_id = (
                await real_time_context_service.queue_context_update(
                    user_id=user_id,
                    interaction_id=interaction_id,
                    conversation_id=str(user_conv_id),
                    update_type="embedding",
                    payload={
                        "conversation_id": str(user_conv_id),
                        "text": message,
                        "title": f"User message in {interaction_id}",
                        "metadata": enhanced_metadata,
                    },
                    priority=2,
                )
            )
            embedding_tasks.append(user_embedding_task_id)
            print(f"   🔗 Queued user message embedding: {user_embedding_task_id}")

        # Queue AI response embedding
        if ai_content:
            # Ensure ai_content is a string for embedding
            ai_content_text = ai_content
            if isinstance(ai_content, dict):
                ai_content_text = json.dumps(ai_content, indent=2)
            elif not isinstance(ai_content, str):
                ai_content_text = str(ai_content)

            ai_embedding_task_id = await real_time_context_service.queue_context_update(
                user_id=user_id,
                interaction_id=interaction_id,
                conversation_id=str(ai_conv_id),
                update_type="embedding",
                payload={
                    "conversation_id": str(ai_conv_id),
                    "text": ai_content_text,
                    "title": f"AI response in {interaction_id}",
                    "metadata": enhanced_metadata,
                },
                priority=2,
            )
            embedding_tasks.append(ai_embedding_task_id)
            print(f"   🔗 Queued AI response embedding: {ai_embedding_task_id}")

        # === STEP 3: Queue conversation context update ===
        context_task_id = await real_time_context_service.queue_context_update(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=str(ai_conv_id),
            update_type="conversation_context",
            payload={
                "context_data": {
                    "user_message": message,
                    "ai_response": ai_content,
                    "relevance_score": 0.9,
                    "importance_score": 0.8,
                    "context_type": "conversation_pair",
                }
            },
            priority=3,
        )
        print(f"   💾 Queued conversation context update: {context_task_id}")

        # === STEP 4: Monitor completion (optional) ===
        # Wait a bit for critical updates to complete
        await asyncio.sleep(2)

        # Check status of critical updates
        semantic_status = await real_time_context_service.get_update_status(
            semantic_task_id
        )
        if semantic_status:
            print(
                f"   📊 Semantic summary status: {semantic_status.get('status', 'unknown')}"
            )

        print(
            f"✅ Enhanced background operations queued successfully for interaction {interaction_id}"
        )
        print(
            f"   Total tasks queued: {1 + len(embedding_tasks) + 1}"
        )  # semantic + embeddings + context

    except Exception as e:
        print(
            f"❌ Enhanced background operations error for interaction {interaction_id}: {e}"
        )
        import traceback

        traceback.print_exc()

        # Fallback to original method if enhanced fails
        print(f"🔄 Falling back to original background operations...")
        await _background_operations(
            user_conv_id, ai_conv_id, user_id, interaction_id, message, ai_content
        )


async def _background_operations(
    user_conv_id: int,
    ai_conv_id: int,
    user_id: str,
    interaction_id: str,
    message: str,
    ai_content: str,
):
    """Original background operations for conversation processing (fallback)"""
    import json  # Import json at the top of the function to avoid scope issues

    try:
        print(
            f"🔄 Starting fallback background operations for interaction {interaction_id}"
        )

        # === STEP 1: Create conversation-level summary ===
        try:
            print(f"   📝 Creating conversation summary...")
            conversation_summary = await langchain_service.summarize_conversation(
                user_message=message, ai_response=ai_content
            )
            print(f"   ✅ Conversation summary created")
            print(f"      Topics: {conversation_summary.get('main_topics', [])}")
            print(f"      Facts: {len(conversation_summary.get('key_facts', []))}")
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
            enhanced_metadata["facts"] = conversation_summary.get("key_facts", [])

        # Create embeddings for user message
        if message:
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(user_conv_id),
                    user_id=user_id,
                    text=message,
                    title=f"User message in {interaction_id}",
                    metadata=enhanced_metadata,
                )
            )

        # Create embeddings for AI response
        if ai_content:
            # Ensure ai_content is a string for embedding
            ai_content_text = ai_content
            if isinstance(ai_content, dict):
                ai_content_text = json.dumps(ai_content, indent=2)
            elif not isinstance(ai_content, str):
                ai_content_text = str(ai_content)

            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(ai_conv_id),
                    user_id=user_id,
                    text=ai_content_text,
                    title=f"AI response in {interaction_id}",
                    metadata=enhanced_metadata,
                )
            )

        # Execute embedding tasks in parallel
        if embedding_tasks:
            print(f"   🔗 Creating {len(embedding_tasks)} embeddings...")
            embedding_results = await asyncio.gather(
                *embedding_tasks, return_exceptions=True
            )

            successful_embeddings = 0
            for i, result in enumerate(embedding_results):
                if isinstance(result, Exception):
                    print(f"   ⚠️  Embedding {i+1} failed: {result}")
                elif result:
                    successful_embeddings += 1

            print(
                f"   ✅ Created {successful_embeddings}/{len(embedding_tasks)} embeddings successfully"
            )

        print(
            f"✅ Fallback background operations completed successfully for interaction {interaction_id}"
        )

    except Exception as e:
        print(
            f"❌ Fallback background operations error for interaction {interaction_id}: {e}"
        )
        import traceback

        traceback.print_exc()
        pass


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

    try:
        # Check if the interaction exists and belongs to the user
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == interaction_id, Interaction.user_id == user_id
            )
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return {
                "success": False,
                "message": "Interaction not found or access denied",
            }

        # Check if there's an active task for this interaction
        task_key = f"{user_id}_{interaction_id}"
        if task_key in active_generation_tasks:
            task = active_generation_tasks[task_key]
            if not task.done():
                task.cancel()
                print(f"✅ Cancelled active AI generation task for {task_key}")
            del active_generation_tasks[task_key]

        # Find the most recent processing conversation for this interaction
        conv_result = await db.execute(
            select(Conversation)
            .where(
                Conversation.interaction_id == interaction_id,
                Conversation.role == ConversationRole.AI,
                Conversation.status == "processing",
            )
            .order_by(Conversation.created_at.desc())
            .limit(1)
        )
        processing_conv = conv_result.scalar_one_or_none()

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
                from app.api.sse_routes import notify_ai_response_ready_sse

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

    except Exception as e:
        print(f"❌ Error cancelling AI generation: {e}")
        await db.rollback()
        return {"success": False, "message": f"Failed to cancel generation: {str(e)}"}


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
        from app.api.websocket_routes import notify_message_received

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
            from app.api.sse_routes import notify_message_received_sse

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
    # Handle both string and dict responses
    if isinstance(ai_response, dict):
        response_preview = str(ai_response)[:100] + "..."
        response_length = len(str(ai_response))
    else:
        response_preview = (
            ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
        )
        response_length = len(ai_response)

    print(f"📝 Response Length: {response_length} characters")
    print(f"📄 Response Preview: {response_preview}")
    print(f"{'='*60}\n")

    # Try WebSocket first
    try:
        from app.api.websocket_routes import notify_ai_response_ready

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
            from app.api.sse_routes import notify_ai_response_ready_sse

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
    interaction: Interaction,
    content_text,
    original_message: str = "",
    assistant_model: Optional[str] = None,
    subscription_plan: Optional[str] = None,
):
    """Fast metadata extraction with dedicated title generation using langchain_service"""
    try:
        # OPTIMIZATION: Skip title generation if we already have a good title
        # Check if we need to update the title
        needs_title_update = (
            not interaction.title
            or not interaction.summary_title
            or interaction.title
            in ["Study Session", "New Interaction", "MCQ Generation"]
            or len(interaction.title) < 5
        )

        # SKIP if we already have a good title (saves 200-500ms)
        if not needs_title_update:
            print(
                f"✅ Skipping title generation - already have good title: '{interaction.title}'"
            )
            return

        print(
            f"🔍 Title update needed. Current title: '{interaction.title}', summary_title: '{interaction.summary_title}'"
        )

        # Handle different content types for response preview
        if isinstance(content_text, dict):
            # For dictionary responses (like MCQ), extract meaningful content
            if content_text.get("type") == "mcq" and "_result" in content_text:
                questions = content_text.get("_result", {}).get("questions", [])
                if questions:
                    # Extract first question for preview
                    first_question = questions[0]
                    question_text = first_question.get("question", "")
                    response_preview = f"{question_text[:200]}"
                else:
                    response_preview = "MCQ Generation"
            else:
                response_preview = str(content_text)[
                    :500
                ]  # Increased for better context
        elif isinstance(content_text, str):
            # For string responses, check if it's MCQ JSON
            if (
                content_text.strip().startswith("```json")
                and "mcq" in content_text.lower()
            ):
                # Extract first question from MCQ JSON
                try:
                    import json
                    import re

                    # Extract JSON from markdown
                    json_match = re.search(
                        r"```json\s*(\{.*?\})\s*```", content_text, re.DOTALL
                    )
                    if json_match:
                        json_str = json_match.group(1)
                        data = json.loads(json_str)
                        if data.get("type") == "mcq" and "_result" in data:
                            questions = data.get("_result", {}).get("questions", [])
                            if questions:
                                first_question = questions[0]
                                question_text = first_question.get("question", "")
                                response_preview = f" {question_text[:200]}"
                            else:
                                response_preview = "MCQ Generation"
                        else:
                            response_preview = content_text[
                                :2000
                            ]  # Increased for better context
                    else:
                        response_preview = content_text[
                            :2000
                        ]  # Increased for better context
                except:
                    response_preview = content_text[
                        :2000
                    ]  # Increased for better context
            else:
                response_preview = content_text[:2000]  # Increased for better context
        else:
            response_preview = str(content_text)[:2000]  # Increased for better context

        print(
            f"🔍 Calling title generation with message: '{original_message}', response_preview: '{response_preview}'"
        )

        # Check if this is a simple greeting or casual message
        from app.constants import SIMPLE_GREETINGS

        is_simple_greeting = (
            original_message and original_message.lower().strip() in SIMPLE_GREETINGS
        )

        # Check if this is an MCQ response
        is_mcq_response = response_preview.startswith("MCQ:")

        if is_simple_greeting:
            # For simple greetings, use a generic but appropriate title
            title = "Chat with StudyGuru"
            summary_title = "General conversation and assistance"

        elif is_mcq_response:
            # For MCQ responses, generate a more specific title
            try:
                # Extract topic from the question for better title
                question_text = response_preview.replace("MCQ:", "").strip()
                if "math" in question_text.lower() or any(
                    op in question_text for op in ["+", "-", "×", "÷", "="]
                ):
                    title = "Math Practice Questions"
                    summary_title = "Mathematics MCQ exercises"
                elif "science" in question_text.lower() or any(
                    word in question_text.lower()
                    for word in ["biology", "chemistry", "physics"]
                ):
                    title = "Science Practice Questions"
                    summary_title = "Science MCQ exercises"
                else:
                    title = "Practice Questions"
                    summary_title = "Educational MCQ exercises"
            except:
                title = "Practice Questions"
                summary_title = "Educational MCQ exercises"

        else:
            # Use AI generation for more substantive messages
            try:
                title, summary_title = (
                    await langchain_service.generate_interaction_title(
                        message=original_message,
                        response_preview=response_preview,
                        assistant_model=assistant_model,
                        subscription_plan=subscription_plan,
                    )
                )

            except Exception as ai_title_error:
                print(f"⚠️ AI title generation failed: {ai_title_error}")
                # Fallback to intelligent title generation
                if original_message:
                    # Check if the message is a URL
                    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                    is_url = bool(re.match(url_pattern, original_message.strip()))

                    if is_url:
                        # For URLs, try to extract meaningful title from response or domain
                        try:
                            from urllib.parse import urlparse

                            parsed_url = urlparse(original_message.strip())
                            domain = parsed_url.netloc.replace("www.", "").split(".")[0]

                            # Try to extract title from response_preview if available
                            if response_preview and len(response_preview) > 20:
                                # Look for common title patterns in the response
                                # Try to find the first sentence or meaningful phrase
                                first_sentence = response_preview.split(".")[0].strip()
                                if (
                                    len(first_sentence) > 10
                                    and len(first_sentence) < 60
                                ):
                                    title = first_sentence
                                    summary_title = f"Content from {domain.title()}"
                                else:
                                    # Use domain-based title
                                    title = f"Research from {domain.title()}"
                                    summary_title = f"Content from {domain.title()}"
                            else:
                                # Use domain-based title
                                title = f"Research from {domain.title()}"
                                summary_title = f"Content from {domain.title()}"
                        except Exception as url_error:
                            print(f"⚠️ URL parsing error: {url_error}")
                            title = "Research Article"
                            summary_title = "Educational content"
                    else:
                        # For regular messages, use first 40 characters
                        title = original_message[:40].strip()
                        summary_title = f"Help with {title.lower()}"
                else:
                    title = "Study Session"
                    summary_title = "Educational assistance"

        # Update interaction with generated titles
        if title:
            interaction.title = title
            print(f"✅ Set interaction title: '{title}'")
        else:
            print(f"⚠️ No title generated")

        if summary_title:
            interaction.summary_title = summary_title
            print(f"✅ Set interaction summary_title: '{summary_title}'")
        else:
            print(f"⚠️ No summary title generated")

    except Exception as e:
        print(f"Metadata extraction error: {e}")
        # Final fallback: create basic title from message
        if original_message and not interaction.title:
            interaction.title = original_message[:40].strip()


def _format_mcq_response(mcq_data: dict) -> str:
    """Format MCQ response into readable text for frontend display"""
    try:
        print(f"🔧 _format_mcq_response called with: {mcq_data}")
        title = mcq_data.get("title", "MCQ Questions")
        questions = mcq_data.get("_result", {}).get("questions", [])
        print(f"🔧 Extracted title: {title}")
        print(f"🔧 Extracted questions: {questions}")

        if not questions:
            result = f"**{title}**\n\nNo questions found in the document."
            print(f"🔧 No questions found, returning: {result}")
            return result

        formatted_response = f"**{title}**\n\n"

        for i, question_data in enumerate(questions, 1):
            question_text = question_data.get("question", "")
            options = question_data.get("options", {})
            answer = question_data.get("answer", "")
            explanation = question_data.get("explanation", "")

            formatted_response += f"**{i}. {question_text}**\n"

            if options:
                for option_key, option_value in options.items():
                    formatted_response += f"   {option_key.upper()}. {option_value}\n"

            if answer:
                formatted_response += f"\n**Answer: {answer.upper()}**\n"

            if explanation:
                formatted_response += f"**Explanation:** {explanation}\n"

            formatted_response += "\n---\n\n"

        result = formatted_response.strip()

        # Safety check: ensure we never return empty content
        if not result or result.strip() == "":
            result = f"**MCQ Questions**\n\nQuestions were generated but could not be formatted properly."

        print(f"🔧 _format_mcq_response returning: {result}")
        return result

    except Exception as e:
        print(f"❌ Error formatting MCQ response: {e}")
        result = f"**MCQ Questions**\n\nError formatting questions: {str(e)}"
        print(f"🔧 Error case returning: {result}")
        return result


# Background task for non-blocking title generation
async def _update_title_background_task(
    interaction_id: str,
    content_text: str,
    original_message: str,
    assistant_model: Optional[str] = None,
    subscription_plan: Optional[str] = None,
):
    """Update interaction title in background without blocking response"""
    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Interaction).where(Interaction.id == interaction_id)
            )
            interaction = result.scalar_one_or_none()
            if interaction:
                await _extract_interaction_metadata_fast(
                    interaction=interaction,
                    content_text=content_text,
                    original_message=original_message,
                    assistant_model=assistant_model,
                    subscription_plan=subscription_plan,
                )
                await db.commit()
                print(f"✅ Background title generation completed")
    except Exception as title_error:
        print(f"⚠️ Background title generation failed: {title_error}")


def _process_ai_content_fast(content_text) -> tuple[str, str]:
    """Fast AI content processing with enhanced formatting and escaped character handling"""
    import json  # Import json at the top of the function to avoid scope issues

    print(
        f"🔧 PROCESS AI CONTENT START: type={type(content_text)}, length={len(str(content_text)) if content_text else 0}"
    )

    # Initialize with safe defaults
    ai_content_type = "written"
    ai_result_content = str(content_text) if content_text else "No content provided"

    # Ensure we always return valid content
    if not content_text:
        print(f"🔧 No content provided, returning default")
        return ai_content_type, ai_result_content

    # Handle structured responses (dictionaries) from document analysis
    if isinstance(content_text, dict):
        print(f"🔧 PROCESSING DICT: {content_text}")
        # This is a structured response from document analysis
        ai_content_type = (
            "mcq" if content_text.get("type") == "mcq" else "document_analysis"
        )

        # For MCQ responses, keep the structured data intact
        if content_text.get("type") == "mcq":
            print(f"🔧 MCQ response detected, keeping structured data")
            # Return the MCQ data directly without formatting
            return ai_content_type, content_text
        else:
            # For other structured responses, convert to JSON
            ai_result_content = json.dumps(content_text, indent=2)

        print(
            f"🔧 DICT RESULT: type={ai_content_type}, length={len(ai_result_content) if ai_result_content else 0}"
        )
        print(f"🔧 FINAL RESULT CONTENT: {ai_result_content}")
        return ai_content_type, ai_result_content

    # Handle string responses (regular conversations)
    if content_text and isinstance(content_text, str):
        print(f"🔧 PROCESSING STRING: {content_text[:100]}...")
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

    # Detect MCQ patterns in plain text content (only for strings)
    if (
        content_text
        and isinstance(content_text, str)
        and not content_text.strip().startswith("{")
    ):
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
        # First, try to extract JSON from markdown code blocks
        json_content = content_text
        if "```json" in content_text:
            # Extract JSON from markdown code blocks
            import re

            json_match = re.search(r"```json\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                print(f"🔧 Extracted JSON from markdown: {json_content[:100]}...")

        # Try to parse as JSON
        if json_content and (
            json_content.strip().startswith("{") or "```json" in content_text
        ):
            parsed_response = json.loads(json_content)
            if isinstance(parsed_response, dict):
                # Check if this is MCQ data first
                if parsed_response.get("type") == "mcq":
                    ai_content_type = "mcq"
                    print(
                        f"🔧 MCQ JSON detected in string content, returning structured data"
                    )
                    # Return the structured MCQ data directly
                    return ai_content_type, parsed_response

                # For other types, use the existing logic
                ai_content_type = parsed_response.get("type", "written")
                result_content = parsed_response.get("_result", {})

                if isinstance(result_content, dict):
                    if "content" in result_content:
                        ai_result_content = result_content["content"]
                    elif "questions" in result_content:
                        # For MCQ data, keep it structured instead of formatting as string
                        questions = result_content["questions"]
                        if questions and isinstance(questions, list):
                            # Return structured MCQ data
                            ai_content_type = "mcq"
                            ai_result_content = {
                                "type": "mcq",
                                "language": "english",
                                "_result": {"questions": questions},
                            }
                            return ai_content_type, ai_result_content

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

            # Normalize markdown format for consistency across different LLMs
            ai_result_content = normalize_markdown_format(ai_result_content)

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # FINAL SAFETY CHECK: Ensure we always return valid content
    if not ai_result_content or (
        isinstance(ai_result_content, str) and ai_result_content.strip() == ""
    ):
        print(f"🔧 FINAL SAFETY: No valid content, using fallback")
        ai_content_type = "error"
        ai_result_content = (
            str(content_text) if content_text else "AI response could not be processed"
        )

    print(
        f"🔧 PROCESS AI CONTENT END: type={ai_content_type}, length={len(str(ai_result_content)) if ai_result_content else 0}"
    )
    return ai_content_type, ai_result_content
