"""
Enhanced Document Integration Service

This service provides comprehensive document processing, indexing, and retrieval capabilities:
- Multi-format document processing (PDF, images, text)
- Structured content extraction and indexing
- Document versioning and update tracking
- Advanced document search and retrieval
- Document relationship mapping
- Content chunking and granular indexing
"""

import asyncio
import json
import re
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from app.core.database import AsyncSessionLocal
from app.models.context import DocumentContext
from app.models.interaction import Interaction, Media
from app.models.user import User
from app.services.langchain_service import langchain_service
from app.services.vector_optimization_service import (
    vector_optimization_service,
    SearchQuery,
)


@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""

    chunk_id: str
    content: str
    chunk_type: str  # 'question', 'answer', 'explanation', 'header', 'paragraph'
    question_number: Optional[str] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = None
    start_position: int = 0
    end_position: int = 0


@dataclass
class DocumentAnalysis:
    """Comprehensive document analysis result"""

    document_id: str
    document_type: str
    total_questions: int
    question_structure: Dict[str, Any]
    main_topics: List[str]
    difficulty_level: str
    subject_area: str
    question_mapping: Dict[str, str]
    answer_key: Dict[str, str]
    full_content: str
    content_summary: str
    key_concepts: List[str]
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]


class DocumentIntegrationService:
    """Enhanced document processing and integration service"""

    def __init__(self):
        self.chunk_size = 1000  # Characters per chunk
        self.overlap_size = 200  # Overlap between chunks
        self.max_chunks_per_document = 50

    async def process_document_comprehensive(
        self,
        media_id: str,
        interaction_id: str,
        user_id: str,
        file_url: str,
        max_tokens: int = 2000,
    ) -> DocumentAnalysis:
        """
        Comprehensive document processing with multi-level analysis
        """
        try:
            print(f"📄 Starting comprehensive document processing for media {media_id}")

            # Step 1: Basic document analysis using LangChain
            basic_analysis = await langchain_service.analyze_document(
                file_url=file_url, max_tokens=max_tokens
            )

            if basic_analysis.get("type") == "error":
                raise Exception(
                    f"Document analysis failed: {basic_analysis.get('_result', {}).get('error', 'Unknown error')}"
                )

            # Step 2: Extract and structure content
            structured_content = await self._extract_structured_content(basic_analysis)

            # Step 3: Create document chunks
            chunks = await self._create_document_chunks(structured_content)

            # Step 4: Analyze document structure
            document_analysis = await self._analyze_document_structure(
                structured_content, chunks, basic_analysis
            )

            # Step 5: Store document context in database
            await self._store_document_context(
                media_id, interaction_id, user_id, document_analysis
            )

            # Step 6: Create embeddings for document chunks
            await self._create_document_embeddings(
                user_id, interaction_id, document_analysis
            )

            print(f"✅ Document processing completed successfully")
            return document_analysis

        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            raise

    async def _extract_structured_content(
        self, basic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract structured content from basic analysis"""
        try:
            result = basic_analysis.get("_result", {})
            content = result.get("content", "")

            # Extract questions if it's MCQ content
            questions = result.get("questions", [])

            structured_content = {
                "raw_content": content,
                "questions": questions,
                "document_type": basic_analysis.get("type", "unknown"),
                "language": basic_analysis.get("language", "english"),
                "title": basic_analysis.get("title", ""),
                "summary": basic_analysis.get("summary_title", ""),
            }

            # Parse questions into structured format
            if questions:
                structured_questions = []
                for i, q in enumerate(questions, 1):
                    structured_q = {
                        "question_number": str(i),
                        "question_text": q.get("question", ""),
                        "options": q.get("options", {}),
                        "answer": q.get("answer", ""),
                        "explanation": q.get("explanation", ""),
                        "difficulty": q.get("difficulty", "medium"),
                        "topic": q.get("topic", ""),
                    }
                    structured_questions.append(structured_q)

                structured_content["structured_questions"] = structured_questions

            return structured_content

        except Exception as e:
            print(f"⚠️ Error extracting structured content: {e}")
            return {"raw_content": "", "questions": [], "document_type": "unknown"}

    async def _create_document_chunks(
        self, structured_content: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create intelligent document chunks"""
        try:
            chunks = []
            raw_content = structured_content.get("raw_content", "")
            questions = structured_content.get("structured_questions", [])

            # Create question-specific chunks
            for question in questions:
                chunk_id = f"q_{question['question_number']}"

                # Build question content
                question_content = f"Question {question['question_number']}: {question['question_text']}\n\n"

                # Add options if available
                if question.get("options"):
                    for opt_key, opt_value in question["options"].items():
                        question_content += f"{opt_key.upper()}. {opt_value}\n"
                    question_content += "\n"

                # Add answer if available
                if question.get("answer"):
                    question_content += f"Answer: {question['answer']}\n\n"

                # Add explanation if available
                if question.get("explanation"):
                    question_content += f"Explanation: {question['explanation']}\n"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=question_content.strip(),
                    chunk_type="question",
                    question_number=question["question_number"],
                    metadata={
                        "difficulty": question.get("difficulty", "medium"),
                        "topic": question.get("topic", ""),
                        "has_options": bool(question.get("options")),
                        "has_explanation": bool(question.get("explanation")),
                    },
                )
                chunks.append(chunk)

            # Create content chunks for non-question content
            if raw_content and not questions:
                content_chunks = await self._chunk_text_content(raw_content)
                chunks.extend(content_chunks)

            # Limit total chunks
            if len(chunks) > self.max_chunks_per_document:
                chunks = chunks[: self.max_chunks_per_document]

            return chunks

        except Exception as e:
            print(f"⚠️ Error creating document chunks: {e}")
            return []

    async def _chunk_text_content(self, content: str) -> List[DocumentChunk]:
        """Chunk text content into manageable pieces"""
        try:
            chunks = []
            lines = content.split("\n")
            current_chunk = ""
            chunk_index = 0

            for line in lines:
                # Check if adding this line would exceed chunk size
                if len(current_chunk) + len(line) > self.chunk_size and current_chunk:
                    # Create chunk
                    chunk_id = f"content_{chunk_index}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=current_chunk.strip(),
                        chunk_type="paragraph",
                        metadata={"chunk_index": chunk_index},
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_text = (
                        current_chunk[-self.overlap_size :]
                        if len(current_chunk) > self.overlap_size
                        else current_chunk
                    )
                    current_chunk = overlap_text + "\n" + line
                    chunk_index += 1
                else:
                    current_chunk += "\n" + line if current_chunk else line

            # Add final chunk
            if current_chunk.strip():
                chunk_id = f"content_{chunk_index}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=current_chunk.strip(),
                    chunk_type="paragraph",
                    metadata={"chunk_index": chunk_index},
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"⚠️ Error chunking text content: {e}")
            return []

    async def _analyze_document_structure(
        self,
        structured_content: Dict[str, Any],
        chunks: List[DocumentChunk],
        basic_analysis: Dict[str, Any],
    ) -> DocumentAnalysis:
        """Analyze document structure and create comprehensive analysis"""
        try:
            questions = structured_content.get("structured_questions", [])
            raw_content = structured_content.get("raw_content", "")

            # Extract main topics
            main_topics = await self._extract_main_topics(raw_content, questions)

            # Determine difficulty level
            difficulty_level = await self._determine_difficulty_level(
                questions, raw_content
            )

            # Determine subject area
            subject_area = await self._determine_subject_area(raw_content, main_topics)

            # Create question mapping
            question_mapping = {}
            answer_key = {}

            for question in questions:
                q_num = question["question_number"]
                question_mapping[q_num] = question["question_text"]
                if question.get("answer"):
                    answer_key[q_num] = question["answer"]

            # Extract key concepts
            key_concepts = await self._extract_key_concepts(raw_content, questions)

            # Create content summary
            content_summary = await self._create_content_summary(structured_content)

            document_analysis = DocumentAnalysis(
                document_id=basic_analysis.get("title", "unknown"),
                document_type=structured_content.get("document_type", "unknown"),
                total_questions=len(questions),
                question_structure={
                    "has_options": any(q.get("options") for q in questions),
                    "has_explanations": any(q.get("explanation") for q in questions),
                    "difficulty_distribution": self._get_difficulty_distribution(
                        questions
                    ),
                },
                main_topics=main_topics,
                difficulty_level=difficulty_level,
                subject_area=subject_area,
                question_mapping=question_mapping,
                answer_key=answer_key,
                full_content=raw_content,
                content_summary=content_summary,
                key_concepts=key_concepts,
                chunks=chunks,
                metadata={
                    "language": structured_content.get("language", "english"),
                    "title": structured_content.get("title", ""),
                    "summary": structured_content.get("summary", ""),
                    "chunk_count": len(chunks),
                    "processing_timestamp": datetime.now().isoformat(),
                },
            )

            return document_analysis

        except Exception as e:
            print(f"⚠️ Error analyzing document structure: {e}")
            # Return minimal analysis
            return DocumentAnalysis(
                document_id="unknown",
                document_type="unknown",
                total_questions=0,
                question_structure={},
                main_topics=[],
                difficulty_level="unknown",
                subject_area="unknown",
                question_mapping={},
                answer_key={},
                full_content="",
                content_summary="",
                key_concepts=[],
                chunks=[],
                metadata={},
            )

    async def _extract_main_topics(
        self, content: str, questions: List[Dict]
    ) -> List[str]:
        """Extract main topics from content and questions"""
        try:
            topics = set()

            # Extract topics from questions
            for question in questions:
                if question.get("topic"):
                    topics.add(question["topic"])

            # Extract topics from content using keyword matching
            topic_keywords = {
                "mathematics": [
                    "math",
                    "algebra",
                    "geometry",
                    "calculus",
                    "trigonometry",
                    "equation",
                    "formula",
                ],
                "science": [
                    "physics",
                    "chemistry",
                    "biology",
                    "experiment",
                    "hypothesis",
                    "theory",
                ],
                "programming": [
                    "code",
                    "programming",
                    "algorithm",
                    "function",
                    "variable",
                    "loop",
                ],
                "language": [
                    "grammar",
                    "vocabulary",
                    "sentence",
                    "paragraph",
                    "essay",
                    "writing",
                ],
                "history": [
                    "historical",
                    "century",
                    "war",
                    "revolution",
                    "ancient",
                    "medieval",
                ],
            }

            content_lower = content.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.add(topic)

            return list(topics)

        except Exception as e:
            print(f"⚠️ Error extracting main topics: {e}")
            return []

    async def _determine_difficulty_level(
        self, questions: List[Dict], content: str
    ) -> str:
        """Determine overall difficulty level of the document"""
        try:
            if questions:
                # Use question difficulty levels
                difficulties = [q.get("difficulty", "medium") for q in questions]
                difficulty_counts = {}
                for diff in difficulties:
                    difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

                # Return most common difficulty
                return max(difficulty_counts, key=difficulty_counts.get)
            else:
                # Analyze content for difficulty indicators
                content_lower = content.lower()
                if any(
                    keyword in content_lower
                    for keyword in ["basic", "simple", "easy", "beginner"]
                ):
                    return "beginner"
                elif any(
                    keyword in content_lower
                    for keyword in ["advanced", "complex", "difficult", "expert"]
                ):
                    return "advanced"
                else:
                    return "intermediate"

        except Exception as e:
            print(f"⚠️ Error determining difficulty level: {e}")
            return "unknown"

    async def _determine_subject_area(
        self, content: str, main_topics: List[str]
    ) -> str:
        """Determine the primary subject area"""
        try:
            if main_topics:
                return main_topics[0]  # Return first topic as primary subject

            # Fallback to content analysis
            content_lower = content.lower()
            if any(
                keyword in content_lower
                for keyword in ["math", "mathematics", "algebra", "geometry"]
            ):
                return "mathematics"
            elif any(
                keyword in content_lower
                for keyword in ["science", "physics", "chemistry", "biology"]
            ):
                return "science"
            else:
                return "general"

        except Exception as e:
            print(f"⚠️ Error determining subject area: {e}")
            return "unknown"

    def _get_difficulty_distribution(self, questions: List[Dict]) -> Dict[str, int]:
        """Get distribution of difficulty levels in questions"""
        try:
            distribution = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
            for question in questions:
                difficulty = question.get("difficulty", "unknown")
                if difficulty in distribution:
                    distribution[difficulty] += 1
                else:
                    distribution["unknown"] += 1
            return distribution
        except Exception:
            return {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}

    async def _extract_key_concepts(
        self, content: str, questions: List[Dict]
    ) -> List[str]:
        """Extract key concepts from content and questions"""
        try:
            concepts = set()

            # Extract concepts from questions
            for question in questions:
                if question.get("topic"):
                    concepts.add(question["topic"])

            # Extract concepts from content using pattern matching
            concept_patterns = [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized terms
                r"\b\w+ing\b",  # -ing words
                r"\b\w+tion\b",  # -tion words
            ]

            for pattern in concept_patterns:
                matches = re.findall(pattern, content)
                concepts.update(matches[:5])  # Limit to 5 per pattern

            return list(concepts)[:15]  # Limit total concepts

        except Exception as e:
            print(f"⚠️ Error extracting key concepts: {e}")
            return []

    async def _create_content_summary(self, structured_content: Dict[str, Any]) -> str:
        """Create a summary of the document content"""
        try:
            questions = structured_content.get("structured_questions", [])
            raw_content = structured_content.get("raw_content", "")

            if questions:
                summary = f"Document contains {len(questions)} questions"
                if any(q.get("options") for q in questions):
                    summary += " with multiple choice options"
                if any(q.get("explanation") for q in questions):
                    summary += " and explanations"
                return summary
            else:
                # Create summary from raw content
                if len(raw_content) > 200:
                    return raw_content[:200] + "..."
                else:
                    return raw_content

        except Exception as e:
            print(f"⚠️ Error creating content summary: {e}")
            return "Content summary unavailable"

    async def _store_document_context(
        self,
        media_id: str,
        interaction_id: str,
        user_id: str,
        document_analysis: DocumentAnalysis,
    ) -> None:
        """Store document context in database"""
        try:
            async with AsyncSessionLocal() as db:
                # Check if document context already exists
                result = await db.execute(
                    select(DocumentContext).where(
                        and_(
                            DocumentContext.media_id == media_id,
                            DocumentContext.interaction_id == interaction_id,
                        )
                    )
                )
                existing_doc = result.scalar_one_or_none()

                if existing_doc:
                    # Update existing document context
                    existing_doc.document_type = document_analysis.document_type
                    existing_doc.total_questions = document_analysis.total_questions
                    existing_doc.question_structure = (
                        document_analysis.question_structure
                    )
                    existing_doc.main_topics = document_analysis.main_topics
                    existing_doc.difficulty_level = document_analysis.difficulty_level
                    existing_doc.subject_area = document_analysis.subject_area
                    existing_doc.question_mapping = document_analysis.question_mapping
                    existing_doc.answer_key = document_analysis.answer_key
                    existing_doc.full_content = document_analysis.full_content
                    existing_doc.content_summary = document_analysis.content_summary
                    existing_doc.key_concepts = document_analysis.key_concepts
                    existing_doc.updated_at = datetime.now()
                else:
                    # Create new document context
                    doc_context = DocumentContext(
                        media_id=media_id,
                        interaction_id=interaction_id,
                        user_id=user_id,
                        document_type=document_analysis.document_type,
                        total_questions=document_analysis.total_questions,
                        question_structure=document_analysis.question_structure,
                        main_topics=document_analysis.main_topics,
                        difficulty_level=document_analysis.difficulty_level,
                        subject_area=document_analysis.subject_area,
                        question_mapping=document_analysis.question_mapping,
                        answer_key=document_analysis.answer_key,
                        full_content=document_analysis.full_content,
                        content_summary=document_analysis.content_summary,
                        key_concepts=document_analysis.key_concepts,
                    )
                    db.add(doc_context)

                await db.commit()
                print(f"✅ Document context stored successfully")

        except Exception as e:
            print(f"❌ Error storing document context: {e}")
            raise

    async def _create_document_embeddings(
        self, user_id: str, interaction_id: str, document_analysis: DocumentAnalysis
    ) -> None:
        """Create embeddings for document chunks"""
        try:
            print(
                f"🔗 Creating embeddings for {len(document_analysis.chunks)} document chunks"
            )

            embedding_tasks = []
            for chunk in document_analysis.chunks:
                # Create enhanced metadata for the chunk
                metadata = {
                    "interaction_id": interaction_id,
                    "document_id": document_analysis.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "question_number": chunk.question_number,
                    "document_type": document_analysis.document_type,
                    "subject_area": document_analysis.subject_area,
                    "difficulty_level": document_analysis.difficulty_level,
                    "main_topics": document_analysis.main_topics,
                    "key_concepts": document_analysis.key_concepts,
                    "section_title": chunk.section_title,
                    **(chunk.metadata or {}),
                }

                # Create embedding task
                task = langchain_service.upsert_embedding(
                    conv_id=f"doc_{chunk.chunk_id}",
                    user_id=user_id,
                    text=chunk.content,
                    title=f"Document Chunk: {chunk.chunk_id}",
                    metadata=metadata,
                )
                embedding_tasks.append(task)

            # Execute embedding tasks in parallel
            if embedding_tasks:
                results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

                success_count = sum(
                    1 for result in results if not isinstance(result, Exception)
                )
                print(
                    f"✅ Created {success_count}/{len(embedding_tasks)} document embeddings"
                )

                # Log any failures
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"⚠️ Embedding failed for chunk {i}: {result}")

        except Exception as e:
            print(f"❌ Error creating document embeddings: {e}")
            raise

    async def search_documents(
        self,
        user_id: str,
        query: str,
        interaction_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        subject_areas: Optional[List[str]] = None,
        difficulty_levels: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search documents with advanced filtering"""
        try:
            # Use vector optimization service for enhanced search
            search_query = SearchQuery(
                query=query,
                user_id=user_id,
                interaction_id=interaction_id,
                top_k=top_k,
                use_hybrid_search=True,
                use_query_expansion=True,
                boost_recent=True,
            )

            # Get search results
            search_results = await vector_optimization_service.hybrid_search(
                search_query
            )

            # Filter results by document-specific criteria
            filtered_results = []
            for result in search_results:
                metadata = result.metadata

                # Apply document type filter
                if (
                    document_types
                    and metadata.get("document_type") not in document_types
                ):
                    continue

                # Apply subject area filter
                if subject_areas and metadata.get("subject_area") not in subject_areas:
                    continue

                # Apply difficulty level filter
                if (
                    difficulty_levels
                    and metadata.get("difficulty_level") not in difficulty_levels
                ):
                    continue

                # Add document-specific information
                result_dict = {
                    "id": result.id,
                    "content": result.content,
                    "title": result.title,
                    "score": result.score,
                    "document_type": metadata.get("document_type", "unknown"),
                    "subject_area": metadata.get("subject_area", "unknown"),
                    "difficulty_level": metadata.get("difficulty_level", "unknown"),
                    "question_number": metadata.get("question_number"),
                    "chunk_type": metadata.get("chunk_type", "unknown"),
                    "main_topics": metadata.get("main_topics", []),
                    "key_concepts": metadata.get("key_concepts", []),
                    "metadata": metadata,
                }

                filtered_results.append(result_dict)

            return filtered_results

        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

    async def get_document_by_question_number(
        self, user_id: str, interaction_id: str, question_number: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific document content by question number"""
        try:
            async with AsyncSessionLocal() as db:
                # Get document context
                result = await db.execute(
                    select(DocumentContext).where(
                        and_(
                            DocumentContext.user_id == user_id,
                            DocumentContext.interaction_id == interaction_id,
                        )
                    )
                )
                doc_context = result.scalar_one_or_none()

                if not doc_context:
                    return None

                # Get question content
                question_text = doc_context.question_mapping.get(question_number)
                answer = doc_context.answer_key.get(question_number)

                if not question_text:
                    return None

                return {
                    "question_number": question_number,
                    "question_text": question_text,
                    "answer": answer,
                    "document_type": doc_context.document_type,
                    "subject_area": doc_context.subject_area,
                    "difficulty_level": doc_context.difficulty_level,
                    "main_topics": doc_context.main_topics,
                    "key_concepts": doc_context.key_concepts,
                }

        except Exception as e:
            print(f"❌ Error getting document by question number: {e}")
            return None

    async def get_document_analytics(
        self, user_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Get document usage analytics"""
        try:
            async with AsyncSessionLocal() as db:
                # Get document statistics
                cutoff_date = datetime.now() - timedelta(days=days)

                result = await db.execute(
                    select(
                        func.count(DocumentContext.id).label("total_documents"),
                        func.count(DocumentContext.id.distinct()).label(
                            "unique_documents"
                        ),
                        func.avg(DocumentContext.total_questions).label(
                            "avg_questions"
                        ),
                    ).where(
                        and_(
                            DocumentContext.user_id == user_id,
                            DocumentContext.created_at >= cutoff_date,
                        )
                    )
                )

                stats = result.first()

                # Get document type distribution
                type_result = await db.execute(
                    select(
                        DocumentContext.document_type,
                        func.count(DocumentContext.id).label("count"),
                    )
                    .where(
                        and_(
                            DocumentContext.user_id == user_id,
                            DocumentContext.created_at >= cutoff_date,
                        )
                    )
                    .group_by(DocumentContext.document_type)
                )

                type_distribution = {
                    row.document_type: row.count for row in type_result
                }

                return {
                    "user_id": user_id,
                    "period_days": days,
                    "total_documents": stats.total_documents or 0,
                    "unique_documents": stats.unique_documents or 0,
                    "average_questions_per_document": float(stats.avg_questions or 0),
                    "document_type_distribution": type_distribution,
                }

        except Exception as e:
            print(f"❌ Error getting document analytics: {e}")
            return {}


# Global instance
document_integration_service = DocumentIntegrationService()
