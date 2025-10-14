"""
Enhanced semantic summary service with validation and versioning
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.interaction import Interaction
from app.models.context import ConversationContext
from app.services.langchain_service import langchain_service


class SemanticSummaryService:
    """Enhanced semantic summary service with validation and versioning"""

    def __init__(self):
        self.max_summary_length = 500
        self.max_facts = 10
        self.max_topics = 8

    async def create_conversation_summary(
        self, user_message: str, ai_response: str
    ) -> Dict[str, Any]:
        """
        Create a semantic summary from a single conversation exchange with validation
        """
        try:
            # Use the enhanced conversation summarization chain
            from app.config.langchain_config import StudyGuruConfig

            chain = StudyGuruConfig.CHAINS.get_conversation_summarization_chain()

            # Truncate inputs to prevent token overflow
            user_msg_truncated = (
                user_message[:500] if len(user_message) > 500 else user_message
            )
            ai_resp_truncated = (
                ai_response[:1000] if len(ai_response) > 1000 else ai_response
            )

            result = await chain.ainvoke(
                {
                    "user_message": user_msg_truncated,
                    "ai_response": ai_resp_truncated,
                }
            )

            # Validate and enhance the result
            validated_result = self._validate_and_enhance_summary(result)

            return validated_result

        except Exception as e:
            print(f"Error creating conversation summary: {e}")
            return self._get_fallback_summary(user_message, ai_response)

    async def update_interaction_summary(
        self,
        current_summary: Optional[Dict[str, Any]],
        new_user_message: str,
        new_ai_response: str,
    ) -> Dict[str, Any]:
        """
        Update the running semantic summary with validation and versioning
        """
        try:
            # For first conversation, create initial summary
            if not current_summary or not current_summary.get("updated_summary"):
                conv_summary = await self.create_conversation_summary(
                    new_user_message, new_ai_response
                )

                initial_summary = {
                    "updated_summary": conv_summary.get("semantic_summary", ""),
                    "key_topics": conv_summary.get("main_topics", []),
                    "recent_focus": conv_summary.get("context_for_future", ""),
                    "accumulated_facts": conv_summary.get("key_facts", []),
                    "question_numbers": conv_summary.get("question_numbers", []),
                    "learning_progression": conv_summary.get("learning_progress", ""),
                    "difficulty_trend": conv_summary.get(
                        "difficulty_level", "beginner"
                    ),
                    "learning_patterns": [],
                    "struggling_areas": [],
                    "mastered_concepts": [],
                    "version": 1,
                    "last_updated": datetime.now().isoformat(),
                }

                return self._validate_and_enhance_summary(initial_summary)

            # Update existing summary using enhanced chain
            from app.config.langchain_config import StudyGuruConfig

            chain = StudyGuruConfig.CHAINS.get_interaction_summary_update_chain()

            # Truncate inputs
            current_summary_text = current_summary.get("updated_summary", "")[:1000]
            new_user_truncated = (
                new_user_message[:500]
                if len(new_user_message) > 500
                else new_user_message
            )
            new_ai_truncated = (
                new_ai_response[:1000]
                if len(new_ai_response) > 1000
                else new_ai_response
            )

            result = await chain.ainvoke(
                {
                    "current_summary": current_summary_text,
                    "new_user_message": new_user_truncated,
                    "new_ai_response": new_ai_truncated,
                }
            )

            # Merge with existing summary data
            updated_summary = {
                **current_summary,  # Keep existing data
                **result,  # Update with new data
                "version": current_summary.get("version", 1) + 1,
                "last_updated": datetime.now().isoformat(),
            }

            # Validate and enhance the updated summary
            validated_summary = self._validate_and_enhance_summary(updated_summary)

            return validated_summary

        except Exception as e:
            print(f"Error updating interaction summary: {e}")
            return current_summary or self._get_fallback_summary(
                new_user_message, new_ai_response
            )

    def _validate_and_enhance_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance a semantic summary
        """
        try:
            # Ensure required fields exist
            validated = {
                "updated_summary": summary.get("updated_summary", ""),
                "key_topics": summary.get("key_topics", []),
                "recent_focus": summary.get("recent_focus", ""),
                "accumulated_facts": summary.get("accumulated_facts", []),
                "question_numbers": summary.get("question_numbers", []),
                "learning_progression": summary.get("learning_progression", ""),
                "difficulty_trend": summary.get("difficulty_trend", "beginner"),
                "learning_patterns": summary.get("learning_patterns", []),
                "struggling_areas": summary.get("struggling_areas", []),
                "mastered_concepts": summary.get("mastered_concepts", []),
                "version": summary.get("version", 1),
                "last_updated": summary.get("last_updated", datetime.now().isoformat()),
            }

            # Validate and clean data
            validated["updated_summary"] = self._clean_text(
                validated["updated_summary"]
            )
            validated["recent_focus"] = self._clean_text(validated["recent_focus"])
            validated["learning_progression"] = self._clean_text(
                validated["learning_progression"]
            )

            # Ensure lists are properly formatted
            for list_field in [
                "key_topics",
                "accumulated_facts",
                "question_numbers",
                "learning_patterns",
                "struggling_areas",
                "mastered_concepts",
            ]:
                if not isinstance(validated[list_field], list):
                    validated[list_field] = []
                else:
                    # Clean and limit list items
                    validated[list_field] = [
                        self._clean_text(str(item)) for item in validated[list_field]
                    ][
                        : (
                            self.max_facts
                            if list_field == "accumulated_facts"
                            else self.max_topics
                        )
                    ]

            # Validate difficulty trend
            if validated["difficulty_trend"] not in [
                "beginner",
                "intermediate",
                "advanced",
            ]:
                validated["difficulty_trend"] = "beginner"

            # Extract question numbers from text if not provided
            if not validated["question_numbers"]:
                validated["question_numbers"] = self._extract_question_numbers(
                    validated["updated_summary"] + " " + validated["recent_focus"]
                )

            # Compress summary if too long
            if len(validated["updated_summary"]) > self.max_summary_length:
                validated["updated_summary"] = self._compress_summary(
                    validated["updated_summary"]
                )

            return validated

        except Exception as e:
            print(f"Error validating summary: {e}")
            return self._get_fallback_summary("", "")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove any non-printable characters
        text = "".join(char for char in text if char.isprintable() or char.isspace())

        return text

    def _extract_question_numbers(self, text: str) -> List[int]:
        """Extract question numbers from text"""
        question_numbers = []
        patterns = [
            r"question\s+(\d+)",
            r"problem\s+(\d+)",
            r"mcq\s+(\d+)",
            r"equation\s+(\d+)",
            r"(\d+)\.\s*$",  # Number at end of line
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match)
                    if num not in question_numbers:
                        question_numbers.append(num)
                except ValueError:
                    continue

        return sorted(question_numbers)

    def _compress_summary(self, summary: str) -> str:
        """Compress summary to fit within length limit"""
        if len(summary) <= self.max_summary_length:
            return summary

        # Try to find a good breaking point
        sentences = summary.split(". ")
        compressed = ""

        for sentence in sentences:
            if len(compressed + sentence + ". ") <= self.max_summary_length:
                compressed += sentence + ". "
            else:
                break

        return compressed.strip()

    def _get_fallback_summary(
        self, user_message: str, ai_response: str
    ) -> Dict[str, Any]:
        """Get a fallback summary when processing fails"""
        return {
            "updated_summary": "Educational conversation about various topics.",
            "key_topics": ["General Discussion"],
            "recent_focus": "User is engaged in educational learning.",
            "accumulated_facts": [],
            "question_numbers": [],
            "learning_progression": "Learning in progress",
            "difficulty_trend": "beginner",
            "learning_patterns": [],
            "struggling_areas": [],
            "mastered_concepts": [],
            "version": 1,
            "last_updated": datetime.now().isoformat(),
        }

    async def get_summary_version_history(
        self, interaction_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get version history of semantic summaries for an interaction
        """
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(ConversationContext)
                    .where(
                        and_(
                            ConversationContext.interaction_id == interaction_id,
                            ConversationContext.context_type == "semantic_summary",
                        )
                    )
                    .order_by(desc(ConversationContext.created_at))
                    .limit(limit)
                )

                contexts = result.scalars().all()

                version_history = []
                for context in contexts:
                    version_history.append(
                        {
                            "version": context.context_data.get("version", 1),
                            "created_at": (
                                context.created_at.isoformat()
                                if context.created_at
                                else ""
                            ),
                            "summary_preview": context.context_data.get(
                                "updated_summary", ""
                            )[:100]
                            + "...",
                            "topics_count": len(
                                context.context_data.get("key_topics", [])
                            ),
                            "facts_count": len(
                                context.context_data.get("accumulated_facts", [])
                            ),
                        }
                    )

                return version_history

        except Exception as e:
            print(f"Error getting summary version history: {e}")
            return []

    async def store_summary_version(
        self, interaction_id: str, user_id: str, summary_data: Dict[str, Any]
    ) -> bool:
        """
        Store a version of the semantic summary for history tracking
        """
        try:
            async with AsyncSessionLocal() as db:
                # Create context entry for version tracking
                context_entry = ConversationContext(
                    interaction_id=interaction_id,
                    user_id=user_id,
                    context_type="semantic_summary",
                    context_data=summary_data,
                    context_hash=self._generate_summary_hash(summary_data),
                    content_length=len(summary_data.get("updated_summary", "")),
                    topic_tags=summary_data.get("key_topics", []),
                    question_numbers=summary_data.get("question_numbers", []),
                    created_at=datetime.now(),
                )

                db.add(context_entry)
                await db.commit()

                return True

        except Exception as e:
            print(f"Error storing summary version: {e}")
            return False

    def _generate_summary_hash(self, summary_data: Dict[str, Any]) -> str:
        """Generate a hash for the summary data"""
        import hashlib

        # Create a string representation of the summary
        summary_string = json.dumps(summary_data, sort_keys=True)
        return hashlib.md5(summary_string.encode()).hexdigest()

    async def analyze_summary_quality(
        self, summary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the quality of a semantic summary
        """
        try:
            quality_metrics = {
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "relevance_score": 0.0,
                "overall_score": 0.0,
                "issues": [],
                "recommendations": [],
            }

            # Check completeness
            required_fields = ["updated_summary", "key_topics", "accumulated_facts"]
            present_fields = sum(
                1 for field in required_fields if summary_data.get(field)
            )
            quality_metrics["completeness_score"] = present_fields / len(
                required_fields
            )

            if quality_metrics["completeness_score"] < 1.0:
                quality_metrics["issues"].append("Missing required fields")
                quality_metrics["recommendations"].append(
                    "Ensure all required fields are present"
                )

            # Check clarity (summary length and structure)
            summary_text = summary_data.get("updated_summary", "")
            if 50 <= len(summary_text) <= 500:
                quality_metrics["clarity_score"] = 1.0
            elif len(summary_text) < 50:
                quality_metrics["clarity_score"] = 0.5
                quality_metrics["issues"].append("Summary too short")
                quality_metrics["recommendations"].append(
                    "Add more detail to the summary"
                )
            else:
                quality_metrics["clarity_score"] = 0.7
                quality_metrics["issues"].append("Summary too long")
                quality_metrics["recommendations"].append("Compress the summary")

            # Check relevance (presence of educational content)
            educational_indicators = [
                "learn",
                "understand",
                "problem",
                "solution",
                "concept",
                "equation",
            ]
            summary_lower = summary_text.lower()
            educational_count = sum(
                1 for indicator in educational_indicators if indicator in summary_lower
            )
            quality_metrics["relevance_score"] = min(1.0, educational_count / 3)

            if quality_metrics["relevance_score"] < 0.5:
                quality_metrics["issues"].append("Low educational relevance")
                quality_metrics["recommendations"].append(
                    "Focus more on educational content"
                )

            # Calculate overall score
            quality_metrics["overall_score"] = (
                quality_metrics["completeness_score"]
                + quality_metrics["clarity_score"]
                + quality_metrics["relevance_score"]
            ) / 3

            return quality_metrics

        except Exception as e:
            print(f"Error analyzing summary quality: {e}")
            return {
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "relevance_score": 0.0,
                "overall_score": 0.0,
                "issues": ["Analysis failed"],
                "recommendations": ["Check summary format"],
            }


# Global instance
semantic_summary_service = SemanticSummaryService()
