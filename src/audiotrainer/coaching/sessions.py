"""Session containers for application-level workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from audiotrainer.api.schemas import FeedbackItem


class CoachingSession(BaseModel):
    """Small immutable-style session record without hidden global state."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    mode: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback: list[FeedbackItem] = Field(default_factory=list)

    def with_feedback(self, items: list[FeedbackItem]) -> "CoachingSession":
        """Return a copy with appended feedback items."""

        return self.model_copy(update={"feedback": [*self.feedback, *items]})
