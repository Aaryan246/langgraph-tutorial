from typing import Dict, List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(description="250 word detailed answer to the question")
    # NOTE : Local models struggles with nested pydantic schemas
    # reflection: Reflection = Field(description="Your reflection of the initial answer")
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer")

class ReviseAnswer(BaseModel):
    """Revise your original answer to the question"""
    answer: str = Field(description="The updated, revised 250-word detailed answer")
    missing: str = Field(description="Critique of what is still missing")
    superfluous: str = Field(description="Critique of what is still superfluous")
    search_queries: List[str] = Field(description="1-3 subsequent search queries if more research is required")
    references: List[str] = Field(description="Citations motivating your updated answer")