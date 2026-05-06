"""
Pydantic output model for Step 1.

Define the structure the LLM must return.  The schema is automatically
included in prompts and used for validation + auto-fixing.

This template extracts a small set of structured items from a text:
named persons (with optional role) and events (with the persons involved).
Replace these classes with your own schema.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(..., description="Full name of the person as mentioned in the text")
    role: Optional[str] = Field(None, description="Title, profession, or role, if mentioned")


class Event(BaseModel):
    description: str = Field(..., description="Short factual description of what happened")
    actors: List[str] = Field(default_factory=list, description="Names of persons involved")


class Step1Output(BaseModel):
    """Replace this with your actual output structure."""

    persons: List[Person] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @classmethod
    def get_example_output(cls) -> dict:
        """Example output shown to the LLM in the prompt.

        Keep this realistic — the LLM uses it to understand the expected format.
        """
        return {
            "persons": [
                {"name": "Maximilian I.", "role": "Kaiser"},
                {"name": "Paul von Liechtenstein", "role": None},
            ],
            "events": [
                {
                    "description": "Maximilian I. orders Paul von Liechtenstein to be ready to come to him.",
                    "actors": ["Maximilian I.", "Paul von Liechtenstein"],
                },
            ],
        }
