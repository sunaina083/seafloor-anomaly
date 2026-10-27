"""
schemas.py

Pydantic models defining the shape of API requests/responses.
"""

from typing import Optional, Literal
from pydantic import BaseModel


class TileOut(BaseModel):
    tile_id: str
    rank: int
    ensemble_score: float
    ae_score: float
    baseline_score: float
    flag_status: Optional[str] = None  # filled in from the flags table if one exists


class FlagIn(BaseModel):
    status: Literal["confirmed_debris", "false_positive", "unsure"]
    note: Optional[str] = None


class FlagOut(BaseModel):
    tile_id: str
    status: str
    note: Optional[str] = None

    class Config:
        from_attributes = True  # allows creating this from an ORM object directly