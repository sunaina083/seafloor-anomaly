"""
models.py

ORM model for analyst flags. A flag records that an analyst reviewed a
tile and marked it (confirmed debris, false positive, or unsure).
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func

from .database import Base


class Flag(Base):
    __tablename__ = "flags"

    tile_id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False)  # "confirmed_debris" | "false_positive" | "unsure"
    note = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())