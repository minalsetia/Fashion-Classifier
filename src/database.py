"""SQLite prediction-tracking database (via SQLAlchemy).

Per the brief we store prediction *tracking records* only — never the image
files themselves. Each row captures one model run on one image: what was
predicted, how confident, which model version, when, and whether it succeeded.

Public helpers:
    init_db()                 -> create tables if missing
    log_prediction(...)       -> insert one tracking row, returns its id
    fetch_history(limit=...)  -> recent rows as a pandas DataFrame
    new_run_id()              -> uuid for grouping a single/batch run
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from . import config

Base = declarative_base()
_engine = create_engine(f"sqlite:///{config.DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=_engine, future=True)


class Prediction(Base):
    """One prediction-tracking record."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)          # groups a single/batch run
    run_type = Column(String)                     # "single" | "batch"
    image_ref = Column(String)                    # image URL or local path

    predicted_gender = Column(String, nullable=True)
    gender_confidence = Column(Float, nullable=True)
    predicted_sleeve = Column(String, nullable=True)
    sleeve_confidence = Column(Float, nullable=True)

    model_version = Column(String)                # e.g. "resnet18_gender_v1+..."
    status = Column(String, default="success")    # "success" | "error"
    error_message = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def init_db() -> None:
    """Create the predictions table if it does not exist."""
    Base.metadata.create_all(_engine)


def new_run_id() -> str:
    """A short unique id to group all rows of one single/batch run."""
    return uuid.uuid4().hex[:12]


def log_prediction(
    *,
    run_id: str,
    run_type: str,
    image_ref: str,
    predicted_gender: str | None = None,
    gender_confidence: float | None = None,
    predicted_sleeve: str | None = None,
    sleeve_confidence: float | None = None,
    model_version: str = "",
    status: str = "success",
    error_message: str | None = None,
) -> int:
    """Insert one tracking row and return its primary-key id."""
    init_db()
    with SessionLocal() as session:
        row = Prediction(
            run_id=run_id,
            run_type=run_type,
            image_ref=image_ref,
            predicted_gender=predicted_gender,
            gender_confidence=gender_confidence,
            predicted_sleeve=predicted_sleeve,
            sleeve_confidence=sleeve_confidence,
            model_version=model_version,
            status=status,
            error_message=error_message,
        )
        session.add(row)
        session.commit()
        return row.id


def fetch_history(limit: int = 200) -> pd.DataFrame:
    """Return the most recent tracking rows as a DataFrame (newest first)."""
    init_db()
    with SessionLocal() as session:
        rows = (
            session.query(Prediction)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .all()
        )
        return pd.DataFrame([r.as_dict() for r in rows])


if __name__ == "__main__":
    init_db()
    print(f"Initialised database at {config.DB_PATH}")
