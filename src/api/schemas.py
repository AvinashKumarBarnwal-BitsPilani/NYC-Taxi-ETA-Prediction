"""Pydantic schemas for the NYC Taxi ETA API."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for NYC Taxi ETA prediction."""

    vendor_id: Literal[1, 2]
    passenger_count: int = Field(ge=0, le=6)
    store_and_fwd_flag: Literal["N", "Y"]
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day_of_week: int = Field(ge=0, le=6)
    pickup_month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    distance_km: float = Field(gt=0)