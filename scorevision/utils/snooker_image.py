"""Versioned, image-coordinate primitives. No video or inferred ball states."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ELEMENT_ID = "manako/DetectSnookerImagePrimitivesV1"
GROUNDTRUTH_TYPE = "snooker_image_primitives_v1"
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
BallLabel = Literal["white", "red", "yellow", "green", "brown", "blue", "pink", "black"]
PocketName = Literal["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"]


class Primitive(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ball(Primitive):
    label: BallLabel
    bbox: tuple[Unit, Unit, Unit, Unit]  # x, y, width, height

    @model_validator(mode="after")
    def valid_extent(self):
        x, y, w, h = self.bbox
        if min(w, h) <= 0 or x+w > 1+1e-9 or y+h > 1+1e-9:
            raise ValueError("Ball box must have positive area and remain inside the image")
        return self


class Pocket(Primitive):
    name: PocketName
    x: Unit
    y: Unit


class SnookerImagePrediction(Primitive):
    type: Literal["snooker_image_primitives_v1"]
    width: int = Field(strict=True, ge=1, le=4096)
    height: int = Field(strict=True, ge=1, le=4096)
    balls: list[Ball] = Field(max_length=64)
    pockets: list[Pocket] = Field(max_length=6)
    surface: list[tuple[Unit, Unit]] = Field(max_length=256)

    @model_validator(mode="after")
    def geometry(self):
        if len({p.name for p in self.pockets}) != len(self.pockets):
            raise ValueError("Pocket names must be unique")
        if self.surface and len(self.surface) < 3:
            raise ValueError("Surface must be an empty miss or a polygon with at least three points")
        if self.width*self.height > 8_388_608:
            raise ValueError("Image exceeds the phase-one pixel limit")
        return self


class SnookerImageGroundTruth(SnookerImagePrediction):
    # Human contours must not be truncated to the miner response budget.
    surface: list[tuple[Unit, Unit]] = Field(max_length=8192)


def from_record(row: dict) -> SnookerImageGroundTruth:
    """Convert canonical human records by category name, never export-specific ID."""
    w, h = row["width"], row["height"]
    polygon = row["surface"]["polygon"]
    return SnookerImageGroundTruth(
        type=GROUNDTRUTH_TYPE, width=w, height=h,
        balls=[{"label": b["class"], "bbox": [b["bbox"][0]/w, b["bbox"][1]/h,
                 b["bbox"][2]/w, b["bbox"][3]/h]} for b in row["balls"]],
        pockets=[{"name": name.replace(" ", "_"), "x": point[0]/w, "y": point[1]/h}
                 for name, point in row["pockets"].items()],
        surface=[(polygon[i]/w, polygon[i+1]/h) for i in range(0, len(polygon), 2)],
    )
