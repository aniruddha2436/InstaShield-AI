from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

Count = Annotated[int, Field(strict=True, ge=0, le=2_000_000_000)]
class Profile(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    username: str = Field(min_length=1, max_length=30, pattern=r'^[A-Za-z0-9._]+$')
    profile_picture: Annotated[int, Field(strict=True, ge=0, le=1)]
    bio_length: Annotated[int, Field(strict=True, ge=0, le=10000)]
    followers: Count
    following: Count
    posts: Count
    account_age_days: Annotated[int, Field(strict=True, ge=0, le=100000)]
    username_changed: Count
