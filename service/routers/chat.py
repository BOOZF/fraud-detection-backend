from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services import chat

router = APIRouter(prefix="/api")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatIn(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=20)
    context: str | None = Field(default=None, max_length=1500)  # text the user highlighted on screen
    alert_id: int | None = Field(default=None, ge=1)  # the single alert the highlighted text belongs to


@router.post("/chat")
def post_chat(body: ChatIn):
    if body.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user")
    try:
        return chat.answer([m.model_dump() for m in body.messages], body.context, body.alert_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Copilot unavailable: {e}")
