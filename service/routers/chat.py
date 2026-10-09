from typing import Literal

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


@router.post("/chat/stream")
def post_chat_stream(body: ChatIn):
    """The same chat as /chat, as server-sent events: steps (what the copilot is doing), answer snapshots as the answer
    is written, then a final `done` event (or `error`)."""
    if body.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user")
    try:
        chat.resolve_alert_id(body.alert_id, body.context)  # an unknown alert is a 404 before the stream starts
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    history = [m.model_dump() for m in body.messages]

    def events():
        try:
            for event in chat.answer_stream(history, body.context, body.alert_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # the stream has started, so the failure is an event, not a status code
            yield f"data: {json.dumps({'type': 'error', 'message': f'Copilot unavailable: {e}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
