from __future__ import annotations
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from strategy_api.db.session import get_session
from strategy_api.schemas import ResearchRunReq
from strategy_api.services import research as research_svc
from strategy_api.services.research import _json_safe

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("/run")
async def run(req: ResearchRunReq, db: Session = Depends(get_session)):
    return await research_svc.run(
        req.symbol, req.days, req.leverage, req.horizon, req.rrs, db
    )


@router.post("/run-stream")
async def run_stream(req: ResearchRunReq, db: Session = Depends(get_session)):
    async def event_generator():
        async for chunk in research_svc.run_stream(
            req.symbol, req.days, req.leverage, req.horizon, req.rrs, db
        ):
            yield f"data: {json.dumps(_json_safe(chunk))}\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream"
    )
