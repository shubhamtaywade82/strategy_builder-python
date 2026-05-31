from __future__ import annotations
from fastapi import APIRouter
from strategy_api.schemas import ResearchRunReq
from strategy_api.services import research as research_svc

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("/run")
async def run(req: ResearchRunReq):
    return await research_svc.run(req.symbol, req.days, req.leverage, req.horizon, req.rrs)
