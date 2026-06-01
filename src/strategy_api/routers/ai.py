from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, delete, desc
from sqlalchemy.orm import Session
from strategy_api.db.session import get_session
from strategy_api.db.models import ChatMessage, ResearchSession, StrategyInsight
from strategy_api.services import ollama
from strategy_api.schemas import (
    ChatReq,
    GenStrategyReq,
    MarketAnalysisReq,
    SaveSessionReq,
)

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/health")
async def health():
    ok = await ollama.health_check()
    models = await ollama.list_models() if ok else []
    return {
        "ok": ok,
        "models": models,
        "message": "Ollama connected" if ok else "Ollama unavailable - check OLLAMA_BASE_URL",
    }


@router.post("/chat")
async def chat(req: ChatReq, db: Session = Depends(get_session)):
    res = await ollama.chat_completion(
        [m.model_dump() for m in req.messages], req.model, req.temperature
    )
    content = res["message"]["content"]
    if req.sessionId:
        db.add(ChatMessage(session_id=req.sessionId, role="user",
                           content=req.messages[-1].content, model=req.model))
        db.add(ChatMessage(session_id=req.sessionId, role="assistant",
                           content=content, model=req.model))
        db.commit()
    return {
        "content": content,
        "model": req.model,
        "timing": {
            "total": res.get("total_duration"),
            "promptTokens": res.get("prompt_eval_count"),
            "completionTokens": res.get("eval_count"),
        },
    }


@router.get("/history")
def history(sessionId: str = Query(...), db: Session = Depends(get_session)):
    rows = db.scalars(
        select(ChatMessage).where(ChatMessage.session_id == sessionId)
        .order_by(ChatMessage.created_at)
    ).all()
    return [{"id": r.id, "sessionId": r.session_id, "role": r.role, "content": r.content,
             "model": r.model, "createdAt": r.created_at} for r in rows]


@router.post("/generate-strategy")
async def generate_strategy(req: GenStrategyReq, db: Session = Depends(get_session)):
    analysis = await ollama.generate_strategy(
        req.symbol, req.rr, req.features, req.metrics, req.model
    )
    if req.sessionId:
        db.add(StrategyInsight(session_id=req.sessionId,
                               strategy_name=f"{req.symbol}_{req.rr}",
                               insight_type="analysis", content=analysis, model=req.model))
        db.commit()
    return {"analysis": analysis}


@router.post("/market-analysis")
async def market_analysis(req: MarketAnalysisReq):
    analysis = await ollama.analyze_market(req.symbol, req.priceData.model_dump(), req.model)
    return {"analysis": analysis}


@router.post("/save-session")
def save_session(req: SaveSessionReq, db: Session = Depends(get_session)):
    db.add(ResearchSession(session_id=req.sessionId, symbol=req.symbol, rr_config=req.rrConfig,
                           leverage=req.leverage, days=req.days, status="completed",
                           result_json=req.resultJson, best_strategy=req.bestStrategy,
                           win_rate=req.winRate, profit_factor=req.profitFactor,
                           expectancy=req.expectancy))
    db.commit()
    return {"success": True}


@router.get("/sessions")
def sessions(db: Session = Depends(get_session)):
    rows = db.scalars(
        select(ResearchSession).order_by(desc(ResearchSession.created_at)).limit(50)
    ).all()
    return [{"id": r.id, "sessionId": r.session_id, "symbol": r.symbol, "rrConfig": r.rr_config,
             "leverage": r.leverage, "days": r.days, "status": r.status,
             "bestStrategy": r.best_strategy, "winRate": r.win_rate,
             "profitFactor": r.profit_factor, "expectancy": r.expectancy,
             "createdAt": r.created_at} for r in rows]


@router.get("/insights")
def insights(sessionId: str = Query(...), db: Session = Depends(get_session)):
    rows = db.scalars(
        select(StrategyInsight).where(StrategyInsight.session_id == sessionId)
        .order_by(desc(StrategyInsight.created_at))
    ).all()
    return [{"id": r.id, "sessionId": r.session_id, "strategyName": r.strategy_name,
             "insightType": r.insight_type, "content": r.content, "model": r.model,
             "createdAt": r.created_at} for r in rows]


@router.post("/clear-history")
def clear_history(sessionId: str = Query(...), db: Session = Depends(get_session)):
    db.execute(delete(ChatMessage).where(ChatMessage.session_id == sessionId))
    db.commit()
    return {"success": True}
