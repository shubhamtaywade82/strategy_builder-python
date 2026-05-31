import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strategy_api.config import settings

def create_app() -> FastAPI:
    app = FastAPI(title="Strategy Builder API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/ping")
    def ping():
        return {"ok": True, "ts": int(time.time() * 1000)}

    from strategy_api.routers import market
    app.include_router(market.router)

    from strategy_api.routers import ai
    app.include_router(ai.router)

    return app

app = create_app()

def run():
    import uvicorn
    uvicorn.run("strategy_api.main:app", host="127.0.0.1", port=8000, reload=False)
