from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from api.production_middleware import BodySizeLimitMiddleware, RateLimitMiddleware


async def _ok(_request):
    return JSONResponse({"ok": True})


def test_body_size_rejects_large_content_length():
    app = Starlette(routes=[Route("/v1/predict", _ok, methods=["POST"])])
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=20)
    c = TestClient(app)
    body = b'{"features":{"a":1}}' * 10
    r = c.post("/v1/predict", content=body, headers={"content-length": str(len(body))})
    assert r.status_code == 413


def test_rate_limit_returns_429():
    app = Starlette(routes=[Route("/x", _ok, methods=["GET"])])
    app.add_middleware(RateLimitMiddleware, per_minute=2)
    c = TestClient(app)
    assert c.get("/x").status_code == 200
    assert c.get("/x").status_code == 200
    r3 = c.get("/x")
    assert r3.status_code == 429
