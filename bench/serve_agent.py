"""vllm serve with the /kv control routes for the retention knob.

Launches the stock OpenAI-compatible server and adds four routes that
forward to the engine's KV protection utilities (branch agent-knobs):

  POST /kv/protect  {"request_id": str, "deadline_ts": float} -> {"parked": int}
  POST /kv/release  {"request_id": str}                       -> {"released": int}
  POST /kv/evict    {"request_id": str}                       -> {"evicted": int}
  GET  /kv/stats                                              -> stats dict

Usage (flags as in `vllm serve`, model via --model):
  VLLM_KV_PROTECTION=1 python -m harness.serve_agent --model <model> [args]

The wrapper only appends a router to the app that build_app returns;
every other server behavior is stock.
"""

from __future__ import annotations

import functools

import uvloop
from fastapi import APIRouter, Request

from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

kv_router = APIRouter()


async def _utility(request: Request, method: str, *args):
    engine_client = request.app.state.engine_client
    return await engine_client.engine_core.call_utility_async(method, *args)


@kv_router.post("/kv/protect")
async def kv_protect(raw: Request):
    body = await raw.json()
    parked = await _utility(
        raw, "kv_protect", body["request_id"], float(body["deadline_ts"])
    )
    return {"parked": parked}


@kv_router.post("/kv/release")
async def kv_release(raw: Request):
    body = await raw.json()
    released = await _utility(raw, "kv_release", body["request_id"])
    return {"released": released}


@kv_router.post("/kv/evict")
async def kv_evict(raw: Request):
    body = await raw.json()
    evicted = await _utility(raw, "kv_evict", body["request_id"])
    return {"evicted": evicted}


@kv_router.get("/kv/stats")
async def kv_stats(raw: Request):
    return await _utility(raw, "kv_protection_stats")


def _build_app_with_kv(orig_build_app, *args, **kwargs):
    app = orig_build_app(*args, **kwargs)
    app.include_router(kv_router)
    return app


def main() -> None:
    api_server.build_app = functools.partial(
        _build_app_with_kv, api_server.build_app
    )
    parser = FlexibleArgumentParser(description="vllm serve + /kv control routes")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
