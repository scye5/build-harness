"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


class ChessSandboxUnavailable(RuntimeError):
    """The execution environment cannot service further actions."""


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        fen = args.get("fen")
        if not isinstance(fen, str):
            raise ValueError("fen is required and must be a string.")
        move = args.get("move")
        if move is not None and not isinstance(move, str):
            raise ValueError("move must be a string if provided.")
        return json.dumps(
            _request_state(
                client, "POST", "/api/simulate", json={"fen": fen, "move": move}
            )
        )
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        if "fen" in args and not isinstance(args["fen"], str):
            raise ValueError("fen must be a string.")
        move = args.get("move")
        if not isinstance(move, str):
            raise ValueError("move is required and must be a string.")
        return json.dumps(_request_state(client, "POST", "/api/move", json={"move": move}))
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        code = args.get("code")
        if not isinstance(code, str):
            raise ValueError("code is required and must be a string.")
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        result = env.execute(
            command=["python", "/opt/assignment/sandbox_python.py", str(port), encoded],
            shell=False,
            timeout=100,
        )
        if not isinstance(result, dict):
            raise ValueError("Sandbox result must be an object.")
        if result.get("returncode") != 0:
            raise RuntimeError(
                result.get("exception_info")
                or result.get("stderr")
                or "Sandbox Python execution failed."
            )
        stdout = result.get("stdout")
        if not isinstance(stdout, str):
            raise ValueError("Sandbox stdout must be a string.")
        return stdout
    except Exception as exc:
        if callable(getattr(env, "is_alive", None)) and not env.is_alive():
            raise ChessSandboxUnavailable(
                "Chess sandbox is unavailable; stopping instead of retrying moves. "
                "Check Docker and the sandbox lifetime."
            ) from exc
        return f"<chess_error>{exc}</chess_error>"


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        skill_name = args.get("skill_name")
        if not isinstance(skill_name, str):
            raise ValueError("skill_name is required and must be a string.")
        skill = skills.get(skill_name)
        if skill is None:
            raise ValueError(f"Skill not found: {skill_name}")
        return json.dumps(skill)
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
