"""
Lumia Security Guardrail tests for LiteLLM.

The Lumia guardrail now uses LiteLLM's unified ``apply_guardrail`` interface.
LiteLLM's per-endpoint translation handlers normalize provider-specific
bodies into a single ``GenericGuardrailAPIInputs`` shape (texts, images,
tool_calls, structured_messages) before calling our class, and they
re-apply any modifications we return back into the original body. These
tests therefore exercise the apply_guardrail path directly with already-
normalized inputs.
"""

import importlib
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from fastapi.exceptions import HTTPException
from httpx import Request, Response

import litellm
from litellm.proxy.guardrails.guardrail_hooks.lumia import (
    LumiaGuardrail,
    LumiaGuardrailMissingSecrets,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """Reload litellm before every test for clean state."""
    import asyncio

    global litellm
    _module = importlib.import_module("litellm")
    litellm = importlib.reload(_module)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)

    litellm.set_verbose = True
    litellm.guardrail_name_config_map = {}

    yield

    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def env_setup(monkeypatch):
    monkeypatch.setenv("LUMIA_GUARDRAIL_API_KEY", "test-lumia-token")
    monkeypatch.setenv("LUMIA_GUARDRAIL_API_BASE", "https://guardrails.test.lumia")
    yield


@pytest.fixture
def lumia_guardrail(env_setup):
    return LumiaGuardrail(
        guardrail_name="lumia-test",
        api_key="test-lumia-token",
        api_base="https://guardrails.test.lumia",
    )


@pytest.fixture
def lumia_fail_closed(env_setup):
    return LumiaGuardrail(
        guardrail_name="lumia-test-fail-closed",
        api_key="test-lumia-token",
        api_base="https://guardrails.test.lumia",
        unreachable_fallback="fail_closed",
    )


@pytest.fixture
def sample_inputs():
    """Inputs as LiteLLM's translation handler would pass them."""
    return {
        "texts": ["You are a helpful assistant.", "Hello, how are you?"],
        "structured_messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def sample_request_data():
    """request_data dict as the translation handler would pass it."""
    return {
        "model": "gpt-4o-mini",
        "litellm_metadata": {
            "user_api_key_user_id": "jane-user-id",
            "user_api_key_user_email": "jane@company.com",
            "user_api_key_team_id": "team-123",
            "user_api_key_alias": "prod-alias",
        },
        "metadata": {
            "headers": {
                "user-agent": "Cursor/0.50.16",
                "x-litellm-end-user-id": "end-user-from-header",
                "authorization": "Bearer should-be-stripped",
            }
        },
        "proxy_server_request": {
            "headers": {
                "user-agent": "Cursor/0.50.16",
                "x-forwarded-for": "203.0.113.7",
            }
        },
    }


def _http_response(json_body=None, status_code=200):
    return Response(
        json=json_body if json_body is not None else {"action": "NONE"},
        status_code=status_code,
        request=Request(method="POST", url="https://guardrails.test.lumia"),
    )


# =============================================================================
# INITIALIZATION
# =============================================================================


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("LUMIA_GUARDRAIL_API_KEY", raising=False)
    with pytest.raises(LumiaGuardrailMissingSecrets):
        LumiaGuardrail(api_key=None, api_base="https://guardrails.test.lumia")


def test_init_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("LUMIA_GUARDRAIL_API_KEY", "env-token")
    guardrail = LumiaGuardrail(api_key=None, api_base="https://guardrails.test.lumia")
    assert guardrail.api_key == "env-token"


def test_init_strips_trailing_slash_from_api_base(env_setup):
    guardrail = LumiaGuardrail(
        api_key="test-token", api_base="https://guardrails.test.lumia/"
    )
    assert guardrail.api_base == "https://guardrails.test.lumia"


def test_init_default_fallback_is_fail_open(lumia_guardrail):
    assert lumia_guardrail.unreachable_fallback == "fail_open"


def test_init_invalid_fallback_falls_back_to_default(env_setup):
    guardrail = LumiaGuardrail(
        api_key="test-token",
        api_base="https://guardrails.test.lumia",
        unreachable_fallback="not_a_real_value",
    )
    assert guardrail.unreachable_fallback == "fail_open"


def test_class_overrides_apply_guardrail():
    """LiteLLM auto-routes to the unified path when ``apply_guardrail`` is
    defined directly on the subclass. Make sure we keep that property —
    if this assertion fails, traffic silently falls back to the legacy
    pre_call path."""
    assert "apply_guardrail" in LumiaGuardrail.__dict__


# =============================================================================
# IDENTITY EXTRACTION
# =============================================================================


def test_slice_request_metadata_extracts_identity_and_headers(
    lumia_guardrail, sample_request_data
):
    """Slice extracts ONLY user_api_key_* identity fields and headers.
    Everything else is intentionally dropped to avoid cycles in the
    accumulated post-call hook state."""
    sliced = lumia_guardrail._slice_request_metadata(sample_request_data)
    assert sliced["request_data"]["user_api_key_user_email"] == "jane@company.com"
    assert sliced["request_data"]["user_api_key_user_id"] == "jane-user-id"
    assert sliced["request_headers"]["user-agent"] == "Cursor/0.50.16"
    assert sliced["request_headers"]["x-forwarded-for"] == "203.0.113.7"
    # ``metadata`` / ``litellm_metadata`` / ``proxy_server_request`` are
    # NOT forwarded as nested dicts — only the flat strings we care about.
    assert "metadata" not in sliced
    assert "litellm_metadata" not in sliced
    assert "proxy_server_request" not in sliced


def test_slice_request_metadata_survives_circular_references(lumia_guardrail):
    """LiteLLM's accumulated post-call state may contain cycles
    (``standard_logging_object`` → ``metadata`` → … → itself). The slice
    must still produce a JSON-serializable payload by only copying flat
    strings."""
    import json as _json

    cyclic_metadata: dict = {
        "user_api_key_user_email": "cycle@test",
        "user_api_key_user_id": "u-1",
    }
    cyclic_metadata["standard_logging_object"] = {"metadata": cyclic_metadata}

    request_data = {"metadata": cyclic_metadata}
    sliced = lumia_guardrail._slice_request_metadata(request_data)
    # Must be JSON-serializable — the bug we're guarding against
    _json.dumps(sliced)
    assert sliced["request_data"]["user_api_key_user_email"] == "cycle@test"


def test_slice_request_metadata_drops_non_serializable_keys(lumia_guardrail):
    """``litellm_logging_obj`` and other non-serializable fields are
    intentionally not forwarded — would break json.dumps."""

    class _Unserializable:
        pass

    request_data = {
        "metadata": {"user_api_key_user_id": "u-1"},
        "litellm_logging_obj": _Unserializable(),
        "litellm_params": _Unserializable(),
        "secret_fields": ["api_key"],
    }
    sliced = lumia_guardrail._slice_request_metadata(request_data)
    assert sliced == {"request_data": {"user_api_key_user_id": "u-1"}}


def test_slice_request_metadata_with_no_context(lumia_guardrail):
    assert lumia_guardrail._slice_request_metadata({}) == {}


# =============================================================================
# APPLY GUARDRAIL — REQUEST PATH
# =============================================================================


@pytest.mark.asyncio
async def test_apply_guardrail_allow(
    lumia_guardrail, sample_inputs, sample_request_data
):
    """action=NONE leaves inputs unchanged."""
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=_http_response({"action": "NONE"})),
    ):
        result = await lumia_guardrail.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )
    assert result["texts"] == sample_inputs["texts"]


@pytest.mark.asyncio
async def test_apply_guardrail_blocked(
    lumia_guardrail, sample_inputs, sample_request_data
):
    """action=BLOCKED raises HTTPException(451) with the blocked_reason."""
    blocked = _http_response(
        {"action": "BLOCKED", "blocked_reason": "Policy violation"}
    )
    with patch.object(
        lumia_guardrail.async_handler, "post", new=AsyncMock(return_value=blocked)
    ):
        with pytest.raises(HTTPException) as exc_info:
            await lumia_guardrail.apply_guardrail(
                inputs=sample_inputs,
                request_data=sample_request_data,
                input_type="request",
            )
    assert exc_info.value.status_code == 451
    assert "Policy violation" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_apply_guardrail_intervened_replaces_texts(
    lumia_guardrail, sample_inputs, sample_request_data
):
    """Modified texts in the response replace inputs['texts'] in order.
    LiteLLM's translation handler then maps these back into the original
    provider-shape body — no splice work on our side."""
    intervened = _http_response(
        {
            "action": "GUARDRAIL_INTERVENED",
            "texts": ["[redacted system prompt]", "[redacted user message]"],
        }
    )
    with patch.object(
        lumia_guardrail.async_handler, "post", new=AsyncMock(return_value=intervened)
    ):
        result = await lumia_guardrail.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )
    assert result["texts"] == [
        "[redacted system prompt]",
        "[redacted user message]",
    ]


@pytest.mark.asyncio
async def test_apply_guardrail_intervention_does_not_redact_images(
    lumia_guardrail, sample_request_data
):
    """Only text content is redactable. Images in inputs stay as-is even
    if the API response includes an ``images`` field."""
    original_images = ["data:image/png;base64,iVBORw0KGgoAAAA=="]
    inputs = {
        "texts": ["What's in this image?"],
        "images": list(original_images),
        "structured_messages": [],
        "model": "gpt-4o",
    }
    intervened = _http_response(
        {
            "action": "GUARDRAIL_INTERVENED",
            "texts": ["[redacted question]"],
            "images": ["data:image/png;base64,REDACTED"],
        }
    )
    with patch.object(
        lumia_guardrail.async_handler, "post", new=AsyncMock(return_value=intervened)
    ):
        result = await lumia_guardrail.apply_guardrail(
            inputs=inputs,
            request_data=sample_request_data,
            input_type="request",
        )
    assert result["texts"] == ["[redacted question]"]
    assert result["images"] == original_images


@pytest.mark.asyncio
async def test_apply_guardrail_fail_open_on_unreachable(
    lumia_guardrail, sample_inputs, sample_request_data
):
    """Default fail_open: connection errors leave inputs unchanged."""
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(side_effect=ConnectionError("connection refused")),
    ):
        result = await lumia_guardrail.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )
    assert result == sample_inputs


@pytest.mark.asyncio
async def test_apply_guardrail_fail_closed_on_unreachable(
    lumia_fail_closed, sample_inputs, sample_request_data
):
    """fail_closed: connection errors raise HTTPException(451)."""
    with patch.object(
        lumia_fail_closed.async_handler,
        "post",
        new=AsyncMock(side_effect=ConnectionError("connection refused")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await lumia_fail_closed.apply_guardrail(
                inputs=sample_inputs,
                request_data=sample_request_data,
                input_type="request",
            )
    assert exc_info.value.status_code == 451
    assert "fail policy" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_apply_guardrail_non_200_response(
    lumia_guardrail, sample_inputs, sample_request_data
):
    """Non-200 response from Lumia falls back to fail_open default."""
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=_http_response(status_code=500)),
    ):
        result = await lumia_guardrail.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )
    assert result == sample_inputs


# =============================================================================
# APPLY GUARDRAIL — RESPONSE PATH
# =============================================================================


@pytest.mark.asyncio
async def test_apply_guardrail_response_swallows_errors(
    lumia_guardrail, sample_request_data
):
    """post-call (input_type='response') must not raise on transport
    errors — Lumia is audit-only on the response side."""
    inputs = {"texts": ["assistant text"], "model": "gpt-4o-mini"}
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(side_effect=ConnectionError("boom")),
    ):
        result = await lumia_guardrail.apply_guardrail(
            inputs=inputs,
            request_data=sample_request_data,
            input_type="response",
        )
    assert result == inputs


# =============================================================================
# PAYLOAD SHAPE
# =============================================================================


@pytest.mark.asyncio
async def test_payload_is_thin_envelope(
    lumia_guardrail, sample_inputs, sample_request_data
):
    """On-the-wire payload is a thin envelope — inputs + JSON-safe slice
    of request_data, no filtering. Lumen owns identity / header /
    traffic-event work."""
    captured = {}

    async def fake_post(url, content, headers, timeout):
        import json as _json

        captured["url"] = url
        captured["payload"] = _json.loads(content)
        captured["headers"] = headers
        return _http_response({"action": "NONE"})

    with patch.object(lumia_guardrail.async_handler, "post", new=fake_post):
        await lumia_guardrail.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )

    payload = captured["payload"]

    # Top-level envelope fields
    assert payload["input_type"] == "request"
    assert payload["litellm_version"]

    # Inputs forwarded as-is
    assert payload["inputs"] == sample_inputs

    # request_metadata block has flattened identity + headers.
    # proxy_server_request.headers wins over metadata.headers (it's the
    # actual inbound HTTP envelope, more authoritative).
    rm = payload["request_metadata"]
    assert rm["request_data"]["user_api_key_user_email"] == "jane@company.com"
    assert rm["request_headers"]["user-agent"] == "Cursor/0.50.16"
    assert rm["request_headers"]["x-forwarded-for"] == "203.0.113.7"

    # Auth is via x-api-key
    assert captured["headers"]["x-api-key"] == "test-lumia-token"


# =============================================================================
# CONFIGURATION REGISTRATION
# =============================================================================


def test_init_guardrails_v2_registers_lumia(env_setup):
    config = [
        {
            "guardrail_name": "lumia-from-config",
            "litellm_params": {
                "guardrail": "lumia",
                "mode": "pre_call",
                "default_on": True,
                "api_key": "test-lumia-token",
                "api_base": "https://guardrails.test.lumia",
            },
        }
    ]

    init_guardrails_v2(
        all_guardrails=config,
        config_file_path="",
    )

    callback_classes = [type(cb).__name__ for cb in litellm.callbacks]
    assert "LumiaGuardrail" in callback_classes
