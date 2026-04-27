"""
Lumia Security Guardrail tests for LiteLLM.

Tests cover initialization, hook execution, response mapping, identity
extraction, header forwarding, and unreachable-fallback behavior. Follows
LiteLLM testing patterns and uses mocked HTTP responses (no real network calls).
"""

import importlib
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from fastapi.exceptions import HTTPException
from httpx import Request, Response

import litellm
from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
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
    """Set environment variables for Lumia tests."""
    monkeypatch.setenv("LUMIA_GUARDRAIL_API_KEY", "test-lumia-token")
    monkeypatch.setenv("LUMIA_GUARDRAIL_API_BASE", "https://guardrails.test.lumia")
    yield


@pytest.fixture
def lumia_guardrail(env_setup):
    """LumiaGuardrail instance with default config."""
    return LumiaGuardrail(
        guardrail_name="lumia-test",
        api_key="test-lumia-token",
        api_base="https://guardrails.test.lumia",
    )


@pytest.fixture
def lumia_fail_closed(env_setup):
    """LumiaGuardrail instance configured to fail closed on errors."""
    return LumiaGuardrail(
        guardrail_name="lumia-test-fail-closed",
        api_key="test-lumia-token",
        api_base="https://guardrails.test.lumia",
        unreachable_fallback="fail_closed",
    )


@pytest.fixture
def user_api_key_dict():
    """Empty UserAPIKeyAuth instance."""
    return UserAPIKeyAuth()


@pytest.fixture
def user_api_key_dict_full():
    """UserAPIKeyAuth instance populated with identity fields."""
    return UserAPIKeyAuth(
        token="hashed-token",
        key_name="prod-key",
        key_alias="prod-alias",
        user_id="jane-user-id",
        user_email="jane@company.com",
        team_id="team-123",
        team_alias="engineering",
        org_id="org-456",
        end_user_id="end-user-789",
    )


@pytest.fixture
def dual_cache():
    """DualCache instance."""
    return DualCache()


@pytest.fixture
def sample_request_data():
    """Sample chat completion request data."""
    return {
        "model": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "user": "user-from-data",
        "litellm_call_id": "call-abc-123",
        "litellm_trace_id": "trace-xyz-789",
        "metadata": {
            "headers": {
                "user-agent": "Cursor/0.50.16",
                "x-litellm-end-user-id": "end-user-from-header",
            }
        },
    }


@pytest.fixture
def multimodal_request_data():
    """Request data with multi-modal content (image, file)."""
    return {
        "model": "gpt-4o",
        "api_base": "https://api.openai.com/v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAA=="},
                    },
                ],
            }
        ],
    }


def _http_response(json_body=None, status_code=200):
    """Helper to build an httpx.Response object for mocking."""
    return Response(
        json=json_body if json_body is not None else {"action": "NONE"},
        status_code=status_code,
        request=Request(method="POST", url="https://guardrails.test.lumia"),
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_init_requires_api_key(monkeypatch):
    """Constructor must raise when no API key is supplied."""
    monkeypatch.delenv("LUMIA_GUARDRAIL_API_KEY", raising=False)
    with pytest.raises(LumiaGuardrailMissingSecrets):
        LumiaGuardrail(api_key=None, api_base="https://guardrails.test.lumia")


def test_init_reads_api_key_from_env(monkeypatch):
    """Constructor should pick up the API key from the environment."""
    monkeypatch.setenv("LUMIA_GUARDRAIL_API_KEY", "env-token")
    guardrail = LumiaGuardrail(api_key=None, api_base="https://guardrails.test.lumia")
    assert guardrail.api_key == "env-token"


def test_init_strips_trailing_slash_from_api_base(env_setup):
    """The api_base attribute should not contain a trailing slash."""
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


# =============================================================================
# IDENTITY EXTRACTION TESTS
# =============================================================================


def test_extract_request_data_with_full_context(
    lumia_guardrail, user_api_key_dict_full, sample_request_data
):
    request_data = lumia_guardrail._extract_request_data(
        user_api_key_dict_full, sample_request_data
    )
    assert request_data["user_api_key_user_id"] == "jane-user-id"
    assert request_data["user_api_key_user_email"] == "jane@company.com"
    assert request_data["user_api_key_team_id"] == "team-123"
    assert request_data["user_api_key_team_alias"] == "engineering"
    assert request_data["user_api_key_org_id"] == "org-456"
    assert request_data["user_api_key_end_user_id"] == "end-user-789"
    assert request_data["user_api_key_alias"] == "prod-alias"
    assert request_data["user"] == "user-from-data"


def test_extract_request_data_with_empty_context(
    lumia_guardrail, user_api_key_dict, sample_request_data
):
    request_data = lumia_guardrail._extract_request_data(
        user_api_key_dict, sample_request_data
    )
    # Fields should exist but be None when user_api_key_dict has no values set
    assert request_data["user_api_key_user_id"] is None
    assert request_data["user"] == "user-from-data"


# =============================================================================
# HEADER EXTRACTION TESTS
# =============================================================================


def test_extract_headers_from_metadata(lumia_guardrail, sample_request_data):
    headers = lumia_guardrail._extract_headers(sample_request_data)
    assert headers["user-agent"] == "Cursor/0.50.16"
    assert headers["x-litellm-end-user-id"] == "end-user-from-header"


def test_extract_headers_with_no_metadata(lumia_guardrail):
    headers = lumia_guardrail._extract_headers({"model": "gpt-4o"})
    assert headers == {}


# =============================================================================
# PRE-CALL HOOK TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_pre_call_hook_allow(
    lumia_guardrail, user_api_key_dict, dual_cache, sample_request_data
):
    """When Lumia returns NONE, the request should pass through unchanged."""
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=_http_response({"action": "NONE"})),
    ):
        result = await lumia_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=sample_request_data,
            call_type="completion",
        )
    assert result == sample_request_data


@pytest.mark.asyncio
async def test_pre_call_hook_blocked(
    lumia_guardrail, user_api_key_dict, dual_cache, sample_request_data
):
    """When Lumia returns BLOCKED, an HTTPException should be raised."""
    blocked_response = _http_response(
        {"action": "BLOCKED", "blocked_reason": "Policy violation"}
    )
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=blocked_response),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await lumia_guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=dual_cache,
                data=sample_request_data,
                call_type="completion",
            )
    assert exc_info.value.status_code == 400
    assert "Policy violation" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_pre_call_hook_intervened_replaces_text(
    lumia_guardrail, user_api_key_dict, dual_cache, sample_request_data
):
    """GUARDRAIL_INTERVENED with texts should replace text content blocks."""
    intervened = _http_response(
        {
            "action": "GUARDRAIL_INTERVENED",
            "texts": [
                "[redacted system prompt]",
                "[redacted user message]",
            ],
        }
    )
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=intervened),
    ):
        result = await lumia_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=sample_request_data,
            call_type="completion",
        )
    assert result["messages"][0]["content"] == "[redacted system prompt]"
    assert result["messages"][1]["content"] == "[redacted user message]"


@pytest.mark.asyncio
async def test_pre_call_hook_intervened_preserves_multimodal(
    lumia_guardrail, user_api_key_dict, dual_cache, multimodal_request_data
):
    """Multi-modal messages keep their structure when intervened on."""
    intervened = _http_response(
        {
            "action": "GUARDRAIL_INTERVENED",
            "texts": ["[redacted question]"],
            "images": ["data:image/png;base64,REDACTED"],
        }
    )
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=intervened),
    ):
        result = await lumia_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=multimodal_request_data,
            call_type="completion",
        )
    content = result["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "[redacted question]"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,REDACTED"


@pytest.mark.asyncio
async def test_pre_call_hook_fail_open_on_unreachable(
    lumia_guardrail, user_api_key_dict, dual_cache, sample_request_data
):
    """Default fail_open: connection errors should let the request through."""
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(side_effect=ConnectionError("connection refused")),
    ):
        result = await lumia_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=sample_request_data,
            call_type="completion",
        )
    assert result == sample_request_data


@pytest.mark.asyncio
async def test_pre_call_hook_fail_closed_on_unreachable(
    lumia_fail_closed, user_api_key_dict, dual_cache, sample_request_data
):
    """fail_closed: connection errors must raise HTTPException(503)."""
    with patch.object(
        lumia_fail_closed.async_handler,
        "post",
        new=AsyncMock(side_effect=ConnectionError("connection refused")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await lumia_fail_closed.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=dual_cache,
                data=sample_request_data,
                call_type="completion",
            )
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_pre_call_hook_non_200_response(
    lumia_guardrail, user_api_key_dict, dual_cache, sample_request_data
):
    """Non-200 response from Lumia should fail open (default)."""
    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=_http_response(status_code=500)),
    ):
        result = await lumia_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=sample_request_data,
            call_type="completion",
        )
    assert result == sample_request_data


# =============================================================================
# POST-CALL HOOK TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_post_call_hook_returns_response_unchanged(
    lumia_guardrail, user_api_key_dict, sample_request_data
):
    """post_call always returns the LLM response unchanged."""
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hello back!"}}]
    }

    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(return_value=_http_response({"action": "NONE"})),
    ):
        result = await lumia_guardrail.async_post_call_success_hook(
            data=sample_request_data,
            user_api_key_dict=user_api_key_dict,
            response=mock_response,
        )
    assert result is mock_response


@pytest.mark.asyncio
async def test_post_call_hook_swallows_errors(
    lumia_guardrail, user_api_key_dict, sample_request_data
):
    """post_call must never raise - errors are logged only."""
    mock_response = Mock()
    mock_response.model_dump.return_value = {"choices": []}

    with patch.object(
        lumia_guardrail.async_handler,
        "post",
        new=AsyncMock(side_effect=ConnectionError("boom")),
    ):
        # Should not raise even if Lumia is unreachable
        result = await lumia_guardrail.async_post_call_success_hook(
            data=sample_request_data,
            user_api_key_dict=user_api_key_dict,
            response=mock_response,
        )
    assert result is mock_response


# =============================================================================
# PAYLOAD CONSTRUCTION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_payload_includes_all_expected_fields(
    lumia_guardrail, user_api_key_dict_full, dual_cache, sample_request_data
):
    """Verify the payload sent to Lumia contains all expected sections."""
    captured = {}

    async def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _http_response({"action": "NONE"})

    with patch.object(lumia_guardrail.async_handler, "post", new=fake_post):
        await lumia_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict_full,
            cache=dual_cache,
            data=sample_request_data,
            call_type="completion",
        )

    payload = captured["json"]

    # Body fields preserved at the top level (matches what per-vendor
    # protocol definitions on the receiving side parse natively).
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == sample_request_data["messages"]
    assert payload["api_base"] == "https://api.openai.com/v1"

    # LiteLLM-internal/bookkeeping fields are stripped from the body.
    assert "litellm_call_id" not in payload
    assert "litellm_trace_id" not in payload
    assert "metadata" not in payload

    # Lumia-specific metadata grouped under the _litellm envelope.
    envelope = payload["_litellm"]
    assert envelope["input_type"] == "request"
    assert envelope["call_type"] == "completion"
    assert envelope["api_base"] == "https://api.openai.com/v1"
    assert envelope["request_data"]["user_api_key_user_id"] == "jane-user-id"
    assert envelope["request_data"]["user_api_key_end_user_id"] == "end-user-789"
    assert envelope["request_headers"]["user-agent"] == "Cursor/0.50.16"
    assert envelope["litellm_call_id"] == "call-abc-123"
    assert envelope["litellm_trace_id"] == "trace-xyz-789"

    assert captured["headers"]["x-api-key"] == "test-lumia-token"


# =============================================================================
# CONFIGURATION REGISTRATION TESTS
# =============================================================================


def test_init_guardrails_v2_registers_lumia(env_setup):
    """Verify init_guardrails_v2 wires the Lumia guardrail correctly."""
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

    # The callback should be registered on litellm.callbacks
    callback_classes = [type(cb).__name__ for cb in litellm.callbacks]
    assert "LumiaGuardrail" in callback_classes
