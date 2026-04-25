# +-------------------------------------------------------------+
#
#                       Lumia Security Guardrail
#                      https://www.lumia.security/
#
# +-------------------------------------------------------------+

import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import LLMResponseTypes

if TYPE_CHECKING:
    pass


class LumiaGuardrailMissingSecrets(Exception):
    """Raised when the Lumia API token is missing."""

    pass


class LumiaGuardrailAPIError(Exception):
    """Raised when calling the Lumia API fails and fail_closed is configured."""

    pass


class LumiaGuardrail(CustomGuardrail):
    """
    Lumia Security guardrail integration for LiteLLM.

    Forwards each LLM request and response to a Lumia inspection endpoint, where
    the customer's tenant policies determine whether the call should be allowed,
    blocked, or modified. The guardrail preserves the natural LiteLLM payload
    format - messages, tools, and multi-modal content are passed through as-is
    so that Lumia's parsing pipeline receives the same structure regardless of
    the underlying LLM provider.

    Configuration is exposed through ``litellm_params`` in ``config.yaml``::

        guardrails:
          - guardrail_name: "lumia-guardrail"
            litellm_params:
              guardrail: lumia
              mode: ["pre_call", "post_call"]
              api_base: https://guardrails.lumia.security
              api_key: os.environ/LUMIA_GUARDRAIL_API_KEY
              default_on: true
              timeout: 5
              unreachable_fallback: fail_open
    """

    SUPPORTED_FALLBACK_ACTIONS = ["fail_open", "fail_closed"]
    DEFAULT_FALLBACK_ACTION = "fail_open"
    DEFAULT_TIMEOUT = 5.0
    GUARDRAIL_PATH = "/api/v1/litellm_guardrail"

    def __init__(
        self,
        guardrail_name: Optional[str] = "lumia-guardrail",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        unreachable_fallback: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        self.api_key = api_key or os.environ.get("LUMIA_GUARDRAIL_API_KEY")
        if not self.api_key:
            raise LumiaGuardrailMissingSecrets(
                "Lumia guardrail token is missing. Set the LUMIA_GUARDRAIL_API_KEY "
                "environment variable or pass api_key in the guardrail config."
            )

        self.api_base = (
            api_base
            or os.environ.get("LUMIA_GUARDRAIL_API_BASE")
            or "https://guardrails.lumia.security"
        )
        self.api_base = self.api_base.rstrip("/")

        self.timeout = self._resolve_timeout(timeout)
        self.unreachable_fallback = self._resolve_fallback_action(unreachable_fallback)

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

        verbose_proxy_logger.debug(
            "Lumia Guardrail: initialized api_base=%s timeout=%ss fallback=%s",
            self.api_base,
            self.timeout,
            self.unreachable_fallback,
        )

    # -------------------------------------------------------------------------
    # Hook entry points
    # -------------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """Run the Lumia guardrail before the LLM call. May block or modify the request."""
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_call
            )
            is not True
        ):
            return data

        payload = self._build_payload(
            data=data,
            user_api_key_dict=user_api_key_dict,
            input_type="request",
            call_type=str(call_type) if call_type is not None else None,
        )

        lumia_response = await self._call_lumia_api(payload)
        return self._apply_response_to_data(data=data, response=lumia_response)

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """Run the Lumia guardrail after the LLM call. Audit only - never blocks."""
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        try:
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else {}  # type: ignore[union-attr]
            )
        except Exception as exc:  # noqa: BLE001
            verbose_proxy_logger.debug(
                "Lumia Guardrail: failed to dump response for post_call hook: %s", exc
            )
            response_dict = {}

        payload = self._build_payload(
            data=data,
            user_api_key_dict=user_api_key_dict,
            input_type="response",
            call_type=None,
            response=response_dict,
        )

        try:
            await self._call_lumia_api(payload, raise_on_error=False)
        except Exception as exc:  # noqa: BLE001
            # post_call is audit-only, never let it raise into the response path
            verbose_proxy_logger.warning(
                "Lumia Guardrail: post_call audit emission failed: %s", exc
            )

        return response

    # -------------------------------------------------------------------------
    # Payload construction
    # -------------------------------------------------------------------------

    def _build_payload(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        input_type: str,
        call_type: Optional[str],
        response: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Build the JSON payload sent to the Lumia API.

        Uses natural LiteLLM payload shape (``structured_messages``,
        ``litellm_call_id``, ``litellm_trace_id``, ``model``, ``tools``, etc.)
        so the Lumia parser can process traffic from any underlying LLM
        provider routed through LiteLLM uniformly, without per-provider
        translation.
        """
        payload: Dict[str, Any] = {
            "input_type": input_type,
            "call_type": call_type,
            "model": data.get("model"),
            "api_base": data.get("api_base"),
            "structured_messages": data.get("messages"),
            "tools": data.get("tools"),
            "tool_calls": data.get("tool_calls"),
            "identity": self._extract_identity(user_api_key_dict, data),
            "request_headers": self._extract_headers(data),
            "litellm_call_id": data.get("litellm_call_id"),
            "litellm_trace_id": data.get("litellm_trace_id"),
            "litellm_version": litellm_version,
        }

        if response is not None:
            payload["response"] = response

        return payload

    def _extract_identity(
        self, user_api_key_dict: UserAPIKeyAuth, data: dict
    ) -> Dict[str, Optional[str]]:
        """Extract all identity fields available from the LiteLLM hook context."""
        return {
            "user_id": getattr(user_api_key_dict, "user_id", None),
            "user_email": getattr(user_api_key_dict, "user_email", None),
            "team_id": getattr(user_api_key_dict, "team_id", None),
            "team_alias": getattr(user_api_key_dict, "team_alias", None),
            "org_id": getattr(user_api_key_dict, "org_id", None),
            "end_user_id": getattr(user_api_key_dict, "end_user_id", None),
            "key_alias": getattr(user_api_key_dict, "key_alias", None),
            "key_name": getattr(user_api_key_dict, "key_name", None),
            "user_field": data.get("user"),
        }

    def _extract_headers(self, data: dict) -> Dict[str, str]:
        """
        Extract original client headers forwarded by LiteLLM.

        These come from the upstream HTTP request the client made to the LiteLLM
        proxy and are useful for tool identification (User-Agent) and for
        propagating customer-specific identity headers.
        """
        metadata = data.get("metadata") or {}
        headers = metadata.get("headers") or {}

        # LiteLLM also exposes proxy_server_request which contains the raw request
        proxy_request = data.get("proxy_server_request") or {}
        proxy_headers = proxy_request.get("headers") or {}

        merged: Dict[str, str] = {}
        for source in (proxy_headers, headers):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if isinstance(key, str) and isinstance(value, (str, int, float)):
                    merged[key.lower()] = str(value)

        return merged

    # -------------------------------------------------------------------------
    # HTTP call
    # -------------------------------------------------------------------------

    async def _call_lumia_api(
        self, payload: Dict[str, Any], raise_on_error: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        POST the payload to the Lumia guardrail API and return the JSON response.

        On failure, behavior depends on ``unreachable_fallback`` and ``raise_on_error``:
        - ``raise_on_error=True`` and ``fail_closed`` -> raises HTTPException(503)
        - ``raise_on_error=True`` and ``fail_open``   -> returns None (allow request)
        - ``raise_on_error=False`` (post_call)        -> returns None on any error
        """
        url = f"{self.api_base}{self.GUARDRAIL_PATH}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
        }

        try:
            response = await self.async_handler.post(
                url=url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - we want to catch all transport errors
            verbose_proxy_logger.warning(
                "Lumia Guardrail: failed to reach the Lumia API at %s: %s", url, exc
            )
            if not raise_on_error:
                return None
            return self._handle_unreachable()

        if response.status_code != 200:
            verbose_proxy_logger.warning(
                "Lumia Guardrail: Lumia API returned non-200 status %s: %s",
                response.status_code,
                response.text,
            )
            if not raise_on_error:
                return None
            return self._handle_unreachable()

        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            verbose_proxy_logger.warning(
                "Lumia Guardrail: failed to parse Lumia API response as JSON: %s", exc
            )
            if not raise_on_error:
                return None
            return self._handle_unreachable()

    def _handle_unreachable(self) -> Optional[Dict[str, Any]]:
        """Apply the configured fallback when the Lumia API is unreachable."""
        if self.unreachable_fallback == "fail_closed":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Lumia guardrail API is unreachable",
                    "guardrail_name": self.guardrail_name,
                },
            )
        # fail_open: signal to caller to allow the request through unchanged
        return None

    # -------------------------------------------------------------------------
    # Response handling
    # -------------------------------------------------------------------------

    def _apply_response_to_data(
        self, data: dict, response: Optional[Dict[str, Any]]
    ) -> dict:
        """
        Apply Lumia's decision to the request data.

        Returns the (possibly mutated) data dict on allow / intervene.
        Raises HTTPException(400) on block.
        """
        if response is None:
            # fail_open path - allow through unchanged
            return data

        action = response.get("action", "NONE")

        if action == "NONE":
            return data

        if action == "BLOCKED":
            blocked_reason = response.get(
                "blocked_reason", "Request blocked by Lumia policy"
            )
            verbose_proxy_logger.info(
                "Lumia Guardrail: blocking request - %s", blocked_reason
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Request blocked by Lumia guardrail",
                    "blocked_reason": blocked_reason,
                    "guardrail_name": self.guardrail_name,
                },
            )

        if action == "GUARDRAIL_INTERVENED":
            return self._apply_intervention(
                data=data,
                modified_texts=response.get("texts"),
                modified_images=response.get("images"),
            )

        verbose_proxy_logger.warning(
            "Lumia Guardrail: unknown action '%s' in response, treating as NONE", action
        )
        return data

    def _apply_intervention(
        self,
        data: dict,
        modified_texts: Optional[List[str]],
        modified_images: Optional[List[str]],
    ) -> dict:
        """
        Apply modified content from Lumia back into the request messages.

        Replaces text and image content blocks in the messages array in order,
        leaving the original message structure intact for any blocks that were
        not modified.
        """
        if not modified_texts and not modified_images:
            return data

        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        text_iter = iter(modified_texts or [])
        image_iter = iter(modified_images or [])

        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                # Plain string content - replace with next modified text if any
                next_text = next(text_iter, None)
                if next_text is not None:
                    message["content"] = next_text
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        next_text = next(text_iter, None)
                        if next_text is not None:
                            block["text"] = next_text
                    elif block_type == "image_url":
                        next_image = next(image_iter, None)
                        if next_image is not None:
                            image_url = block.get("image_url")
                            if isinstance(image_url, dict):
                                image_url["url"] = next_image
                            else:
                                block["image_url"] = {"url": next_image}

        return data

    # -------------------------------------------------------------------------
    # Configuration helpers
    # -------------------------------------------------------------------------

    def _resolve_timeout(self, timeout: Optional[float]) -> float:
        if timeout is not None:
            try:
                return float(timeout)
            except (ValueError, TypeError):
                verbose_proxy_logger.warning(
                    "Lumia Guardrail: invalid timeout '%s', using default %ss",
                    timeout,
                    self.DEFAULT_TIMEOUT,
                )
        env_timeout = os.environ.get("LUMIA_GUARDRAIL_TIMEOUT")
        if env_timeout:
            try:
                return float(env_timeout)
            except (ValueError, TypeError):
                verbose_proxy_logger.warning(
                    "Lumia Guardrail: invalid LUMIA_GUARDRAIL_TIMEOUT '%s', using default %ss",
                    env_timeout,
                    self.DEFAULT_TIMEOUT,
                )
        return self.DEFAULT_TIMEOUT

    def _resolve_fallback_action(self, action: Optional[str]) -> str:
        candidate = action or os.environ.get("LUMIA_GUARDRAIL_UNREACHABLE_FALLBACK")
        if candidate and candidate in self.SUPPORTED_FALLBACK_ACTIONS:
            return candidate
        if candidate:
            verbose_proxy_logger.warning(
                "Lumia Guardrail: invalid unreachable_fallback '%s', using default '%s'",
                candidate,
                self.DEFAULT_FALLBACK_ACTION,
            )
        return self.DEFAULT_FALLBACK_ACTION
