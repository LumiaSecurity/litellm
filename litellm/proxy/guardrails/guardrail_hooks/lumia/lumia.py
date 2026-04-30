# +-------------------------------------------------------------+
#
#                       Lumia Security Guardrail
#                      https://www.lumia.security/
#
# +-------------------------------------------------------------+

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )


class LumiaGuardrailMissingSecrets(Exception):
    pass


class LumiaGuardrail(CustomGuardrail):
    """
    Lumia Security guardrail integration for LiteLLM.

    Configure via ``litellm_params`` in ``config.yaml``::

        guardrails:
          - guardrail_name: "lumia-guardrail"
            litellm_params:
              guardrail: lumia
              mode: ["pre_call", "post_call"]
              api_base: https://app.lumiasecurity.com
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
            or "https://app.lumiasecurity.com"
        )
        self.api_base = self.api_base.rstrip("/")

        self.timeout = self._resolve_timeout(timeout)
        self.unreachable_fallback = self._resolve_fallback_action(unreachable_fallback)

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ],
            **kwargs,
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        payload = self._build_payload(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )

        response = await self._call_lumia_api(
            payload, raise_on_error=(input_type == "request")
        )
        if response is None:
            return inputs

        return self._apply_response_to_inputs(inputs=inputs, response=response)

    def _build_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> Dict[str, Any]:
        # Forward LiteLLM's full surfaces verbatim:
        #   - ``request_data``            — per-request dict (model, messages,
        #                                   user_api_key_dict, metadata, IDs);
        #                                   ``safe_dumps`` handles cycles + Pydantic.
        #   - ``standard_logging_object`` — LiteLLM's curated observability
        #                                   payload (response, costs, tokens,
        #                                   timings, applied_guardrails). Already
        #                                   JSON-safe by construction.
        return {
            "input_type": input_type,
            "litellm_version": litellm_version,
            "inputs": inputs,
            "request_data": json.loads(safe_dumps(request_data)),
            "standard_logging_object": (
                logging_obj.standard_logging_object
                if logging_obj and getattr(logging_obj, "standard_logging_object", None)
                else None
            ),
        }

    async def _call_lumia_api(
        self, payload: Dict[str, Any], raise_on_error: bool
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.api_base}{self.GUARDRAIL_PATH}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
        }
        # ``request_data`` is already safe-dumped; ``standard_logging_object``
        # is JSON-safe by construction. ``default=str`` is kept as a hedge in
        # case something inside ``inputs`` ever surfaces a non-JSON-native value.
        body = json.dumps(payload, default=str)

        try:
            response = await self.async_handler.post(
                url=url,
                content=body,
                headers=headers,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            verbose_proxy_logger.warning(
                "Lumia Guardrail: failed to reach %s: %s", url, exc
            )
            if not raise_on_error:
                return None
            return self._handle_unreachable()

        if response.status_code != 200:
            verbose_proxy_logger.warning(
                "Lumia Guardrail: non-200 status %s: %s",
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
                "Lumia Guardrail: failed to parse response: %s", exc
            )
            if not raise_on_error:
                return None
            return self._handle_unreachable()

    def _handle_unreachable(self) -> Optional[Dict[str, Any]]:
        if self.unreachable_fallback == "fail_closed":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Request blocked by Lumia guardrail default fail policy",
                    "guardrail_name": self.guardrail_name,
                    "mode": "fail_closed",
                },
            )
        return None

    def _apply_response_to_inputs(
        self,
        inputs: GenericGuardrailAPIInputs,
        response: Dict[str, Any],
    ) -> GenericGuardrailAPIInputs:
        action = response.get("action", "NONE")

        if action == "BLOCKED":
            blocked_reason = (
                response.get("blocked_reason") or "Request blocked by Lumia policy"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Request blocked by Lumia guardrail",
                    "blocked_reason": blocked_reason,
                    "guardrail_name": self.guardrail_name,
                    "mode": "block",
                },
            )

        texts = response.get("texts")
        if texts is not None:
            inputs["texts"] = texts

        return inputs

    def _resolve_timeout(self, timeout: Optional[float]) -> float:
        if timeout is not None:
            try:
                return float(timeout)
            except (ValueError, TypeError):
                pass
        env_timeout = os.environ.get("LUMIA_GUARDRAIL_TIMEOUT")
        if env_timeout:
            try:
                return float(env_timeout)
            except (ValueError, TypeError):
                pass
        return self.DEFAULT_TIMEOUT

    def _resolve_fallback_action(self, action: Optional[str]) -> str:
        candidate = action or os.environ.get("LUMIA_GUARDRAIL_UNREACHABLE_FALLBACK")
        if candidate and candidate in self.SUPPORTED_FALLBACK_ACTIONS:
            return candidate
        return self.DEFAULT_FALLBACK_ACTION
