"""
Lumia Security Guardrail integration for LiteLLM.
"""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .lumia import (
    LumiaGuardrail,
    LumiaGuardrailMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


_DEFAULT_MODE = ["pre_call", "post_call"]


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Lumia guardrail name is required")

    mode = litellm_params.mode or _DEFAULT_MODE

    _lumia_callback = LumiaGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        timeout=getattr(litellm_params, "timeout", None),
        unreachable_fallback=getattr(litellm_params, "unreachable_fallback", None),
        event_hook=mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_lumia_callback)
    return _lumia_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.LUMIA.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.LUMIA.value: LumiaGuardrail,
}


__all__ = [
    "LumiaGuardrail",
    "LumiaGuardrailMissingSecrets",
]
