"""Input and output safety guardrails backed by Google Cloud Model Armor."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    # Set when Model Armor de-identified sensitive data instead of blocking outright.
    sanitized_text: str | None = None


class NullGuard:
    """Used when Model Armor is disabled (local development)."""

    def check_prompt(self, text: str) -> GuardResult:
        return GuardResult(allowed=True)

    def check_response(self, text: str) -> GuardResult:
        return GuardResult(allowed=True)


def _matched(node) -> bool:
    """True if any nested filter result in this dict reports MATCH_FOUND."""
    if isinstance(node, dict):
        if node.get("match_state") == "MATCH_FOUND":
            return True
        return any(_matched(v) for v in node.values())
    if isinstance(node, list):
        return any(_matched(v) for v in node)
    return False


def _deidentified_text(filter_results: dict) -> str | None:
    sdp = filter_results.get("sdp", {}).get("sdp_filter_result", {})
    text = sdp.get("deidentify_result", {}).get("data", {}).get("text")
    return text or None


class ModelArmorGuard:
    def __init__(self, project: str, location: str, template: str, fail_open: bool = False):
        from google.api_core.client_options import ClientOptions
        from google.cloud import modelarmor_v1

        self._types = modelarmor_v1
        self._client = modelarmor_v1.ModelArmorClient(
            client_options=ClientOptions(api_endpoint=f"modelarmor.{location}.rep.googleapis.com")
        )
        self._template = f"projects/{project}/locations/{location}/templates/{template}"
        self._fail_open = fail_open

    def _evaluate(self, sanitization_result, allow_deidentified: bool) -> GuardResult:
        result = type(sanitization_result).to_dict(
            sanitization_result, use_integers_for_enums=False
        )
        if result.get("filter_match_state") != "MATCH_FOUND":
            return GuardResult(allowed=True)
        filter_results = result.get("filter_results", {})
        reasons = sorted(name for name, value in filter_results.items() if _matched(value))
        if allow_deidentified and reasons == ["sdp"]:
            sanitized = _deidentified_text(filter_results)
            if sanitized:
                return GuardResult(allowed=True, reasons=reasons, sanitized_text=sanitized)
        return GuardResult(allowed=False, reasons=reasons)

    def _on_error(self, exc: Exception) -> GuardResult:
        logger.exception("Model Armor call failed", exc_info=exc)
        if self._fail_open:
            return GuardResult(allowed=True, reasons=["guardrail_unavailable"])
        return GuardResult(allowed=False, reasons=["guardrail_unavailable"])

    def check_prompt(self, text: str) -> GuardResult:
        try:
            response = self._client.sanitize_user_prompt(
                request=self._types.SanitizeUserPromptRequest(
                    name=self._template, user_prompt_data=self._types.DataItem(text=text)
                )
            )
        except Exception as exc:  # noqa: BLE001 - any failure goes through the fail-open/closed policy
            return self._on_error(exc)
        return self._evaluate(response.sanitization_result, allow_deidentified=False)

    def check_response(self, text: str) -> GuardResult:
        try:
            response = self._client.sanitize_model_response(
                request=self._types.SanitizeModelResponseRequest(
                    name=self._template, model_response_data=self._types.DataItem(text=text)
                )
            )
        except Exception as exc:  # noqa: BLE001
            return self._on_error(exc)
        return self._evaluate(response.sanitization_result, allow_deidentified=True)
