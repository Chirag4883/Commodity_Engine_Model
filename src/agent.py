from __future__ import annotations

"""
Level 2 — Structured Quantamental Feature Agent

The agent only accepts an AnonymizedDocument. Raw source text is deliberately
not accepted by the inference API.

The LLM/provider boundary is represented by StructuredLLMClient so:
- CI can be deterministic,
- provider credentials are not required for unit tests,
- model providers can be changed without changing quantitative semantics.

The agent requests three model-derived features:

    E_sub       ∈ [-1, +1]
    P_pricing   ∈ [-1, +1]
    T_policy    ∈ [ 0, +1]

The deterministic raw bottleneck score is computed locally:

    S_i = T_policy + (P_pricing * E_sub)

The LLM is NOT allowed to supply S_i itself.
"""

from collections.abc import Mapping
import json
import re
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
)

from src.anonymizer import (
    AnonymizedDocument,
    LeakageDetectedError,
    assert_no_sensitive_literals,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for feature-agent failures."""


class LLMResponseValidationError(AgentError):
    """Raised when the model response cannot pass strict schema validation."""


class AgentOutputLeakageError(AgentError):
    """Raised when model output reintroduces masked sensitive information."""


class CandidateMismatchError(AgentError):
    """Raised when the model responds for a different candidate node."""


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class CandidateNode(BaseModel):
    """
    Anonymous graph-node descriptor supplied to the agent.

    The identifier must itself already be anonymous for any Layer-4 equity.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    anonymized_id: str = Field(
        min_length=1,
        max_length=128,
    )

    layer: int = Field(
        ge=0,
        le=4,
    )

    physical_role: str = Field(
        min_length=1,
        max_length=1000,
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """
    Short auditable observation rather than free-form chain-of-thought.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    observation: str = Field(
        min_length=1,
        max_length=500,
    )

    effect: Literal[
        "supports_risk",
        "offsets_risk",
        "supports_pricing_power",
        "weakens_pricing_power",
        "supports_protection",
        "no_policy_support",
        "uncertain",
    ]


class FeatureAssessment(BaseModel):
    """
    Strict validated feature vector.

    JSON aliases intentionally match the project's mathematical notation.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    candidate_node_alias: str = Field(
        min_length=1,
        max_length=128,
    )

    e_sub: float = Field(
        alias="E_sub",
        ge=-1.0,
        le=1.0,
        description=(
            "Input Elasticity Risk. "
            "-1 = extreme squeeze vulnerability; "
            "+1 = fully captive/secure input."
        ),
    )

    p_pricing: float = Field(
        alias="P_pricing",
        ge=-1.0,
        le=1.0,
        description=(
            "Downstream Pricing Power. "
            "-1 = pure price taker; "
            "+1 = monopolistic pricing power."
        ),
    )

    t_policy: float = Field(
        alias="T_policy",
        ge=0.0,
        le=1.0,
        description=(
            "Domestic Protection Delta. "
            "0 = unprotected; "
            "+1 = maximal import restriction."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: tuple[EvidenceItem, ...] = Field(
        min_length=1,
        max_length=6,
    )

    @computed_field(
        alias="S_raw",
        return_type=float,
    )
    @property
    def raw_bottleneck_score(self) -> float:
        """
        Project-specified raw bottleneck score.

            S_i = T_policy + (P_pricing * E_sub)
        """
        return self.t_policy + (
            self.p_pricing * self.e_sub
        )


# ---------------------------------------------------------------------------
# Provider-neutral LLM boundary
# ---------------------------------------------------------------------------


@runtime_checkable
class StructuredLLMClient(Protocol):
    """
    Minimal provider contract.

    Implementations may use any model/vendor capable of returning JSON.
    """

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[str, Any],
    ) -> str | Mapping[str, Any]:
        """
        Return either:
        - a JSON string, or
        - an already-decoded mapping.
        """
        ...


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class QuantamentalFeatureAgent:
    """
    Extracts Level-2 physical/quantamental features from PIT-safe documents.
    """

    def __init__(
        self,
        client: StructuredLLMClient,
    ) -> None:
        if not isinstance(client, StructuredLLMClient):
            raise TypeError(
                "client must satisfy StructuredLLMClient"
            )

        self._client = client

    def evaluate(
        self,
        *,
        document: AnonymizedDocument,
        candidate: CandidateNode,
    ) -> FeatureAssessment:
        """
        Execute one anonymized feature-extraction evaluation.
        """
        if not isinstance(document, AnonymizedDocument):
            raise TypeError(
                "document must be an AnonymizedDocument; "
                "raw strings are not accepted"
            )

        document.assert_safe()

        self._validate_candidate_alias(candidate)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            document=document,
            candidate=candidate,
        )

        # Defense-in-depth: confirm the constructed prompt itself contains
        # neither original sensitive literals nor explicit years.
        assert_no_sensitive_literals(
            user_prompt,
            document.prohibited_literals(),
        )

        raw_response = self._client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=FeatureAssessment.model_json_schema(
                by_alias=True,
            ),
        )

        assessment = self._parse_response(raw_response)

        if (
            assessment.candidate_node_alias
            != candidate.anonymized_id
        ):
            raise CandidateMismatchError(
                "Model response candidate_node_alias does not match "
                f"requested candidate: "
                f"{assessment.candidate_node_alias!r} != "
                f"{candidate.anonymized_id!r}"
            )

        # An anonymized model can still try to recall a real company/event.
        # Treat any re-identification as a hard pipeline failure.
        serialized_output = json.dumps(
            assessment.model_dump(
                by_alias=True,
                mode="json",
            ),
            ensure_ascii=False,
            sort_keys=True,
        )

        try:
            assert_no_sensitive_literals(
                serialized_output,
                document.prohibited_literals(),
            )
        except LeakageDetectedError as exc:
            raise AgentOutputLeakageError(
                "LLM output reintroduced masked source information"
            ) from exc

        return assessment

    @staticmethod
    def _validate_candidate_alias(
        candidate: CandidateNode,
    ) -> None:
        """
        Prevent obvious ticker/year leakage through candidate identifiers.
        """
        alias = candidate.anonymized_id

        if re.search(
            r"\.(?:NS|BO)\b",
            alias,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "candidate.anonymized_id appears to contain "
                "an NSE/BSE ticker"
            )

        if re.search(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            alias,
        ):
            raise ValueError(
                "candidate.anonymized_id contains an explicit calendar year"
            )

    @staticmethod
    def _build_system_prompt() -> str:
        return """
You are a point-in-time physical supply-chain feature extractor.

You are not selecting stocks and you are not forecasting returns.

Evaluate only the anonymous candidate node using only the supplied anonymous
source text and physical-role description.

Return a strict JSON object matching the provided schema.

Feature semantics:

1. E_sub: Input Elasticity Risk, from -1.0 to +1.0.
   -1.0 means extreme vulnerability to input shortage or an effectively
   non-substitutable external dependency.
   +1.0 means highly secure, captive, vertically integrated, or readily
   substitutable supply.
   Use values near 0 when evidence is weak or mixed.

2. P_pricing: Downstream Pricing Power, from -1.0 to +1.0.
   -1.0 means pure price taker with little ability to pass through costs.
   +1.0 means exceptionally strong price-setting ability.
   Use values near 0 when evidence is weak or mixed.

3. T_policy: Domestic Protection Delta, from 0.0 to +1.0.
   0.0 means no demonstrated domestic trade/regulatory protection.
   +1.0 represents the strongest practical import restriction.
   Anti-dumping duties, countervailing duties, quality-control orders,
   tariffs, quotas, or import restrictions can support positive values.
   Do not infer protection when the source text does not support it.

Physical constraints:
- Prefer direct evidence about feedstock concentration, qualification times,
  import dependence, captive production, switching ability, lead times,
  pass-through mechanisms, contract structures, domestic capacity and
  regulatory restrictions.
- Do not use memorized real-world company identities.
- Do not attempt to reverse anonymous identifiers.
- Do not introduce real company names, tickers, executives, calendar years,
  or famous event names.
- Do not fabricate missing facts.
- When evidence is insufficient, move scores toward neutral values.
- Evidence observations must be concise and auditable.
- Do not provide hidden chain-of-thought.
- Do not return S_raw. It is calculated deterministically downstream.
""".strip()

    @staticmethod
    def _build_user_prompt(
        *,
        document: AnonymizedDocument,
        candidate: CandidateNode,
    ) -> str:
        payload = {
            "candidate": {
                "anonymized_id": candidate.anonymized_id,
                "layer": candidate.layer,
                "physical_role": candidate.physical_role,
            },
            "anonymized_source_text": document.prompt_text,
            "instruction": (
                "Return the three required numerical features, confidence, "
                "and concise structured evidence."
            ),
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _parse_response(
        raw_response: str | Mapping[str, Any],
    ) -> FeatureAssessment:
        """
        Strictly decode and Pydantic-validate provider output.
        """
        if isinstance(raw_response, Mapping):
            payload: Any = dict(raw_response)

        elif isinstance(raw_response, str):
            cleaned = _strip_markdown_fence(
                raw_response.strip()
            )

            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise LLMResponseValidationError(
                    "LLM response is not valid JSON"
                ) from exc

        else:
            raise LLMResponseValidationError(
                "LLM response must be JSON text or a mapping"
            )

        if not isinstance(payload, dict):
            raise LLMResponseValidationError(
                "LLM response root must be a JSON object"
            )

        try:
            return FeatureAssessment.model_validate(payload)
        except ValidationError as exc:
            raise LLMResponseValidationError(
                "LLM response failed FeatureAssessment validation"
            ) from exc


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _strip_markdown_fence(value: str) -> str:
    """
    Tolerate a provider wrapping otherwise-valid JSON in ```json fences.
    """
    if not value.startswith("```"):
        return value

    value = re.sub(
        r"^```(?:json)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s*```$",
        "",
        value,
    )

    return value.strip()
