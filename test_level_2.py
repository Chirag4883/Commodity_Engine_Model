import json
import re
from typing import Any, Mapping

import pytest

from src.agent import (
    AgentOutputLeakageError,
    CandidateMismatchError,
    CandidateNode,
    LLMResponseValidationError,
    QuantamentalFeatureAgent,
)
from src.anonymizer import (
    LeakageDetectedError,
    PITAnonymizer,
    assert_no_calendar_years,
)


SAMPLE_TEXT = """
In 2020, Borosil Renewables stated that imported solar glass remained an
important competitive factor. BORORENEW.NS described domestic manufacturing
capacity expansion and exposure to imported inputs. Executive Rohan Mehta
discussed longer qualification cycles during COVID-19.

By 2022, the company noted that trade measures and product qualification
requirements could alter competitive conditions. The source also stated that
some raw-material costs could not immediately be passed through to customers.
""".strip()


class FakeStructuredLLM:
    """
    Deterministic isolated provider used by CI.

    Captures the actual prompts so tests can assert that sensitive information
    never crosses the provider boundary.
    """

    def __init__(
        self,
        response: str | Mapping[str, Any],
    ) -> None:
        self.response = response

        self.last_system_prompt = None
        self.last_user_prompt = None
        self.last_json_schema = None

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[str, Any],
    ) -> str | Mapping[str, Any]:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_json_schema = json_schema

        return self.response


@pytest.fixture
def anonymizer():
    return PITAnonymizer(
        executive_names=[
            "Rohan Mehta",
        ]
    )


@pytest.fixture
def document(anonymizer):
    return anonymizer.anonymize(SAMPLE_TEXT)


@pytest.fixture
def candidate():
    return CandidateNode(
        anonymized_id="Supplier_K",
        layer=4,
        physical_role=(
            "Anonymous domestic solar-glass manufacturing exposure "
            "linked to silica/soda-ash inputs and downstream PV demand."
        ),
    )


def valid_response():
    return {
        "candidate_node_alias": "Supplier_K",
        "E_sub": -0.80,
        "P_pricing": 0.50,
        "T_policy": 0.40,
        "confidence": 0.82,
        "evidence": [
            {
                "observation": (
                    "The source describes imported-input exposure and "
                    "long qualification cycles."
                ),
                "effect": "supports_risk",
            },
            {
                "observation": (
                    "Trade measures are described as changing domestic "
                    "competitive conditions."
                ),
                "effect": "supports_protection",
            },
            {
                "observation": (
                    "Some input costs cannot immediately be passed "
                    "through downstream."
                ),
                "effect": "weakens_pricing_power",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Required anonymization gates
# ---------------------------------------------------------------------------


def test_anonymizer_removes_company_name(document):
    folded = document.masked_text.casefold()

    assert "borosil renewables" not in folded
    assert "borosil" not in folded

    assert "Supplier_K" in document.masked_text


def test_anonymizer_removes_ticker(document):
    assert "BORORENEW.NS" not in document.masked_text
    assert "Equity_C" in document.masked_text


def test_anonymizer_removes_executive_name(document):
    assert "Rohan Mehta" not in document.masked_text
    assert "Executive_A" in document.masked_text


def test_anonymizer_removes_historical_event_identifier(document):
    assert "COVID-19" not in document.masked_text
    assert "Macro_Disruption_Alpha" in document.masked_text


def test_anonymizer_removes_all_explicit_calendar_years(document):
    assert "2020" not in document.masked_text
    assert "2022" not in document.masked_text

    assert_no_calendar_years(
        document.masked_text
    )

    assert re.search(
        r"(?<!\d)(?:19|20)\d{2}(?!\d)",
        document.masked_text,
    ) is None


def test_year_mapping_is_deterministic_and_preserves_order(anonymizer):
    first = anonymizer.anonymize(
        "The first observation occurred in 2020 and the next in 2022."
    )

    second = anonymizer.anonymize(
        "The first observation occurred in 2020 and the next in 2022."
    )

    assert first.masked_text == second.masked_text

    assert "T_0" in first.masked_text
    assert "T_1" in first.masked_text

    assert "2020" not in first.masked_text
    assert "2022" not in first.masked_text


def test_anonymized_document_internal_safety_gate(document):
    # Must complete without raising.
    document.assert_safe()


# ---------------------------------------------------------------------------
# Required agent/Pydantic gate
# ---------------------------------------------------------------------------


def test_agent_parses_valid_bounded_pydantic_json(
    document,
    candidate,
):
    client = FakeStructuredLLM(
        json.dumps(valid_response())
    )

    agent = QuantamentalFeatureAgent(client)

    result = agent.evaluate(
        document=document,
        candidate=candidate,
    )

    assert -1.0 <= result.e_sub <= 1.0
    assert -1.0 <= result.p_pricing <= 1.0
    assert -1.0 <= result.t_policy <= 1.0

    # Stronger project-specific constraint for policy.
    assert 0.0 <= result.t_policy <= 1.0

    assert 0.0 <= result.confidence <= 1.0

    assert result.candidate_node_alias == "Supplier_K"

    # Exact project formula:
    #
    # S = T_policy + (P_pricing * E_sub)
    #   = 0.40 + (0.50 * -0.80)
    #   = 0.00
    assert result.raw_bottleneck_score == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_agent_prompt_contains_no_original_sensitive_literals(
    document,
    candidate,
):
    client = FakeStructuredLLM(
        valid_response()
    )

    agent = QuantamentalFeatureAgent(client)

    agent.evaluate(
        document=document,
        candidate=candidate,
    )

    prompt = client.last_user_prompt

    assert prompt is not None

    folded = prompt.casefold()

    assert "borosil" not in folded
    assert "bororenew.ns" not in folded
    assert "rohan mehta" not in folded
    assert "covid-19" not in folded
    assert "2020" not in prompt
    assert "2022" not in prompt

    assert "Supplier_K" in prompt
    assert "Macro_Disruption_Alpha" in prompt
    assert "T_0" in prompt
    assert "T_1" in prompt


def test_agent_receives_json_schema(
    document,
    candidate,
):
    client = FakeStructuredLLM(
        valid_response()
    )

    agent = QuantamentalFeatureAgent(client)

    agent.evaluate(
        document=document,
        candidate=candidate,
    )

    schema = client.last_json_schema

    assert schema is not None
    assert isinstance(schema, Mapping)

    serialized = json.dumps(schema)

    assert "E_sub" in serialized
    assert "P_pricing" in serialized
    assert "T_policy" in serialized


# ---------------------------------------------------------------------------
# Strict boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("E_sub", -1.0001),
        ("E_sub", 1.0001),
        ("P_pricing", -1.5),
        ("P_pricing", 1.5),
        ("T_policy", -0.01),
        ("T_policy", 1.01),
    ],
)
def test_out_of_bounds_model_scores_are_rejected(
    document,
    candidate,
    field,
    bad_value,
):
    response = valid_response()
    response[field] = bad_value

    client = FakeStructuredLLM(response)
    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(
        LLMResponseValidationError
    ):
        agent.evaluate(
            document=document,
            candidate=candidate,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_model_scores_are_rejected(
    document,
    candidate,
    bad_value,
):
    response = valid_response()
    response["E_sub"] = bad_value

    client = FakeStructuredLLM(response)
    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(
        LLMResponseValidationError
    ):
        agent.evaluate(
            document=document,
            candidate=candidate,
        )


def test_invalid_json_is_rejected(
    document,
    candidate,
):
    client = FakeStructuredLLM(
        "{this is not valid json}"
    )

    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(
        LLMResponseValidationError
    ):
        agent.evaluate(
            document=document,
            candidate=candidate,
        )


def test_extra_model_fields_are_rejected(
    document,
    candidate,
):
    response = valid_response()
    response["stock_recommendation"] = "BUY"

    client = FakeStructuredLLM(response)
    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(
        LLMResponseValidationError
    ):
        agent.evaluate(
            document=document,
            candidate=candidate,
        )


def test_candidate_mismatch_is_rejected(
    document,
    candidate,
):
    response = valid_response()
    response["candidate_node_alias"] = "Supplier_X"

    client = FakeStructuredLLM(response)
    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(
        CandidateMismatchError
    ):
        agent.evaluate(
            document=document,
            candidate=candidate,
        )


def test_model_reidentification_leakage_is_rejected(
    document,
    candidate,
):
    response = valid_response()

    response["evidence"][0]["observation"] = (
        "Borosil Renewables has imported-input exposure."
    )

    client = FakeStructuredLLM(response)
    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(
        AgentOutputLeakageError
    ):
        agent.evaluate(
            document=document,
            candidate=candidate,
        )


def test_raw_string_cannot_be_sent_directly_to_agent(
    candidate,
):
    client = FakeStructuredLLM(
        valid_response()
    )

    agent = QuantamentalFeatureAgent(client)

    with pytest.raises(TypeError):
        agent.evaluate(
            document=SAMPLE_TEXT,  # type: ignore[arg-type]
            candidate=candidate,
        )


def test_ticker_like_candidate_alias_is_rejected(
    document,
):
    client = FakeStructuredLLM(
        valid_response()
    )

    agent = QuantamentalFeatureAgent(client)

    unsafe_candidate = CandidateNode(
        anonymized_id="BORORENEW.NS",
        layer=4,
        physical_role="Unsafe raw equity identifier.",
    )

    with pytest.raises(ValueError):
        agent.evaluate(
            document=document,
            candidate=unsafe_candidate,
        )


def test_calendar_year_candidate_alias_is_rejected(
    document,
):
    client = FakeStructuredLLM(
        valid_response()
    )

    agent = QuantamentalFeatureAgent(client)

    unsafe_candidate = CandidateNode(
        anonymized_id="Supplier_2020",
        layer=4,
        physical_role="Unsafe temporal identifier.",
    )

    with pytest.raises(ValueError):
        agent.evaluate(
            document=document,
            candidate=unsafe_candidate,
        )


def test_output_serialization_contains_computed_raw_score(
    document,
    candidate,
):
    client = FakeStructuredLLM(
        valid_response()
    )

    agent = QuantamentalFeatureAgent(client)

    result = agent.evaluate(
        document=document,
        candidate=candidate,
    )

    serialized = result.model_dump(
        by_alias=True,
        mode="json",
    )

    assert "S_raw" in serialized
    assert serialized["S_raw"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
