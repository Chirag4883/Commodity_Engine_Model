from __future__ import annotations

"""
Level 2 — Point-in-Time Anonymizer

Purpose
-------
Prevent historical backtests from leaking:
- company identities,
- NSE/BSE ticker identities,
- executive identities,
- historically memorable event identifiers,
- explicit calendar years.

The anonymizer is deterministic. Given identical configuration and text,
it produces identical masking.

Important
---------
The replacement map remains local. It must NEVER be inserted into an LLM
prompt. Level 3 may use local metadata to reconnect an anonymized observation
to its point-in-time graph/equity identifier after inference.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnonymizationError(Exception):
    """Base exception for anonymization failures."""


class LeakageDetectedError(AnonymizationError):
    """Raised when sensitive information remains in masked text."""


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class RedactionCategory(str, Enum):
    COMPANY = "company"
    TICKER = "ticker"
    EXECUTIVE = "executive"
    HISTORICAL_EVENT = "historical_event"
    YEAR = "year"


@dataclass(frozen=True, slots=True)
class RedactionRecord:
    category: RedactionCategory
    original: str
    replacement: str


@dataclass(frozen=True, slots=True)
class AnonymizedDocument:
    """
    Safe-to-prompt representation.

    `redactions` is intentionally retained outside the LLM prompt so Level 3
    can maintain local PIT provenance.

    Call `prompt_text` when constructing model prompts.
    """

    masked_text: str
    redactions: Tuple[RedactionRecord, ...]

    @property
    def prompt_text(self) -> str:
        """
        The only text field intended to cross the model boundary.
        """
        return self.masked_text

    @property
    def redaction_count(self) -> int:
        return len(self.redactions)

    def prohibited_literals(self) -> Tuple[str, ...]:
        """
        Sensitive source literals that must not appear in prompts or output.
        """
        return tuple(
            record.original
            for record in self.redactions
            if record.category != RedactionCategory.YEAR
        )

    def assert_safe(self) -> None:
        """
        Re-run leakage checks against the masked document.
        """
        _assert_no_literal_leakage(
            text=self.masked_text,
            redactions=self.redactions,
        )


@dataclass(frozen=True, slots=True)
class _ReplacementRule:
    category: RedactionCategory
    source: str
    replacement: str


# ---------------------------------------------------------------------------
# Seed masking dictionaries
# ---------------------------------------------------------------------------
#
# These aliases are arbitrary anonymization identifiers. They contain no
# investment meaning.
#
# Multiple textual variants may intentionally map to one anonymous entity.
# ---------------------------------------------------------------------------


DEFAULT_COMPANY_ALIASES: Mapping[str, str] = {
    # Solar glass
    "Borosil Renewables Limited": "Supplier_K",
    "Borosil Renewables": "Supplier_K",
    "Borosil": "Supplier_K",

    # Transformers
    "Transformers & Rectifiers (India) Limited": "Supplier_T",
    "Transformers and Rectifiers (India) Limited": "Supplier_T",
    "Transformers and Rectifiers India": "Supplier_T",
    "Transformers & Rectifiers India": "Supplier_T",

    "Voltamp Transformers Limited": "Supplier_V",
    "Voltamp Transformers": "Supplier_V",

    # Fluorochemicals
    "SRF Limited": "Supplier_F",
    "SRF": "Supplier_F",

    "Navin Fluorine International Limited": "Supplier_N",
    "Navin Fluorine International": "Supplier_N",
    "Navin Fluorine": "Supplier_N",

    # Specialty chemicals
    "Aarti Industries Limited": "Supplier_A",
    "Aarti Industries": "Supplier_A",
}


DEFAULT_TICKER_ALIASES: Mapping[str, str] = {
    "TRIL.NS": "Equity_A",
    "VOLTAMP.NS": "Equity_B",
    "BORORENEW.NS": "Equity_C",
    "SRF.NS": "Equity_D",
    "NAVINFLUOR.NS": "Equity_E",
    "AARTIIND.NS": "Equity_F",
}


DEFAULT_EVENT_ALIASES: Mapping[str, str] = {
    "COVID-19": "Macro_Disruption_Alpha",
    "COVID 19": "Macro_Disruption_Alpha",
    "coronavirus pandemic": "Macro_Disruption_Alpha",
}


# Relevant PIT research universe is modern, so deliberately target ordinary
# explicit years from 1900 through 2099.
_CALENDAR_YEAR_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?!\d)"
)


# ---------------------------------------------------------------------------
# Public anonymizer
# ---------------------------------------------------------------------------


class PITAnonymizer:
    """
    Deterministic point-in-time document anonymizer.

    Parameters
    ----------
    company_aliases:
        Optional mapping added to the default company dictionary.

    ticker_aliases:
        Optional mapping added to the default ticker dictionary.

    executive_names:
        Executive names to redact for the current PIT corpus. These are
        deterministically mapped to Executive_A, Executive_B, ...

    event_aliases:
        Optional event identifier mappings added to the defaults.
    """

    def __init__(
        self,
        *,
        company_aliases: Optional[Mapping[str, str]] = None,
        ticker_aliases: Optional[Mapping[str, str]] = None,
        executive_names: Optional[Sequence[str]] = None,
        event_aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        companies = dict(DEFAULT_COMPANY_ALIASES)
        tickers = dict(DEFAULT_TICKER_ALIASES)
        events = dict(DEFAULT_EVENT_ALIASES)

        if company_aliases:
            companies.update(company_aliases)

        if ticker_aliases:
            tickers.update(ticker_aliases)

        if event_aliases:
            events.update(event_aliases)

        self._rules = self._build_rules(
            company_aliases=companies,
            ticker_aliases=tickers,
            executive_names=executive_names or (),
            event_aliases=events,
        )

    def anonymize(self, text: str) -> AnonymizedDocument:
        """
        Return an anonymized immutable document.

        Raises
        ------
        ValueError
            If text is not a non-empty string.

        LeakageDetectedError
            If masking completes but a prohibited literal/calendar year
            remains.
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        masked = text
        records = []

        # Longest literals first prevents a short alias such as "Borosil"
        # from partially consuming "Borosil Renewables Limited".
        for rule in self._rules:
            pattern = _literal_pattern(rule.source)

            if not pattern.search(masked):
                continue

            masked = pattern.sub(rule.replacement, masked)

            records.append(
                RedactionRecord(
                    category=rule.category,
                    original=rule.source,
                    replacement=rule.replacement,
                )
            )

        # Calendar-year anonymization is deterministic by chronological order,
        # preserving relative time ordering while removing absolute time.
        years = sorted(
            set(_CALENDAR_YEAR_RE.findall(masked)),
            key=int,
        )

        for index, year in enumerate(years):
            replacement = f"T_{index}"

            masked = re.sub(
                rf"(?<!\d){re.escape(year)}(?!\d)",
                replacement,
                masked,
            )

            records.append(
                RedactionRecord(
                    category=RedactionCategory.YEAR,
                    original=year,
                    replacement=replacement,
                )
            )

        document = AnonymizedDocument(
            masked_text=masked,
            redactions=tuple(records),
        )

        document.assert_safe()

        return document

    @staticmethod
    def _build_rules(
        *,
        company_aliases: Mapping[str, str],
        ticker_aliases: Mapping[str, str],
        executive_names: Iterable[str],
        event_aliases: Mapping[str, str],
    ) -> Tuple[_ReplacementRule, ...]:
        rules = []

        for source, replacement in company_aliases.items():
            _validate_rule(source, replacement)

            rules.append(
                _ReplacementRule(
                    RedactionCategory.COMPANY,
                    source,
                    replacement,
                )
            )

        for source, replacement in ticker_aliases.items():
            _validate_rule(source, replacement)

            rules.append(
                _ReplacementRule(
                    RedactionCategory.TICKER,
                    source,
                    replacement,
                )
            )

        for source, replacement in event_aliases.items():
            _validate_rule(source, replacement)

            rules.append(
                _ReplacementRule(
                    RedactionCategory.HISTORICAL_EVENT,
                    source,
                    replacement,
                )
            )

        normalized_executives = sorted(
            {
                name.strip()
                for name in executive_names
                if isinstance(name, str) and name.strip()
            },
            key=str.casefold,
        )

        for index, executive_name in enumerate(normalized_executives):
            rules.append(
                _ReplacementRule(
                    RedactionCategory.EXECUTIVE,
                    executive_name,
                    _executive_alias(index),
                )
            )

        # Critical: perform the longest literal replacements first.
        rules.sort(
            key=lambda item: (
                len(item.source),
                item.source.casefold(),
            ),
            reverse=True,
        )

        return tuple(rules)


# ---------------------------------------------------------------------------
# Leakage utilities
# ---------------------------------------------------------------------------


def assert_no_calendar_years(text: str) -> None:
    """
    Public assertion utility useful in tests/pipeline validation.
    """
    match = _CALENDAR_YEAR_RE.search(text)

    if match:
        raise LeakageDetectedError(
            f"Explicit calendar year remains after anonymization: "
            f"{match.group(0)!r}"
        )


def assert_no_sensitive_literals(
    text: str,
    prohibited_literals: Iterable[str],
) -> None:
    """
    Assert that arbitrary model-bound/model-produced text does not contain
    sensitive literals.
    """
    folded_text = text.casefold()

    for literal in prohibited_literals:
        if literal and literal.casefold() in folded_text:
            raise LeakageDetectedError(
                f"Sensitive literal detected: {literal!r}"
            )

    assert_no_calendar_years(text)


def _assert_no_literal_leakage(
    *,
    text: str,
    redactions: Iterable[RedactionRecord],
) -> None:
    prohibited = (
        record.original
        for record in redactions
        if record.category != RedactionCategory.YEAR
    )

    assert_no_sensitive_literals(
        text,
        prohibited,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _literal_pattern(value: str) -> re.Pattern[str]:
    """
    Literal matcher with conservative alphanumeric boundaries.

    Works for:
        SRF
        SRF.NS
        COVID-19
        Borosil Renewables
        names containing punctuation
    """
    escaped = re.escape(value)

    return re.compile(
        rf"(?<!\w){escaped}(?!\w)",
        flags=re.IGNORECASE,
    )


def _executive_alias(index: int) -> str:
    """
    Deterministic aliases:
        0  -> Executive_A
        1  -> Executive_B
        ...
        25 -> Executive_Z
        26 -> Executive_27
    """
    if 0 <= index < 26:
        return f"Executive_{chr(ord('A') + index)}"

    return f"Executive_{index + 1}"


def _validate_rule(source: str, replacement: str) -> None:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Masking-rule source must be a non-empty string")

    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError(
            "Masking-rule replacement must be a non-empty string"
        )
