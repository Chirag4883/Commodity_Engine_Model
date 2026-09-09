from __future__ import annotations

"""
Level 5 — Live regulatory / corporate ingestion.

Sources
-------
DGFT  : trade notifications / notices
DGTR  : trade-remedy investigations and findings
BIS   : Quality Control Orders / upcoming enforcement
BSE   : corporate announcements

The ingestion layer produces normalized IngestedEvent objects.

Important
---------
A source failure is isolated and recorded. One broken government/exchange
website must not corrupt another source or silently generate fabricated data.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
from io import BytesIO, StringIO
import re
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class EventSource(str, Enum):
    DGFT = "DGFT"
    DGTR = "DGTR"
    BIS = "BIS"
    BSE = "BSE"
    MOCK = "MOCK"


@dataclass(frozen=True, slots=True)
class IngestedEvent:
    event_id: str
    source: EventSource
    event_type: str
    published_date: date
    title: str
    url: str
    text: str
    fetched_at_utc: datetime
    metadata: Mapping[str, str]

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True, slots=True)
class IngestionBatch:
    events: Tuple[IngestedEvent, ...]
    source_failures: Mapping[str, str]


class IngestionSource(Protocol):
    name: str

    def fetch(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> Sequence[IngestedEvent]:
        ...


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class ResilientHttpClient:

    def __init__(
        self,
        *,
        timeout_seconds: int = 20,
        max_retries: int = 3,
    ) -> None:

        self.timeout_seconds = timeout_seconds

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=0.8,
            status_forcelist=(
                429,
                500,
                502,
                503,
                504,
            ),
            allowed_methods=(
                "GET",
            ),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry
        )

        self.session = requests.Session()

        self.session.mount(
            "https://",
            adapter,
        )

        self.session.mount(
            "http://",
            adapter,
        )

        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(compatible; CommodityEngineModel/1.0; "
                    "+research)"
                ),
                "Accept-Language":
                    "en-IN,en;q=0.9",
            }
        )

    def get(
        self,
        url: str,
        *,
        params: Optional[
            Mapping[str, object]
        ] = None,
        headers: Optional[
            Mapping[str, str]
        ] = None,
    ) -> requests.Response:

        response = self.session.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )

        response.raise_for_status()

        return response

    def get_text(
        self,
        url: str,
        *,
        params: Optional[
            Mapping[str, object]
        ] = None,
        headers: Optional[
            Mapping[str, str]
        ] = None,
    ) -> str:

        return self.get(
            url,
            params=params,
            headers=headers,
        ).text

    def extract_document_text(
        self,
        url: str,
        *,
        max_chars: int = 12_000,
        max_pdf_pages: int = 8,
    ) -> str:
        """
        Retrieve useful text from a linked HTML/PDF document.

        Failure is deliberately non-fatal; the announcement title remains
        available to the pipeline.
        """

        try:
            response = self.get(
                url
            )
        except Exception:
            return ""

        content_type = (
            response.headers
            .get(
                "Content-Type",
                "",
            )
            .lower()
        )

        is_pdf = (
            "pdf"
            in content_type
            or url.lower().endswith(
                ".pdf"
            )
        )

        if is_pdf:
            try:
                reader = PdfReader(
                    BytesIO(
                        response.content
                    )
                )

                chunks = []

                for page in (
                    reader.pages[
                        :max_pdf_pages
                    ]
                ):
                    value = (
                        page.extract_text()
                        or ""
                    )

                    chunks.append(
                        value
                    )

                return (
                    "\n".join(
                        chunks
                    )[:max_chars]
                )

            except Exception:
                return ""

        try:
            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                ]
            ):
                element.decompose()

            return (
                " ".join(
                    soup.stripped_strings
                )[:max_chars]
            )

        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def make_event(
    *,
    source: EventSource,
    event_type: str,
    published_date: date,
    title: str,
    url: str,
    text: str = "",
    metadata: Optional[
        Mapping[str, str]
    ] = None,
    fetched_at_utc: Optional[
        datetime
    ] = None,
) -> IngestedEvent:

    normalized_title = (
        " ".join(
            title.split()
        )
    )

    canonical = (
        f"{source.value}|"
        f"{event_type}|"
        f"{published_date.isoformat()}|"
        f"{normalized_title.casefold()}|"
        f"{url}"
    )

    event_id = hashlib.sha256(
        canonical.encode(
            "utf-8"
        )
    ).hexdigest()

    return IngestedEvent(
        event_id=event_id,
        source=source,
        event_type=event_type,
        published_date=(
            published_date
        ),
        title=normalized_title,
        url=url,
        text=text.strip(),
        fetched_at_utc=(
            fetched_at_utc
            or datetime.now(
                timezone.utc
            )
        ),
        metadata=dict(
            metadata
            or {}
        ),
    )


_DATE_PATTERNS = (
    r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b",
    r"\b\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\s+(?:19|20)\d{2}\b",
    r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b",
)


def extract_date(
    text: str,
) -> Optional[date]:

    for pattern in _DATE_PATTERNS:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        try:
            parsed = pd.to_datetime(
                match.group(0),
                dayfirst=True,
                errors="raise",
            )

            return parsed.date()

        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# DGTR
# ---------------------------------------------------------------------------


class DGTRSource:

    name = "DGTR"

    URL = (
        "https://www.dgtr.gov.in/en"
    )

    KEYWORDS = re.compile(
        r"\b("
        r"anti[- ]dumping|"
        r"countervailing|"
        r"anti[- ]subsidy|"
        r"final findings?|"
        r"final finding|"
        r"initiation|"
        r"trade remed|"
        r"safeguard"
        r")\b",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        client: Optional[
            ResilientHttpClient
        ] = None,
    ) -> None:

        self.client = (
            client
            or ResilientHttpClient()
        )

    def fetch(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> Sequence[
        IngestedEvent
    ]:

        html = self.client.get_text(
            self.URL
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        events = []

        for anchor in soup.find_all(
            "a",
            href=True,
        ):

            parent = (
                anchor.parent
                or anchor
            )

            context = " ".join(
                parent.stripped_strings
            )

            if not self.KEYWORDS.search(
                context
            ):
                continue

            published = extract_date(
                context
            )

            if published is None:
                continue

            if not (
                start_date
                <= published
                <= end_date
            ):
                continue

            title = (
                anchor.get_text(
                    " ",
                    strip=True,
                )
                or context
            )

            url = urljoin(
                self.URL,
                anchor[
                    "href"
                ],
            )

            document_text = (
                self.client
                .extract_document_text(
                    url
                )
            )

            events.append(
                make_event(
                    source=(
                        EventSource.DGTR
                    ),
                    event_type=(
                        "trade_remedy"
                    ),
                    published_date=(
                        published
                    ),
                    title=title,
                    url=url,
                    text=(
                        document_text
                        or context
                    ),
                )
            )

        return events


# ---------------------------------------------------------------------------
# DGFT
# ---------------------------------------------------------------------------


class DGFTSource:

    name = "DGFT"

    URL = (
        "https://www.dgft.gov.in/CP/"
        "?opt=notification"
    )

    KEYWORDS = re.compile(
        r"\b("
        r"notification|"
        r"public notice|"
        r"trade notice|"
        r"policy circular|"
        r"circular|"
        r"import|"
        r"export"
        r")\b",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        client: Optional[
            ResilientHttpClient
        ] = None,
    ) -> None:

        self.client = (
            client
            or ResilientHttpClient()
        )

    def fetch(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> Sequence[
        IngestedEvent
    ]:

        html = self.client.get_text(
            self.URL
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        events = []

        for anchor in soup.find_all(
            "a",
            href=True,
        ):

            parent = (
                anchor.parent
                or anchor
            )

            context = " ".join(
                parent.stripped_strings
            )

            if not self.KEYWORDS.search(
                context
            ):
                continue

            published = extract_date(
                context
            )

            if published is None:
                continue

            if not (
                start_date
                <= published
                <= end_date
            ):
                continue

            title = (
                anchor.get_text(
                    " ",
                    strip=True,
                )
                or context
            )

            url = urljoin(
                self.URL,
                anchor[
                    "href"
                ],
            )

            document_text = (
                self.client
                .extract_document_text(
                    url
                )
            )

            events.append(
                make_event(
                    source=(
                        EventSource.DGFT
                    ),
                    event_type=(
                        "trade_policy"
                    ),
                    published_date=(
                        published
                    ),
                    title=title,
                    url=url,
                    text=(
                        document_text
                        or context
                    ),
                )
            )

        return events


# ---------------------------------------------------------------------------
# BIS
# ---------------------------------------------------------------------------


class BISSource:

    name = "BIS"

    URL = (
        "https://www.bis.gov.in/"
        "upcoming-qcos-notified-and-due-for-implementation/"
        "?lang=en"
    )

    def __init__(
        self,
        client: Optional[
            ResilientHttpClient
        ] = None,
    ) -> None:

        self.client = (
            client
            or ResilientHttpClient()
        )

    def fetch(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> Sequence[
        IngestedEvent
    ]:

        html = self.client.get_text(
            self.URL
        )

        tables = pd.read_html(
            StringIO(
                html
            )
        )

        events = []

        for table in tables:

            normalized = {
                str(column)
                .strip()
                .casefold():
                    column
                for column
                in table.columns
            }

            product_column = next(
                (
                    original
                    for (
                        normalized_name,
                        original,
                    )
                    in normalized.items()
                    if "product"
                    in normalized_name
                ),
                None,
            )

            date_column = next(
                (
                    original
                    for (
                        normalized_name,
                        original,
                    )
                    in normalized.items()
                    if (
                        "enforcement"
                        in normalized_name
                        or "implementation"
                        in normalized_name
                    )
                ),
                None,
            )

            if (
                product_column is None
                or date_column is None
            ):
                continue

            ministry_column = next(
                (
                    original
                    for (
                        normalized_name,
                        original,
                    )
                    in normalized.items()
                    if "ministry"
                    in normalized_name
                ),
                None,
            )

            standard_column = next(
                (
                    original
                    for (
                        normalized_name,
                        original,
                    )
                    in normalized.items()
                    if "standard"
                    in normalized_name
                ),
                None,
            )

            for _, row in (
                table.iterrows()
            ):

                product = str(
                    row[
                        product_column
                    ]
                ).strip()

                if (
                    not product
                    or product.casefold()
                    == "nan"
                ):
                    continue

                try:
                    enforcement = (
                        pd.to_datetime(
                            row[
                                date_column
                            ],
                            dayfirst=True,
                            errors="raise",
                        )
                        .date()
                    )
                except Exception:
                    continue

                # The scheduled pipeline looks back one week, but future QCOs
                # are economically relevant before their enforcement date.
                #
                # Accept orders becoming effective up to 180 days ahead.
                horizon_end = (
                    end_date
                    + pd.Timedelta(
                        days=180
                    )
                ).date()

                if not (
                    start_date
                    <= enforcement
                    <= horizon_end
                ):
                    continue

                ministry = (
                    str(
                        row[
                            ministry_column
                        ]
                    )
                    if ministry_column
                    is not None
                    else ""
                )

                standard = (
                    str(
                        row[
                            standard_column
                        ]
                    )
                    if standard_column
                    is not None
                    else ""
                )

                title = (
                    f"Quality Control Order: "
                    f"{product}"
                )

                text = (
                    f"Product: {product}. "
                    f"Enforcement date: "
                    f"{enforcement.isoformat()}. "
                    f"Ministry/Department: "
                    f"{ministry}. "
                    f"Indian Standard: "
                    f"{standard}."
                )

                events.append(
                    make_event(
                        source=(
                            EventSource.BIS
                        ),
                        event_type="qco",
                        published_date=(
                            min(
                                enforcement,
                                end_date,
                            )
                        ),
                        title=title,
                        url=self.URL,
                        text=text,
                        metadata={
                            "enforcement_date":
                                enforcement
                                .isoformat(),
                            "product":
                                product,
                        },
                    )
                )

        return events


# ---------------------------------------------------------------------------
# BSE
# ---------------------------------------------------------------------------


class BSESource:

    name = "BSE"

    API_URL = (
        "https://api.bseindia.com/"
        "BseIndiaAPI/api/"
        "AnnSubCategoryGetData/w"
    )

    REFERER = (
        "https://www.bseindia.com/"
        "corporates/ann.html"
    )

    def __init__(
        self,
        client: Optional[
            ResilientHttpClient
        ] = None,
    ) -> None:

        self.client = (
            client
            or ResilientHttpClient()
        )

    def fetch(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> Sequence[
        IngestedEvent
    ]:

        params = {
            "pageno": 1,
            "strCat": -1,
            "strPrevDate":
                start_date.strftime(
                    "%Y%m%d"
                ),
            "strScrip": "",
            "strSearch": "P",
            "strToDate":
                end_date.strftime(
                    "%Y%m%d"
                ),
            "strType": "C",
        }

        response = self.client.get(
            self.API_URL,
            params=params,
            headers={
                "Referer":
                    self.REFERER,
                "Origin":
                    "https://www.bseindia.com",
            },
        )

        payload = response.json()

        rows = (
            payload.get(
                "Table",
                []
            )
            if isinstance(
                payload,
                dict,
            )
            else []
        )

        events = []

        for row in rows:

            if not isinstance(
                row,
                dict,
            ):
                continue

            title = str(
                row.get(
                    "NEWSSUB"
                )
                or row.get(
                    "HEADLINE"
                )
                or row.get(
                    "NEWS_SUB"
                )
                or ""
            ).strip()

            if not title:
                continue

            date_value = (
                row.get(
                    "NEWS_DT"
                )
                or row.get(
                    "DissemDT"
                )
                or row.get(
                    "DT_TM"
                )
            )

            try:
                published = (
                    pd.to_datetime(
                        date_value,
                        dayfirst=True,
                        errors="raise",
                    )
                    .date()
                )
            except Exception:
                continue

            if not (
                start_date
                <= published
                <= end_date
            ):
                continue

            attachment = str(
                row.get(
                    "ATTACHMENTNAME"
                )
                or ""
            ).strip()

            if attachment:
                if attachment.startswith(
                    "http"
                ):
                    url = attachment
                else:
                    url = (
                        "https://www.bseindia.com/"
                        "xml-data/corpfiling/"
                        "AttachLive/"
                        f"{attachment}"
                    )
            else:
                url = self.REFERER

            document_text = (
                self.client
                .extract_document_text(
                    url
                )
                if attachment
                else ""
            )

            company_name = str(
                row.get(
                    "SLONGNAME"
                )
                or row.get(
                    "LONG_NAME"
                )
                or ""
            ).strip()

            scrip_code = str(
                row.get(
                    "SCRIP_CD"
                )
                or ""
            ).strip()

            category = str(
                row.get(
                    "CATEGORYNAME"
                )
                or ""
            ).strip()

            events.append(
                make_event(
                    source=(
                        EventSource.BSE
                    ),
                    event_type=(
                        "corporate_announcement"
                    ),
                    published_date=(
                        published
                    ),
                    title=title,
                    url=url,
                    text=(
                        document_text
                        or title
                    ),
                    metadata={
                        "company_name":
                            company_name,
                        "scrip_code":
                            scrip_code,
                        "category":
                            category,
                    },
                )
            )

        return events


# ---------------------------------------------------------------------------
# Mock source
# ---------------------------------------------------------------------------


class MockSource:

    name = "MOCK"

    def __init__(
        self,
        events: Sequence[
            IngestedEvent
        ],
    ) -> None:

        self._events = tuple(
            events
        )

    def fetch(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> Sequence[
        IngestedEvent
    ]:

        return tuple(
            event
            for event
            in self._events
            if (
                start_date
                <= event.published_date
                <= end_date
            )
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class LiveIngestor:

    def __init__(
        self,
        sources: Iterable[
            IngestionSource
        ],
    ) -> None:

        self.sources = tuple(
            sources
        )

    @classmethod
    def default(
        cls,
    ) -> "LiveIngestor":

        client = (
            ResilientHttpClient()
        )

        return cls(
            (
                DGFTSource(
                    client
                ),
                DGTRSource(
                    client
                ),
                BISSource(
                    client
                ),
                BSESource(
                    client
                ),
            )
        )

    def collect(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> IngestionBatch:

        unique: Dict[
            str,
            IngestedEvent,
        ] = {}

        failures: Dict[
            str,
            str,
        ] = {}

        for source in (
            self.sources
        ):

            try:
                events = source.fetch(
                    start_date=start_date,
                    end_date=end_date,
                )

                for event in events:
                    unique[
                        event.event_id
                    ] = event

            except Exception as exc:
                failures[
                    source.name
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        ordered = tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.published_date,
                    item.source.value,
                    item.event_id,
                ),
            )
        )

        return IngestionBatch(
            events=ordered,
            source_failures=failures,
        )
