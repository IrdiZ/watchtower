"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: str = field(default_factory=lambda: os.environ.get("WATCHTOWER_DATA_DIR", "/data"))
    # Politeness: minimum seconds between two requests to the same host (rule: <= 1 req/s).
    min_request_interval: float = field(default_factory=lambda: _f("WATCHTOWER_MIN_REQUEST_INTERVAL", 1.05))
    # Slower floors for hosts with a published limit, as data ("host=seconds,..."). e-Zamowienia publishes
    # 30 requests/min/IP for its public API.
    host_intervals: str = field(default_factory=lambda: os.environ.get(
        "WATCHTOWER_HOST_INTERVALS", "ezamowienia.gov.pl=2.1,find-tender.service.gov.uk=6"))   # FTS: 429 at 1 req/3 s
    # Team decision 21 Nov: off by default (maximise coverage); WATCHTOWER_RESPECT_ROBOTS=1 enforces robots.txt.
    respect_robots: bool = field(default_factory=lambda: os.environ.get("WATCHTOWER_RESPECT_ROBOTS", "0") == "1")
    # Background refresh cadence for portals (seconds). 0 disables the scheduler.
    refresh_interval: float = field(default_factory=lambda: _f("WATCHTOWER_REFRESH_INTERVAL", 900))
    # Documents above this size are left out (rule allows omitting files > 250 MB).
    max_document_bytes: int = field(default_factory=lambda: int(_f("WATCHTOWER_MAX_DOCUMENT_BYTES", 250 * 1024 * 1024)))
    # Hard cap on total unpacked bytes from one ZIP (disk safety).
    max_zip_total_bytes: int = field(default_factory=lambda: int(_f("WATCHTOWER_MAX_ZIP_TOTAL_BYTES", 2 * 1024 * 1024 * 1024)))
    # Request budgets per job kind (attach/read/repair), see architecture v1.1 §6.
    attach_request_budget: int = field(default_factory=lambda: int(_f("WATCHTOWER_ATTACH_REQUEST_BUDGET", 1500)))
    repair_request_budget: int = field(default_factory=lambda: int(_f("WATCHTOWER_REPAIR_REQUEST_BUDGET", 1500)))
    repair_max_attempts: int = field(default_factory=lambda: int(_f("WATCHTOWER_REPAIR_MAX_ATTEMPTS", 3)))
    repair_spend_cap_usd: float = field(default_factory=lambda: _f("WATCHTOWER_REPAIR_SPEND_CAP_USD", 1.0))
    # A listing the portal caps (e.g. a 2,000-result window) is partial. Serve it only if Patterno accepts that.
    # Patterno (20 Nov): a working integration with incomplete coverage serves, marked "Coverage: partial".
    serve_capped: bool = field(default_factory=lambda: os.environ.get("WATCHTOWER_SERVE_CAPPED", "1") == "1")
    http_timeout: float = field(default_factory=lambda: _f("WATCHTOWER_HTTP_TIMEOUT", 30))
    # Says what we are (automated, polite) without the token "bot": servers commonly give user agents matching /bot/i
    # no session of their own (Tomcat's CrawlerSessionManagerValve), so a cookie check redirects forever. Team decision
    # 21 Nov (maximise coverage; the affected portal's robots.txt sets no rules).
    user_agent: str = field(default_factory=lambda: os.environ.get(
        "WATCHTOWER_USER_AGENT", "Watchtower/0.1 (+hackathon; automated tender monitor; polite 1 req/s)"))

    # Continuous verification (audit.py): every audit_interval seconds (0 = only on demand) per serving portal, also
    # soon after a new spec version. K tenders sampled, at most audit_renders browser renders, own request budget.
    audit_interval: float = field(default_factory=lambda: _f("WATCHTOWER_AUDIT_INTERVAL", 6 * 3600))
    audit_sample: int = field(default_factory=lambda: int(_f("WATCHTOWER_AUDIT_SAMPLE", 8)))
    audit_renders: int = field(default_factory=lambda: int(_f("WATCHTOWER_AUDIT_RENDERS", 4)))
    audit_request_budget: int = field(default_factory=lambda: int(_f("WATCHTOWER_AUDIT_REQUEST_BUDGET", 250)))
    audit_degraded_below: float = field(default_factory=lambda: _f("WATCHTOWER_AUDIT_DEGRADED_BELOW", 0.9))

    # --- LLM routing (OpenAI-compatible). Model names are NOT hard-coded: set them per task. ---
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_strong: str | None = field(default_factory=lambda: os.environ.get("MODEL_STRONG"))
    model_cheap: str | None = field(default_factory=lambda: os.environ.get("MODEL_CHEAP"))
    # One model budget per attach or repair job: calls beyond this fail the job ("model budget exhausted").
    model_max_calls_per_job: int = field(default_factory=lambda: int(_f("MODEL_MAX_CALLS_PER_JOB", 20)))
    llm_concurrency: int = field(default_factory=lambda: int(_f("LLM_CONCURRENCY", 4)))

    def reasoning_for(self, task: str) -> str | None:
        """Reasoning effort per task (cost/latency control). MODEL_REASONING_<TASK> overrides the tier default."""
        override = os.environ.get(f"MODEL_REASONING_{task.upper()}")
        if override is not None:
            return override or None
        cheap = task in {"classify", "audit", "triage", "verify", "narrow"}
        return (os.environ.get("MODEL_REASONING_CHEAP", "none") if cheap
                else os.environ.get("MODEL_REASONING_STRONG", "low")) or None

    def model_for(self, task: str) -> str | None:
        """Per-task override via MODEL_<TASK> (e.g. MODEL_SYNTHESIZE), else tier default."""
        override = os.environ.get(f"MODEL_{task.upper()}")
        if override:
            return override
        cheap_tasks = {"classify", "audit", "triage", "verify", "narrow"}
        return self.model_cheap if task in cheap_tasks else self.model_strong
