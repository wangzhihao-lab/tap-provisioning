"""Loaders for the generic LLM serving traces used as the Q1 control group.

None of them is a campus workload (refs/scans/E_week0_leftovers_I1.md sections A1-A3):

* BurstGPT -- regional Azure OpenAI GPT services, CC-BY-4.0. ``Timestamp`` is *seconds from
  0:00:00 of the first day*, and the release does not say which calendar day that was, so an
  anchor is supplied by ``LoaderConfig.burstgpt_origin`` (a Monday by default). Weekday slots are
  therefore self-consistent but not calendar-true; nothing in Q1 depends on the true weekday.
* Azure LLM inference traces 2023 / 2024 -- ``TIMESTAMP``, ``ContextTokens``, ``GeneratedTokens``;
  one day (2023-11-11) and ten days (2024-05-10..19) respectively, CC-BY.
* SageServe -- the release schema is not documented on the pages we could reach, so the loader
  inspects the header at runtime and raises with the columns it found when nothing looks like an
  arrival time.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from ..config import LoaderConfig
from . import register_source
from .base import ArrivalSource, parse_timestamps

logger = logging.getLogger(__name__)

# deliberately narrow: a bare "time" would also match the prompt_time / e2e_time columns of
# the SageServe performance-model CSVs, which are not arrival traces
_TIME_CANDIDATES = ("timestamp", "arrival", "start_time", "req_time", "invocation", "submit")


def _pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """First column whose lower-cased name contains one of ``candidates``."""
    lowered = {str(c): str(c).lower().replace(" ", "_") for c in columns}
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


def _read_csv(path: Path, cfg: LoaderConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trace file not found: {path}")
    if path.suffix.lower() not in {".csv", ".gz", ".zip"}:
        raise ValueError(f"{path.name}: expected a CSV (optionally gzipped), got '{path.suffix}'")
    return pd.read_csv(path, nrows=cfg.max_rows or None)


class _TraceSource(ArrivalSource):
    """Shared plumbing: one CSV in, an arrival frame out, no timetable."""

    workload_class = "generic"
    n_paths = 1
    timetable_kind = "detected"
    trace_name = "trace"

    def __init__(self, path: Path, cfg: Optional[LoaderConfig] = None,
                 group: Optional[str] = None) -> None:
        self.path = Path(path)
        self.cfg = cfg or LoaderConfig()
        self.group = group or self.trace_name

    def _frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Turn the raw CSV into ts/group/kind; implemented per trace."""
        raise NotImplementedError

    def _tokens(self, raw: pd.DataFrame, df: pd.DataFrame, req: Sequence[str],
                resp: Sequence[str]) -> pd.DataFrame:
        """Attach request/response token counts when the trace carries them."""
        for dst, cands in (("req_tokens", req), ("resp_tokens", resp)):
            col = _pick_column(list(raw.columns), cands)
            if col:
                df[dst] = pd.to_numeric(raw[col], errors="coerce")
        return df

    def load(self) -> pd.DataFrame:
        raw = _read_csv(self.path, self.cfg)
        df = self._frame(raw)
        out = self.validate(df)
        logger.info("%s: %d requests, %s to %s", self.path.name, len(out), out["ts"].min(),
                    out["ts"].max())
        return out


@register_source("burstgpt")
class BurstGPTTrace(_TraceSource):
    """BurstGPT CSV: Timestamp (seconds from day 0), Request tokens, Response tokens, Model."""

    trace_name = "burstgpt"

    def _frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        col = _pick_column(list(raw.columns), ("timestamp", "time"))
        if col is None:
            raise ValueError(f"{self.path.name}: no Timestamp column in {list(raw.columns)}")
        seconds = pd.to_numeric(raw[col], errors="coerce")
        if seconds.isna().mean() > 0.5:
            ts = parse_timestamps(raw[col])
        else:
            origin = pd.Timestamp(self.cfg.burstgpt_origin)
            logger.warning("BurstGPT timestamps are relative; anchoring day 0 at %s (nominal)",
                           origin.date())
            ts = origin + pd.to_timedelta(seconds, unit="s")
        df = pd.DataFrame({"ts": ts, "group": self.group, "kind": "request"})
        return self._tokens(raw, df, ("request_token", "context"), ("response_token", "generated"))


class _AzureTrace(_TraceSource):
    """Azure LLM inference trace: TIMESTAMP, ContextTokens, GeneratedTokens."""

    def _frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        col = _pick_column(list(raw.columns), ("timestamp", "time"))
        if col is None:
            raise ValueError(f"{self.path.name}: no TIMESTAMP column in {list(raw.columns)}")
        df = pd.DataFrame({"ts": parse_timestamps(raw[col]), "group": self.group,
                           "kind": "request"})
        return self._tokens(raw, df, ("contexttokens", "context"), ("generatedtokens", "generated"))


@register_source("azure_2023")
class AzureTrace2023(_AzureTrace):
    """AzureLLMInferenceDataset2023 (single day, 2023-11-11)."""

    trace_name = "azure_2023"


@register_source("azure_2024")
class AzureTrace2024(_AzureTrace):
    """AzureLLMInferenceDataset2024 (2024-05-10..19, Code and Conversation files)."""

    trace_name = "azure_2024"


@register_source("sageserve")
class SageServeTrace(_TraceSource):
    """SageServe release trace; the schema is inspected at runtime because it is undocumented."""

    trace_name = "sageserve"

    def _frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        columns: List[str] = [str(c) for c in raw.columns]
        col = _pick_column(columns, _TIME_CANDIDATES)
        if col is None:
            raise ValueError(
                f"{self.path.name}: no arrival-time column among {columns}. The SageServe release "
                "(github.com/shashwatj07/SageServe) ships no data dictionary; pass a file whose "
                "time column matches one of " + ", ".join(_TIME_CANDIDATES) + ", or extend "
                "SageServeTrace._frame with the real schema."
            )
        numeric = pd.to_numeric(raw[col], errors="coerce")
        ts = parse_timestamps(raw[col] if numeric.isna().mean() > 0.5 else numeric)
        if ts.isna().mean() > 0.5:
            raise ValueError(
                f"{self.path.name}: column '{col}' does not parse as arrival times "
                f"(sample: {list(raw[col].head(3))})."
            )
        df = pd.DataFrame({"ts": ts, "group": self.group, "kind": "request"})
        return self._tokens(raw, df, ("input", "prompt", "context"), ("output", "generated"))
