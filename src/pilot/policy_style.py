"""One colour, marker and dash per policy, shared by every results figure of the paper.

The colours are the house palette taken in order down the policy list below, so a legend that
lists its policies in ``ORDER`` runs through the palette in order.  The list follows the paper's
families: TAP (``tap``, the planner of Section IV) first, in indigo as in Fig. 1, with the plan it
is built on straight after it (the timetable with learned timing and the previous semester's
prior, ``calendar_learn_prior``) in the same indigo, told apart by its marker and dash; the
timetable with known timing and its variants; the feedback rules and the history forecast; the
reactive rule in rose, as in Fig. 1; the robust rules; the optimum in wine.  Fifteen policies do
not fit eight colours, so a variant shares the colour of the policy it modifies and gets its own
marker, and the trivial base-tier baseline is neutral grey.  A feedback rule tuned against the
SLO floor keeps the colour and marker of the rule, drawn hollow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

PALETTE: Tuple[str, ...] = ("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499",
                            "#882255")
INDIGO, GREEN, TEAL, LIGHTBLUE, SAND, ROSE, PURPLE, WINE = PALETTE
GREY = "#7A7A7A"          # neutral: references, bounds, the base tier
TUNED = " (SLO>=0.95, disp)"


@dataclass(frozen=True)
class PolicyStyle:
    """How one policy is drawn wherever it appears."""

    color: str
    marker: str
    dash: object = "-"
    hollow: bool = False

    def line_kw(self) -> Dict[str, object]:
        """Keyword arguments for ``Axes.plot``."""
        kw: Dict[str, object] = {"color": self.color, "marker": self.marker, "linestyle": self.dash,
                                 "markeredgecolor": self.color}
        if self.hollow:
            kw["markerfacecolor"] = "white"
        return kw


ORDER: Tuple[str, ...] = (
    "tap", "calendar_learn_prior", "calendar", "calendar_adapt", "calendar_learn", "stale_calendar",
    "velocity", "velocity" + TUNED, "forecast_ilp", "reactive", "reactive" + TUNED,
    "guard", "lcp", "offline", "informed", "fixed",
)

STYLE: Dict[str, PolicyStyle] = {
    "tap": PolicyStyle(INDIGO, "o", "-"),
    "calendar_learn_prior": PolicyStyle(INDIGO, ">", (0, (4, 1.5))),
    "calendar": PolicyStyle(GREEN, "s", (0, (4, 1.5))),
    "calendar_adapt": PolicyStyle(GREEN, "h", (0, (1.2, 1.2))),
    "calendar_learn": PolicyStyle(TEAL, "v", (0, (4, 1.5))),
    "stale_calendar": PolicyStyle(TEAL, "<", (0, (1.2, 1.2))),
    "velocity": PolicyStyle(LIGHTBLUE, "D", (0, (5, 1.5, 1.2, 1.5))),
    "velocity" + TUNED: PolicyStyle(LIGHTBLUE, "D", (0, (5, 1.5, 1.2, 1.5)), hollow=True),
    "forecast_ilp": PolicyStyle(SAND, "P", "-"),
    "reactive": PolicyStyle(ROSE, "^", (0, (4, 1.5))),
    "reactive" + TUNED: PolicyStyle(ROSE, "^", (0, (4, 1.5)), hollow=True),
    "guard": PolicyStyle(PURPLE, "X", (0, (5, 1.5, 1.2, 1.5))),
    "lcp": PolicyStyle(PURPLE, "d", (0, (1.2, 1.2))),
    "offline": PolicyStyle(WINE, "*", "-"),
    "informed": PolicyStyle(WINE, "p", (0, (1.2, 1.2))),
    "fixed": PolicyStyle(GREY, "o", (0, (1.2, 1.2))),
}


def style(policy: str) -> PolicyStyle:
    """The style of ``policy``; an unknown name is an error, not a silent default colour."""
    try:
        return STYLE[policy]
    except KeyError as exc:
        raise KeyError(f"no style for policy {policy!r}; add it to pilot.policy_style") from exc


def rank(policy: str) -> int:
    """Position in ``ORDER``, for sorting legend entries."""
    return ORDER.index(policy) if policy in ORDER else len(ORDER)


__all__ = ["GREY", "INDIGO", "GREEN", "TEAL", "LIGHTBLUE", "SAND", "ROSE", "PURPLE", "WINE", "ORDER", "PALETTE",
           "PolicyStyle", "STYLE", "TUNED", "rank", "style"]
