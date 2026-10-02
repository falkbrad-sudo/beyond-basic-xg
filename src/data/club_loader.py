"""Adapter stub for a club's own shot and tracking data. NOT IMPLEMENTED.

This file makes the extension path from public sample data to a real club
feed concrete rather than asserted. Given access to a club's tracking feed
and shot/event data, implement the functions below to match the schemas
metrica_loader.py already produces. Every function in src/features/
(geometry, pitch_control_at_shot, shot_context) and src/models/ operates on
those standardized schemas and needs no changes to work with real data
instead of the Metrica sample.

What it would take:
1. Get the raw tracking and shot-event export format from the club or vendor.
2. Add a `<vendor>_to_meters()` conversion function to src/data/cleaning.py.
3. Add access config (credentials, paths) to config.yaml under a new
   `club:` section.
4. Implement extract_shot_events() and get_tracking_frame_at_shot() below
   to match metrica_loader.py's corresponding function signatures exactly.

What would NOT need to change: src/features/geometry.py (pure math,
vendor-agnostic already), pitch_control_at_shot.py, shot_context.py,
src/models/*.py, or src/viz/*.py. The StatsBomb freeze-frame path already
demonstrates this: it feeds a second data source through the same
shot_context functions with no source-specific feature code.

With a full season of club tracking data, the sample-size limit that shapes
this project (METHODOLOGY.md principle #2: 66 shots and 9 goals in the
public Metrica tracking sample) would go away: hundreds of shots, enough to
support the statistical claims this project is careful not to make on the
public sample. The power analysis in src/models/power.py quantifies how
many. The methodology is ready; the sample size is a constraint of using
public data, not a limitation of the approach itself.
"""
from __future__ import annotations

import pandas as pd


def extract_shot_events(match_id: str) -> pd.DataFrame:
    """Extract shot events from a club match. NOT IMPLEMENTED.

    Must return the same schema as metrica_loader.extract_shot_events().
    """
    raise NotImplementedError(
        "No club shot/event data source is connected. See this module's "
        "docstring for the extension path."
    )


def get_tracking_frame_at_shot(match_id: str, shot_frame: int) -> pd.DataFrame:
    """All players' positions at the moment of a club shot. NOT IMPLEMENTED.

    Must return the same schema as
    metrica_loader.get_tracking_frame_at_shot().
    """
    raise NotImplementedError(
        "No club tracking data source is connected. See this module's "
        "docstring for the extension path."
    )
