# Methodology and data principles

This project asks whether defensive context (pitch control, defenders in the shooting lane, goalkeeper position) changes how good a shot really was, beyond what distance and angle already capture. It uses only public data, and two of its samples are small enough that what the data can support shapes every result. This document sets out the principles the analysis follows, the results as they stand, and the pitfalls a future change could easily reintroduce.

## Principles

1. **No proprietary or fabricated club data.** Only the public datasets described in the README are used. No placeholder shot or tracking data is generated and presented as real. Synthetic data appears only inside unit tests, where it is labelled as such.

2. **No general claim that the enhanced model beats basic xG on the tracking sample.** The pitch-control model can only be fit on the Metrica sample: 68 shots and 10 goals across 3 games, and 66 shots with 9 goals after excluding one penalty and one shot whose event and tracking data are out of sync. For a binary outcome the 9 goals are what limit the analysis. Comparisons on this sample describe specific shots ("on these N shots, the two models disagreed most on X") and never make claims that would need statistical power the sample does not have. `evaluate.summary_statistics` enforces this in code by withholding log-loss and AUC below 20 goals.

3. **Full pitch control is not computed from StatsBomb data, although that data does include player positions.** No NWSL match in the release has a `three-sixty/` file (0 of 137). However, 3,493 of 3,530 NWSL shots (99%) carry a `shot.freeze_frame` inside the shot event, listing every player in camera view at the instant of the shot with a teammate or opponent flag and a playing position. The goalkeeper appears in 99% of frames, and the median frame shows 9 opponents.

   - **Computed from freeze frames:** defender-position features, namely defenders in the shooting cone and goalkeeper distance from the centre of goal. On StatsBomb data these can be fit against 299 goals instead of 9.
   - **Not computed from freeze frames:** Spearman pitch control and whole-pitch Voronoi. A freeze frame is one instant with no velocities and only the players in camera view, so time-to-intercept is undefined and a full-pitch tessellation would be built from partial data. Both stay Metrica-only.
   - **Coverage is recorded, not assumed.** `statsbomb_loader` stores the number of opponents in each frame, so any undercount is visible.

4. **Only the configured season, verified.** The StatsBomb sample is the 2023 NWSL season and nothing else. The season is checked against the release's own competitions listing before any data is read (`statsbomb_loader.verify_season_available`), and the pipeline raises an error rather than substituting another season.

5. **Credit real sources.** The pitch-control model adapts Spearman (2018) (https://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf) as implemented in the "Friends of Tracking" reference code (https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking). It is credited as an adaptation and simplification in both the module docstring and the README, and is not presented as original work.

## Data sources

See the README for full detail.

- **Metrica Sports open sample data** (`data/external/metrica/`): 3 matches of continuous tracking, with 68 shots and 10 goals. The only source for pitch-control features. Games 1 and 2 are Metrica CSV; game 3 is FIFA EPTS, read with kloppy.
- **StatsBomb open data, 2023 NWSL** (competition_id=49, season_id=107, `data/external/statsbomb/`): event and shot data with freeze frames. Used to validate the baseline and to fit the large-sample defender-context model.

`scripts/download_data.sh` downloads both and does not depend on anything else being present on the machine.

## Code conventions

- Python 3.11+, type hints on all public functions, numpy-style docstrings.
- One coordinate convention everywhere after `src/data/cleaning.py`: meters, origin at the centre of the pitch, x positive toward the attacking goal.
- Feature functions in `src/features/` are pure functions on cleaned DataFrames and do not load data themselves.
- Every feature function has a test with a hand-verifiable expected value wherever the maths allows it. `tests/test_geometry.py` shows the pattern: a shot from the centre of the goal line has a known distance and the maximum possible angle.
- `pytest` and `ruff check .` must pass before a change is finished.

## Current status

Everything listed under "What's implemented" in the README is done. `python -m src.pipeline` runs end to end, `python -m scripts.render_figures` writes eight figures to `reports/figures/`, and `streamlit run app/streamlit_app.py` serves the interactive comparison.

The one deliberate exception is `src/data/club_loader.py`, a documented stub describing what connecting a club's own tracking data would involve. It is meant to raise an error.

### Two results on two samples, kept separate

1. **Metrica, 66 shots and 9 goals: pitch control shows no effect, and the test is uninformative.** The two models' predictions correlate at 0.9993, and the 95% bootstrap interval for the `pitch_control` coefficient is [-0.57, +0.33], which crosses zero. The power analysis puts this sample at about 9% power for a moderate effect, so the null result says nothing about the hypothesis either way. It should not be read as "pitch control doesn't matter".
2. **StatsBomb freeze frames, 3,490 shots and 299 goals: defender context has a real effect.** `goalkeeper_distance` is +0.34 [+0.24, +0.44] and `defenders_in_cone` is -0.32 [-0.48, -0.16]. Both intervals exclude zero, and together the features raise AUC from 0.7215 to 0.7553.

The accurate summary is that defensive context matters and the tracking sample was too small to show it. That is different from "pitch control works", because result 2 does not test pitch control, which needs velocities. It is also different from "pitch control fails", because result 1 does not have the power to say so.

The lane-control feature is also partly collinear with shot angle (r = 0.35) by construction, because the lane triangle is the angle. That flaw should be fixed before drawing conclusions about pitch control specifically.

### Pitfalls a future change could reintroduce

- **Playing direction.** Metrica's coordinates are fixed to the pitch. The teams attack opposite goals, swap ends at halftime, and do not attack the same way in every sample game. `geometry.py` places the attacking goal at x = +52.5, so coordinates must go through `cleaning.normalize_playing_direction` first, and each shot and its tracking frame must be rotated together. `metrica_loader.get_normalized_tracking_frame_at_shot` does both steps at once so they cannot drift apart. Skipping this measures distance to the wrong goal, and the numbers still look plausible.
- **Direction is inferred from the goalkeeper**, not from the team's average position. The average position was too weak a signal and got game 1 backwards. A check raises an error if both teams come out attacking the same goal.
- **Control at the shot point is degenerate.** It is 1.0 for 65 of 66 shots because the shooter is standing there. The model uses lane control instead.
- **The two inference methods suit different sample sizes.** `src/models/uncertainty.py` uses the bootstrap, which makes no distributional assumption, and is used on the 9-goal sample. `src/models/power.py` uses Wald standard errors from the Fisher information and is calibrated only on the large sample, where asymptotic normality is reasonable. Bootstrap resampling is stratified by outcome, because plain resampling of 9 goals produces some draws with no goals at all.
- **Testing the Streamlit app needs more than `AppTest`.** `AppTest` runs inside pytest, where the repo root is already on `sys.path` and dataframes are never serialized with Arrow, so it missed two bugs that broke the real `streamlit run`. `tests/test_app.py` therefore also runs the script as a subprocess from an unrelated directory.

`pytest -m "not integration"` and `ruff check .` are a quick way to confirm an environment is set up correctly.

## When in doubt

If an analysis would need data, statistical power or claims the project does not have, especially anything amounting to "prove the enhanced model is better" on the tracking sample, it should be flagged rather than overstated. Every claim here is meant to be checkable, true and appropriately hedged.
