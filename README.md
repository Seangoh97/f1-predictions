# F1 race prediction model

Gives each driver a chance of winning, finishing on the podium, scoring points and retiring in a
Grand Prix, using qualifying, recent form, lap data from practice, sprints and earlier races,
and the rain forecast.

## Setup

On a laptop (e.g. in Antigravity or any other editor), from the project folder:

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Python 3.10 or newer. On a Mac, LightGBM also needs `brew install libomp`.

On Google Colab, upload f1model.zip to My Drive and open `f1model_colab.ipynb`. It mounts Drive,
unzips the project there and runs everything from Drive, so nothing is lost when Colab resets.

The data folder already contains everything up to the 2026 United States Grand Prix, including the
OpenF1 lap cache, so you don't need to re-download it.

## Predict a race

After qualifying (prints win %, podium %, points %, DNF % and expected position):

    python predict.py 2026 19

With grid penalties:

    python predict.py 2026 19 --grid verstappen=15 hamilton=20

Rerun a past race as if it hadn't happened, with the real result alongside:

    python predict.py 2026 18 --replay

After each race weekend, run `python collect.py` so the new result is added to the training data.
`predict.py` fetches new OpenF1 practice data itself.

## Automatic updates on GitHub

`auto_update.py` does a full update in one go: new results, lap data and weather, a saved race
prediction once qualifying is in, championship odds after each race, scoring, and a fresh dashboard.
`.github/workflows/update.yml` runs it every 6 hours on GitHub and publishes the dashboard with
GitHub Pages. See `GITHUB_SETUP.md` for the setup steps.

## Dashboard

    python dashboard.py

Writes `dashboard.html`, an interactive page you can open in any browser. It covers the current and
previous season: predicted order and position odds for every race (completed races are replays using
only earlier data, the next race uses qualifying once it's in, later races are pre-weekend forecasts),
driver form and pace, championship odds, and how well the model has done. Rerun it after qualifying
or a race to refresh it. Charts load from the web, so the page needs an internet connection.

## Prediction log

Every live run of `predict.py` and `season.py` is saved in `logs/` with a timestamp. Replays of past
races are never saved. After a race weekend:

    python collect.py
    python score_log.py

For each race this uses the last prediction saved before the start, and compares it with the result
and with the grid-slot baseline. It also shows how the title odds moved race by race.

## Championship odds

    python season.py                  # current season, after the latest race
    python season.py 2025 --after 19  # how the odds looked after round 19 of 2025

Simulates every remaining race and sprint 20,000 times and prints each driver's title chance,
top-3 chance, expected final points with a 5-95% range, when the title is likely to be decided,
and the constructors' standings.

## Rebuild and test

    python collect.py          # results, qualifying, sprints from Jolpica-F1 (2018 onward)
    python collect_openf1.py   # practice/sprint/race laps and tyre stints from OpenF1 (2023 onward)
    python weather.py          # rain forecasts for each race from Open-Meteo (2022 onward)
    python tune.py             # picks pace model settings using 2024 only, writes tuned.py
    python evaluate.py         # finishing order test on 2025 and 2026 against the grid baseline
    python evaluate_probs.py   # probability test: tunes the simulator on 2024, scores 2025 and 2026
    python tune_season.py      # season simulator settings, one set per season, measured on 2022 onward

## Files

- `collect.py`, `collect_openf1.py`: data download with local caches.
- `pace.py`: turns raw laps into pace features (long-run pace, tyre degradation, one-lap pace,
  top speed, sprint pace, race pace).
- `features.py`: one row per driver per race. Rolling features are shifted by one race so a row never
  sees its own result.
- `evaluate.py`: retrains before every test race on all earlier races.
- `probability.py`: retirement model and race simulator (20,000 runs per race in `predict.py`).
- `evaluate_probs.py`: scores the probabilities and picks how much to blend them with grid-slot
  history. Settings are saved in `sim_params.json`.
- `dashboard.py`, `dashboard_template.html`: builds the dashboard page.
- `auto_update.py`, `.github/workflows/update.yml`: scheduled updates and publishing on GitHub.
- `logbook.py`, `score_log.py`: saving and scoring live predictions.
- `season.py`: championship simulator. A replay of a past season only uses settings measured on
  earlier seasons. Uses a pre-weekend pace model (no qualifying or practice
  yet) plus a lasting form shift per driver, so a good or bad run carries from race to race.
- `weather.py`: past races use what the forecast said at the time, not what actually fell, so the
  model learns from the same kind of input it gets before a real race.
- `results_v1.txt`, `results_v2.txt`: finishing order test results.
- `results_v3_probabilities.txt`: probability test results.

## Notes

- FastF1 now needs an F1 account login (set F1_USERNAME and F1_PASSWORD), so this project uses
  OpenF1 instead. OpenF1's free tier covers historical sessions; live data during a session needs
  the paid tier, but practice data is free from 30 minutes after each session ends.
- Pace features only exist from 2023. Earlier races are still used for training without them.
- From 2024 Jolpica often records retirements as "Retired" without a cause.
- `drv_starts` only counts starts from 2018.
- Wet race labels come from OpenF1 (intermediates or wets used), so only 2023 onward has them, and
  only about ten races were wet. The rain effect is a rough estimate.
- The season simulator assumes the current driver line-up stays for the rest of the year and that
  every remaining race has the usual ~11% chance of rain.
- Open-Meteo's free service sometimes returns busy errors; `weather.py` retries and saves as it goes.
