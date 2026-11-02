# GitHub deployment

Repository: https://github.com/Seangoh97/f1-predictions

Dashboard: https://seangoh97.github.io/f1-predictions/

The Update predictions workflow runs every six hours, at 00:17, 06:17, 12:17 and 18:17 UTC (08:17, 14:17, 20:17 and 02:17 Malaysia time). It also runs on code pushes and can be started manually. GitHub can delay scheduled starts.

Each successful update collects results, lap data and weather, saves predictions, scores past predictions and publishes dashboard.html as the website index. Only the dashboard is included in the Pages artifact. The public repository also holds code, datasets and F1 prediction history.

Historical completed-session tables are reused. API caches are stored with Actions cache rather than committed. Current-season session lists are refreshed each run. Collection automatically includes the current year.

Pages source must be GitHub Actions. Workflow permissions are specified per job; the repository-wide default does not need write access. No personal access token is required.

To publish the saved dashboard without downloading data, open Actions → Update predictions → Run workflow and uncheck refresh. Leave it checked for a full update.

If a refresh fails, the previously published site remains available. Inspect the failed step in Actions. Free upstream data services may be unavailable temporarily. GitHub may disable scheduled workflows after 60 days of repository inactivity; re-enable in Actions if necessary. This is unattended operation, not a guarantee that maintenance will never be needed.

Race predictions use qualifying order; grid penalties require a manual grid override with predict.py. Model parameter retuning remains a separate maintenance task. Pull the latest repository before running locally, because Actions commits updated data and logs.
