# Running the model on GitHub

Once set up, GitHub runs the model every 6 hours on its own servers and publishes the dashboard as a
website. After qualifying it saves a race prediction, after each race it adds the result, scores the
saved predictions and updates the championship odds. You don't need to run anything yourself.

It's free as long as the repository is public. (GitHub Pages on a private repository needs a paid plan.)

## 1. Create the repository

1. Sign in at github.com (or create a free account).
2. Click **+** at the top right, then **New repository**.
3. Name it, for example `f1-predictions`. Choose **Public**. Leave every box unticked. Click **Create repository**.

## 2. Upload the project

1. Unzip `f1model-github.zip` on your computer.
2. On the new repository's page, click **uploading an existing file**.
3. Open the unzipped `f1model` folder, select everything inside it (not the folder itself) and drag it
   onto the page. Wait for all the files to appear, then click **Commit changes**.

Browsers often skip the hidden `.github` folder, so add the automation file by hand:

4. Click **Add file**, then **Create new file**.
5. In the name box type exactly `.github/workflows/update.yml` (the slashes create the folders).
6. Paste in the contents of `update.yml` from the zip (it's also in `.github/workflows/`).
7. Click **Commit changes**.

## 3. Turn on the website and permissions

1. Go to **Settings**, then **Pages** (left menu). Under **Source**, choose **GitHub Actions**.
2. Go to **Settings**, then **Actions**, then **General**. Under **Workflow permissions**, choose
   **Read and write permissions** and click **Save**.

## 4. First run

1. Open the **Actions** tab. If asked, click the button to enable workflows.
2. Click **Update predictions** on the left, then **Run workflow**, then the green **Run workflow** button.
3. The first run takes about 45 minutes, because it downloads all the lap data from OpenF1 (later
   runs take about 5 minutes).
4. When both jobs show a green tick, your dashboard is live at
   `https://YOUR-USERNAME.github.io/f1-predictions/`
   (the link also appears in **Settings**, then **Pages**).

## Good to know

- **Schedule:** runs every 6 hours. GitHub sometimes starts scheduled runs late when its servers are
  busy. You can always start one by hand with **Run workflow**.
- **Grid penalties:** the automatic race prediction uses the qualifying order. If penalties change
  the grid, run `python predict.py SEASON ROUND --grid driver=slot` yourself, or start a run after
  the official grid is out.
- **If a run fails:** open it in the **Actions** tab to see which step stopped. A free data service
  being briefly down is the usual cause, and the next run normally fixes itself. If GitHub ever
  pauses the schedule after a long quiet spell, re-enable it from the **Actions** tab.
- **Working on your laptop or Colab as well:** the repository is now the master copy, since GitHub
  saves new data and logs into it. Download or `git pull` the latest version before running
  anything locally, so the two copies don't drift apart.
- **New season:** once in the winter break, run `python tune_season.py` locally and upload the new
  `season_params.json`, so the season simulator learns from the year just finished.
