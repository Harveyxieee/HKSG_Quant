# Roostoo Round 1 Trading Bot

This repository contains our autonomous spot-trading bot for the SG vs HK University Web3 Quant Hackathon Round 1.

## What This Bot Does
- Trades only on Roostoo spot pairs through the official API.
- Uses a multi-horizon momentum and trend-quality ranking model.
- Applies market regime filtering, capped position sizing, stop-loss and trailing-stop controls.
- Runs continuously on AWS with internal request, trade, signal, and portfolio logs.

## Why This Fits The Round 1 Rules
- Fully autonomous: no manual frontend trading or manual signed API calls are required.
- Spot only: no leverage, shorting, arbitrage, market making, or HFT logic is used.
- Cloud-ready: designed to run continuously on one AWS EC2 instance.
- Audit-friendly: strategy logic, logs, and deployment flow are all traceable in-repo.

## Strategy Summary
The bot ranks liquid spot pairs using short and medium-horizon returns, volatility-adjusted trend strength, price efficiency, and position-in-range features. It only allocates capital when the market breadth and cross-sectional signal quality are supportive. Portfolio construction is long-only, exposure-capped, and rebalanced with turnover controls so the bot remains active without degenerating into noisy overtrading.

## Repository Contents
- `roostoo_bot.py`: main bot, API client, signal engine, execution logic, and runtime safety controls.
- `requirements.txt`: Python dependencies.
- `.env.example`: generic local template.
- `.env.testing.example`: testing account template.
- `.env.round1.live.example`: Round 1 live account template.
- `run.sh`: simple launch script for Linux environments.

## Runtime Logs
The bot writes the following files under `data/`:
- `data/logs/bot.log`
- `data/logs/requests.csv`
- `data/logs/trades.csv`
- `data/logs/portfolio.csv`
- `data/logs/signals.csv`
- `data/state.json`
- `data/history.json`
- `data/bot.lock`

## Local Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:
```env
ROOSTOO_API_KEY=<your_round1_key>
ROOSTOO_API_SECRET=<your_round1_secret>
BOT_NAME=<your_team_bot_name>
DRY_RUN=false
```

If you want to switch environments quickly:
```bash
cp .env.testing.example .env
# or
cp .env.round1.live.example .env
```

To test with non-live behavior, temporarily change:
```env
DRY_RUN=true
```

## AWS Round 1 Deployment
The hackathon statement specifies:
- Region: `ap-southeast-2`
- Instance: `t3.medium`
- Access: `Session Manager`
- Run exactly one EC2 instance

Once connected to the AWS instance terminal:
```bash
cd ~
git clone <your-repo-url>
cd <your-repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your Round 1 competition credentials, then launch inside `tmux`:
```bash
sudo dnf install -y tmux
tmux
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python roostoo_bot.py
```

Detach from `tmux` without stopping the bot:
```text
Ctrl+B, then D
```

Reattach later:
```bash
tmux attach
```

## Recommended Launch Flow
1. Put the Roostoo Round 1 API key and secret into `.env`.
2. Run once with `DRY_RUN=true` to confirm startup, logging, and environment loading.
3. Change `DRY_RUN=false`.
4. Start the bot in `tmux`.
5. Monitor `data/logs/bot.log` and `data/logs/trades.csv`.

## Submission Notes
This repository is structured to support the Round 1 judging criteria:
- Clear strategy implementation
- Clean and maintainable code structure
- Continuous AWS compatibility
- Traceable internal logs
- Open-source reviewability

## Important Operating Notes
- Use the Round 1 competition credentials, not the testing credentials, when trading live.
- Do not run more than one copy of the bot against the same account.
- Do not manually call signed trading APIs outside the bot.
- Keep commit history continuous and strategy changes traceable.
