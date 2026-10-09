#!/usr/bin/env bash
# Kill a tmux session and any e4b_eval.py processes it left behind (pattern matching runs in this file,
# so it never matches the ssh command line that launched it).
tmux kill-session -t "$1" 2>/dev/null
sleep 2
pkill -f "scripts/e4b_eval.py" 2>/dev/null
sleep 2
pgrep -fc "scripts/e4b_eval.py" || true
