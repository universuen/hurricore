kill $(ps aux | grep -E "run.py" | grep -v grep | awk '{print $2}')
