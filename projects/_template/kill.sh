kill $(ps aux | grep -E "*.py" | grep -v grep | awk '{print $2}')
