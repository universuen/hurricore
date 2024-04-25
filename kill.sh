# Kill all processes that match the pattern "train.*\.py"
kill $(ps aux | grep -E "train.*\.py" | grep -v grep | awk '{print $2}')
