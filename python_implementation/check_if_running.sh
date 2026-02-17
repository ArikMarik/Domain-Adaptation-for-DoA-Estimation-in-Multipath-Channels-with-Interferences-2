#!/bin/bash

echo "Checking if main.py is running..."
echo ""

# Check for Python process
PIDS=$(ps aux | grep "[p]ython.*main.py" | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "❌ No main.py process found!"
    echo "   Either it finished, crashed, or hasn't started yet."
    exit 1
fi

echo "✓ Found Python process(es) running main.py:"
echo ""

for PID in $PIDS; do
    echo "PID: $PID"
    echo "Command: $(ps -p $PID -o args=)"
    echo ""
    
    # Show CPU and memory
    echo "CPU & Memory usage:"
    ps -p $PID -o pid,pcpu,pmem,vsz,rss,etime,args | head -2
    echo ""
    
    # Check if it's using CPU (sign it's computing)
    CPU=$(ps -p $PID -o pcpu= | tr -d ' ')
    if (( $(echo "$CPU > 5" | bc -l) )); then
        echo "✓ Process is ACTIVELY COMPUTING (CPU: $CPU%)"
    else
        echo "⚠ Process seems IDLE or WAITING (CPU: $CPU%)"
        echo "  This might be normal if it's doing I/O or just starting"
    fi
    echo ""
    echo "----------------------------------------"
done

echo ""
echo "To monitor continuously, run: watch -n 2 ./check_if_running.sh"
