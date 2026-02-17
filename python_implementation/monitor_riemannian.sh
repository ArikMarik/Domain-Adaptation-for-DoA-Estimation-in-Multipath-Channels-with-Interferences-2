#!/bin/bash
# Monitor script for Riemannian scenario run

echo "======================================"
echo "Riemannian Scenario Monitor"
echo "======================================"
echo ""
echo "Output file: nohup_riemannian.out"
echo "Press Ctrl+C to stop monitoring (process will continue running)"
echo ""

# Check if process is running
if [ -f nohup_riemannian.pid ]; then
    PID=$(cat nohup_riemannian.pid)
    if ps -p $PID > /dev/null; then
        echo "✓ Process is running (PID: $PID)"
        echo "  Started at: $(ps -o lstart= -p $PID)"
        echo ""
    else
        echo "✗ Process is not running (PID $PID is dead)"
        echo ""
    fi
else
    echo "⚠ No PID file found"
    echo ""
fi

# Tail the log file
if [ -f nohup_riemannian.out ]; then
    echo "Log output (last 50 lines, then following):"
    echo "--------------------------------------"
    tail -n 50 -f nohup_riemannian.out
else
    echo "⚠ Log file nohup_riemannian.out not found yet"
fi
