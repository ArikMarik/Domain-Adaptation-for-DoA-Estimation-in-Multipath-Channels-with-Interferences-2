#!/bin/bash
# Quick progress check script for Riemannian scenario

echo "======================================"
echo "Riemannian Scenario Progress Check"
echo "======================================"
echo ""

# Check if process is running
if [ -f nohup_riemannian.pid ]; then
    PID=$(cat nohup_riemannian.pid)
    if ps -p $PID > /dev/null; then
        echo "✓ Process is RUNNING (PID: $PID)"
        
        # Calculate runtime
        START_TIME=$(ps -o etimes= -p $PID | tr -d ' ')
        HOURS=$((START_TIME / 3600))
        MINUTES=$(( (START_TIME % 3600) / 60))
        echo "  Runtime: ${HOURS}h ${MINUTES}m"
        
        # Estimate remaining time (assuming 4.3 hours total)
        TOTAL_SECONDS=$((4 * 3600 + 18 * 60))  # 4h 18min
        REMAINING=$((TOTAL_SECONDS - START_TIME))
        if [ $REMAINING -gt 0 ]; then
            REM_HOURS=$((REMAINING / 3600))
            REM_MINUTES=$(( (REMAINING % 3600) / 60))
            echo "  Estimated remaining: ~${REM_HOURS}h ${REM_MINUTES}m"
        else
            echo "  Should be finishing soon!"
        fi
        echo ""
    else
        echo "✗ Process STOPPED (PID $PID is dead)"
        echo ""
    fi
else
    echo "✗ Process NOT STARTED (no PID file)"
    echo ""
fi

# Show last 20 lines of output
if [ -f nohup_riemannian.out ]; then
    echo "Latest output (last 20 lines):"
    echo "--------------------------------------"
    tail -n 20 nohup_riemannian.out
    echo "--------------------------------------"
    echo ""
    echo "Full log: nohup_riemannian.out"
    echo "To monitor live: ./monitor_riemannian.sh"
else
    echo "⚠ No log file found yet"
fi

# Check for results file
if [ -f results/results_t60-sweep_clean_riemannian.pkl ]; then
    SIZE=$(ls -lh results/results_t60-sweep_clean_riemannian.pkl | awk '{print $5}')
    echo ""
    echo "✓ Results file exists: $SIZE"
fi
