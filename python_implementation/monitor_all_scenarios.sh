#!/bin/bash
# Monitor all running scenarios

echo "======================================"
echo "Running Scenarios Status"
echo "======================================"
echo ""
date
echo ""

# Function to check scenario status
check_scenario() {
    local name=$1
    local pid_file=$2
    local log_file=$3
    
    echo "[$name]"
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo "  Status: 🟢 RUNNING"
            echo "  PID: $PID"
            
            # Get process info
            RUNTIME=$(ps -o etimes= -p $PID | tr -d ' ')
            HOURS=$((RUNTIME / 3600))
            MINUTES=$(( (RUNTIME % 3600) / 60))
            CPU=$(ps -o %cpu= -p $PID | tr -d ' ')
            MEM=$(ps -o %mem= -p $PID | tr -d ' ')
            
            echo "  Runtime: ${HOURS}h ${MINUTES}m"
            echo "  CPU: ${CPU}%"
            echo "  Memory: ${MEM}%"
            
            # Show last line from log
            if [ -f "$log_file" ]; then
                echo "  Latest:"
                tail -3 "$log_file" | grep -E "Processing|Beta|%" | tail -1 | sed 's/^/    /'
            fi
        else
            echo "  Status: ❌ STOPPED (PID $PID is dead)"
            if [ -f "$log_file" ]; then
                echo "  Check log: $log_file"
            fi
        fi
    else
        echo "  Status: ⚪ NOT STARTED"
    fi
    echo ""
}

# Check each scenario
check_scenario "Riemannian" "nohup_riemannian.pid" "nohup_riemannian.out"
check_scenario "Wideband Incoherent" "nohup_wideband_incoherent.pid" "nohup_wideband_incoherent.out"
check_scenario "Wideband Coherent" "nohup_wideband_coherent.pid" "nohup_wideband_coherent.out"

echo "======================================"
echo "System Resources"
echo "======================================"
echo "CPU Cores: $(nproc)"
echo "Total Load: $(uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1)"
echo ""
free -h | head -2
echo ""

echo "======================================"
echo "Commands:"
echo "  Monitor live: ./monitor_all_scenarios.sh"
echo "  Tail logs:"
echo "    tail -f nohup_riemannian.out"
echo "    tail -f nohup_wideband_incoherent.out"
echo "======================================"
