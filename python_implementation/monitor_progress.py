import os
import time
import psutil

print("Monitoring Python process activity...\n")
print("If the process is stuck, CPU usage will be near 0%")
print("If it's computing, CPU should be 50-100%\n")
print("Press Ctrl+C to stop monitoring\n")

python_procs = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if 'python' in proc.info['name'].lower() and proc.info['cmdline']:
            if 'main.py' in ' '.join(proc.info['cmdline']):
                python_procs.append(proc)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not python_procs:
    print("No Python main.py process found!")
    print("Either it finished, crashed, or hasn't started yet.")
else:
    print(f"Found {len(python_procs)} Python process(es) running main.py\n")
    
    try:
        while True:
            for proc in python_procs:
                try:
                    cpu_percent = proc.cpu_percent(interval=1)
                    mem_info = proc.memory_info()
                    mem_mb = mem_info.rss / 1024 / 1024
                    
                    status = "✓ COMPUTING" if cpu_percent > 5 else "⚠ IDLE/WAITING"
                    
                    print(f"PID {proc.pid}: {status}")
                    print(f"  CPU: {cpu_percent:5.1f}% | Memory: {mem_mb:6.1f} MB")
                    print()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"PID {proc.pid}: Process ended")
                    python_procs.remove(proc)
            
            if not python_procs:
                print("All processes finished!")
                break
                
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
