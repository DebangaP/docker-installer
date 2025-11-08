"""
Wrapper script to start KiteWS and run it until 3:30 PM IST
This script can be called manually or via API
"""

import sys
import os
import subprocess
import signal
import time
from datetime import datetime
from pytz import timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_kitews_until_330pm():
    """
    Start KiteWS.py and run it until 3:30 PM IST
    """
    ist = timezone('Asia/Kolkata')
    target_time = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    current_time = datetime.now(ist)
    
    # If current time is past 3:30 PM, set target to next day
    if current_time >= target_time:
        from datetime import timedelta
        target_time = target_time + timedelta(days=1)
    
    # Calculate seconds until 3:30 PM
    seconds_until_330 = (target_time - current_time).total_seconds()
    
    print(f"Starting KiteWS at {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"Will run until {target_time.strftime('%Y-%m-%d %H:%M:%S IST')} ({seconds_until_330:.0f} seconds)")
    
    # Start KiteWS as a subprocess
    kitews_path = os.path.join(os.path.dirname(__file__), 'KiteWS.py')
    process = subprocess.Popen(
        [sys.executable, kitews_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create new process group
    )
    
    try:
        # Wait until 3:30 PM or until process exits
        start_time = time.time()
        while time.time() - start_time < seconds_until_330:
            # Check if process is still running
            if process.poll() is not None:
                # Process has exited
                stdout, stderr = process.communicate()
                print(f"KiteWS process exited with code {process.returncode}")
                if stdout:
                    print(f"STDOUT: {stdout.decode()}")
                if stderr:
                    print(f"STDERR: {stderr.decode()}")
                return process.returncode
            
            # Sleep for 1 second before checking again
            time.sleep(1)
        
        # Time reached 3:30 PM, stop the process
        print(f"Reached 3:30 PM IST. Stopping KiteWS...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        # Wait for process to terminate gracefully
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
        
        print("KiteWS stopped successfully")
        return 0
        
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping KiteWS...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
        return 1
    except Exception as e:
        print(f"Error running KiteWS: {e}")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except:
            pass
        return 1

if __name__ == "__main__":
    exit_code = run_kitews_until_330pm()
    sys.exit(exit_code)

