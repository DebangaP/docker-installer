"""
Cron Job Monitor
Periodically checks cron job status and optionally restarts failed jobs
Also checks KiteWS status by monitoring database for recent data
"""

import logging
from datetime import datetime
from common.CronJobStatusChecker import CronJobStatusChecker
import subprocess
import os
import sys

# Add parent directory to path to import KiteAccessToken functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def check_and_restart_failed_jobs(auto_restart: bool = False, check_kitews: bool = True):
    """
    Check all cron jobs and optionally restart failed ones.
    Also checks KiteWS status by monitoring database for recent data.
    
    Args:
        auto_restart: If True, automatically restart jobs that are expected to run but aren't
        check_kitews: If True, check and auto-start KiteWS if not running (default: True)
    """
    try:
        checker = CronJobStatusChecker()
        status = checker.get_all_jobs_status()
        
        current_hour = datetime.now().hour
        failed_jobs = []
        
        for job_name, job_status in status['jobs'].items():
            if job_status['status'] == 'expected_but_not_running':
                failed_jobs.append(job_name)
                logger.warning(f"Cron job '{job_name}' is expected to run but is not running")
        
        if failed_jobs:
            logger.warning(f"Found {len(failed_jobs)} failed cron jobs: {', '.join(failed_jobs)}")
            
            if auto_restart:
                logger.info("Auto-restart is enabled, attempting to restart failed jobs...")
                for job_name in failed_jobs:
                    try:
                        restart_job(job_name)
                    except Exception as e:
                        logger.error(f"Failed to restart job '{job_name}': {e}")
        else:
            logger.info("All expected cron jobs are running correctly")
        
        # Check KiteWS status by monitoring database for recent data
        kitews_status = None
        if check_kitews:
            try:
                from KiteAccessToken import check_and_auto_start_kitews
                kitews_status = check_and_auto_start_kitews()
                if kitews_status.get('auto_started'):
                    logger.info(f"KiteWS auto-started: {kitews_status.get('message')}")
                elif kitews_status.get('checked'):
                    logger.debug(f"KiteWS check: {kitews_status.get('message')}")
            except Exception as e:
                logger.error(f"Error checking KiteWS status: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_jobs': status['summary']['total'],
            'running': status['summary']['running'],
            'failed': status['summary']['expected_but_not_running'],
            'failed_jobs': failed_jobs,
            'kitews_status': kitews_status
        }
        
    except Exception as e:
        logger.error(f"Error checking cron jobs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def restart_job(job_name: str):
    """
    Restart a specific cron job
    
    Args:
        job_name: Name of the job to restart
    """
    checker = CronJobStatusChecker()
    job_config = checker.CRON_JOBS.get(job_name)
    
    if not job_config:
        raise ValueError(f"Job '{job_name}' not found in CRON_JOBS configuration")
    
    script_name = job_config['script']
    module_path = job_config.get('module_path')
    
    logger.info(f"Restarting job '{job_name}' (script: {script_name})")
    
    # Build the command
    if module_path:
        # Use module path if available
        cmd = ['python', '-m', module_path]
    else:
        # Use script path
        script_path = os.path.join('/app', script_name.split('/')[-1])
        cmd = ['python', script_path]
    
    # Start the process in background
    process = subprocess.Popen(
        cmd,
        cwd='/app',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )
    
    logger.info(f"Started job '{job_name}' with PID {process.pid}")
    return process.pid


if __name__ == '__main__':
    # Run status check (without auto-restart by default, but with KiteWS check enabled)
    result = check_and_restart_failed_jobs(auto_restart=False, check_kitews=True)
    if result:
        print(f"Status check completed: {result['running']} running, {result['failed']} failed")
        if result['failed_jobs']:
            print(f"Failed jobs: {', '.join(result['failed_jobs'])}")
        if result.get('kitews_status'):
            kitews = result['kitews_status']
            print(f"KiteWS status: {kitews.get('message', 'Unknown')}")
            if kitews.get('auto_started'):
                print("✓ KiteWS was auto-started successfully")

