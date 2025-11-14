"""
Cron Job Status Checker
Monitors the status of scheduled cron jobs and reports their running state
"""

import subprocess
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Try to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


class CronJobStatusChecker:
    """
    Checks the status of cron jobs defined in crontab.txt
    """
    
    # Define the cron jobs and their expected scripts
    CRON_JOBS = {
        'KiteWS': {
            'script': 'KiteWS.py',
            'module_path': 'kite.KiteWS',
            'schedule': 'Daily at 9:04 AM',
            'expected_during_hours': [9, 10, 11, 12, 13, 14, 15],  # 9 AM to 3:30 PM
            'log_file': None
        },
        'CalculateTPO': {
            'script': 'CalculateTPO.py',
            'module_path': 'market.CalculateTPO',
            'schedule': 'Daily at 9:16 AM',
            'expected_during_hours': None,
            'log_file': None
        },
        'RefreshHoldings': {
            'script': 'RefreshHoldings.py',
            'module_path': 'holdings.RefreshHoldings',
            'schedule': 'Every 5 minutes during market hours (9 AM - 3:30 PM)',
            'expected_during_hours': [9, 10, 11, 12, 13, 14, 15],
            'log_file': '/app/logs/holdings_refresh.log'
        },
        'KiteFetchOptions': {
            'script': 'KiteFetchOptions.py',
            'module_path': 'kite.KiteFetchOptions',
            'schedule': 'Every 5 minutes during market hours (9 AM - 3:30 PM)',
            'expected_during_hours': [9, 10, 11, 12, 13, 14, 15],
            'log_file': '/app/logs/options_fetch.log'
        },
        'InsertOHLC': {
            'script': 'InsertOHLC.py',
            'module_path': 'kite.InsertOHLC',
            'schedule': 'Every 30 minutes during market hours (9 AM - 3:30 PM)',
            'expected_during_hours': [9, 10, 11, 12, 13, 14, 15],
            'log_file': None
        },
        'RefreshSwingTrades': {
            'script': 'RefreshSwingTrades.py',
            'module_path': 'stocks.RefreshSwingTrades',
            'schedule': 'Every 30 minutes during market hours (9 AM - 3:30 PM)',
            'expected_during_hours': [9, 10, 11, 12, 13, 14, 15],
            'log_file': '/app/logs/swing_trades_refresh.log'
        },
        'RefreshMFNAV': {
            'script': 'RefreshMFNAV.py',
            'module_path': 'holdings.RefreshMFNAV',
            'schedule': 'Daily at 6:00 PM (after market close)',
            'expected_during_hours': [18],
            'log_file': '/app/logs/mf_nav_refresh.log'
        },
        'AccumDistAnalysis': {
            'script': 'AccumDistAnalysis.py',
            'module_path': 'market.AccumDistAnalysis',
            'schedule': 'Daily at 4:30 PM (after market close)',
            'expected_during_hours': [16],
            'log_file': '/app/logs/accum_dist.log'
        }
    }
    
    def __init__(self):
        self.app_dir = '/app'
    
    def check_process_running(self, script_name: str) -> Dict:
        """
        Check if a process is running by script name
        
        Args:
            script_name: Name of the script file (e.g., 'KiteWS.py')
            
        Returns:
            Dictionary with status information
        """
        try:
            # Try using pgrep first (more reliable)
            try:
                result = subprocess.run(
                    ['pgrep', '-f', script_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
                    if pids:
                        # Get process details using psutil if available
                        process_info = []
                        for pid in pids:
                            try:
                                pid_int = int(pid)
                                if PSUTIL_AVAILABLE:
                                    proc = psutil.Process(pid_int)
                                    process_info.append({
                                        'pid': pid_int,
                                        'start_time': datetime.fromtimestamp(proc.create_time()).isoformat(),
                                        'cpu_percent': proc.cpu_percent(interval=0.1),
                                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                                        'status': proc.status()
                                    })
                                else:
                                    # Fallback: just store PID
                                    process_info.append({
                                        'pid': pid_int,
                                        'start_time': None,
                                        'cpu_percent': None,
                                        'memory_mb': None,
                                        'status': 'unknown'
                                    })
                            except (ValueError, Exception) as e:
                                if PSUTIL_AVAILABLE and isinstance(e, (psutil.NoSuchProcess, psutil.AccessDenied)):
                                    logger.debug(f"Process {pid} no longer exists or access denied: {e}")
                                else:
                                    logger.debug(f"Error processing PID {pid}: {e}")
                                continue
                        
                        if process_info:
                            return {
                                'running': True,
                                'pids': [p['pid'] for p in process_info],
                                'process_info': process_info,
                                'count': len(process_info)
                            }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # pgrep not available, try alternative method
                pass
            
            # Alternative: Use psutil to search for processes (if available)
            if PSUTIL_AVAILABLE:
                try:
                    matching_processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                        try:
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any(script_name in str(arg) for arg in cmdline):
                                matching_processes.append({
                                    'pid': proc.info['pid'],
                                    'start_time': datetime.fromtimestamp(proc.info['create_time']).isoformat(),
                                    'cmdline': ' '.join(cmdline[:3])  # First 3 args for display
                                })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    if matching_processes:
                        return {
                            'running': True,
                            'pids': [p['pid'] for p in matching_processes],
                            'process_info': matching_processes,
                            'count': len(matching_processes)
                        }
                except Exception as e:
                    logger.debug(f"Error using psutil to check process: {e}")
            
            return {
                'running': False,
                'pids': [],
                'process_info': [],
                'count': 0
            }
            
        except Exception as e:
            logger.error(f"Error checking process for {script_name}: {e}")
            return {
                'running': False,
                'error': str(e),
                'pids': [],
                'process_info': [],
                'count': 0
            }
    
    def get_log_file_info(self, log_file: Optional[str]) -> Dict:
        """
        Get information about log file (last modified time, size)
        
        Args:
            log_file: Path to log file
            
        Returns:
            Dictionary with log file information
        """
        if not log_file or not os.path.exists(log_file):
            return {
                'exists': False,
                'last_modified': None,
                'size_bytes': 0,
                'size_mb': 0
            }
        
        try:
            stat = os.stat(log_file)
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            size_bytes = stat.st_size
            
            return {
                'exists': True,
                'last_modified': last_modified.isoformat(),
                'last_modified_readable': last_modified.strftime('%Y-%m-%d %H:%M:%S'),
                'size_bytes': size_bytes,
                'size_mb': round(size_bytes / 1024 / 1024, 2),
                'age_seconds': (datetime.now() - last_modified).total_seconds()
            }
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
            return {
                'exists': False,
                'error': str(e),
                'last_modified': None,
                'size_bytes': 0
            }
    
    def is_expected_to_run_now(self, job_name: str, current_hour: int) -> bool:
        """
        Check if a job is expected to be running at the current hour
        
        Args:
            job_name: Name of the cron job
            current_hour: Current hour (0-23)
            
        Returns:
            True if job is expected to run, False otherwise
        """
        job_config = self.CRON_JOBS.get(job_name)
        if not job_config:
            return False
        
        expected_hours = job_config.get('expected_during_hours')
        if expected_hours is None:
            # Job doesn't have specific hours, check based on schedule
            return True  # Assume it might run
        
        return current_hour in expected_hours
    
    def get_all_jobs_status(self) -> Dict:
        """
        Get status of all cron jobs
        
        Returns:
            Dictionary with status of all jobs
        """
        current_hour = datetime.now().hour
        current_time = datetime.now()
        
        jobs_status = {}
        
        for job_name, job_config in self.CRON_JOBS.items():
            script_name = job_config['script']
            
            # Check if process is running
            process_status = self.check_process_running(script_name)
            
            # Get log file info if available
            log_info = self.get_log_file_info(job_config.get('log_file'))
            
            # Determine overall status
            is_running = process_status.get('running', False)
            is_expected = self.is_expected_to_run_now(job_name, current_hour)
            
            # Status determination
            if is_running:
                status = 'running'
                status_color = 'green'
            elif is_expected:
                status = 'expected_but_not_running'
                status_color = 'red'
            else:
                status = 'not_expected'
                status_color = 'gray'
            
            # Determine last run timestamp
            # Priority: 1) Process start time (if running), 2) Log file last modified, 3) None
            last_run_timestamp = None
            last_run_timestamp_readable = None
            
            if is_running and process_status.get('process_info'):
                # Use the earliest process start time (in case multiple instances)
                process_infos = process_status.get('process_info', [])
                if process_infos:
                    start_times = []
                    for proc_info in process_infos:
                        if proc_info.get('start_time'):
                            try:
                                if isinstance(proc_info['start_time'], str):
                                    start_times.append(datetime.fromisoformat(proc_info['start_time']))
                                else:
                                    start_times.append(proc_info['start_time'])
                            except (ValueError, TypeError):
                                pass
                    
                    if start_times:
                        # Use the earliest start time (most recent run)
                        earliest_start = min(start_times)
                        last_run_timestamp = earliest_start.isoformat()
                        last_run_timestamp_readable = earliest_start.strftime('%Y-%m-%d %H:%M:%S')
            
            # If no process start time, use log file last modified time
            if not last_run_timestamp and log_info and log_info.get('exists') and log_info.get('last_modified'):
                try:
                    if isinstance(log_info['last_modified'], str):
                        last_run_timestamp = log_info['last_modified']
                        last_run_timestamp_readable = log_info.get('last_modified_readable')
                    else:
                        last_run_timestamp = log_info['last_modified'].isoformat()
                        last_run_timestamp_readable = log_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')
                except (AttributeError, TypeError):
                    pass
            
            jobs_status[job_name] = {
                'name': job_name,
                'script': script_name,
                'schedule': job_config['schedule'],
                'status': status,
                'status_color': status_color,
                'is_running': is_running,
                'is_expected_now': is_expected,
                'process_count': process_status.get('count', 0),
                'pids': process_status.get('pids', []),
                'process_info': process_status.get('process_info', []),
                'log_file': job_config.get('log_file'),
                'log_info': log_info,
                'last_run_timestamp': last_run_timestamp,
                'last_run_timestamp_readable': last_run_timestamp_readable,
                'last_check': current_time.isoformat()
            }
        
        return {
            'timestamp': current_time.isoformat(),
            'current_hour': current_hour,
            'jobs': jobs_status,
            'summary': {
                'total': len(jobs_status),
                'running': sum(1 for j in jobs_status.values() if j['is_running']),
                'expected_but_not_running': sum(1 for j in jobs_status.values() if j['status'] == 'expected_but_not_running'),
                'not_expected': sum(1 for j in jobs_status.values() if j['status'] == 'not_expected')
            }
        }
    
    def get_job_status(self, job_name: str) -> Optional[Dict]:
        """
        Get status of a specific cron job
        
        Args:
            job_name: Name of the cron job
            
        Returns:
            Dictionary with job status or None if job not found
        """
        all_status = self.get_all_jobs_status()
        return all_status['jobs'].get(job_name)

