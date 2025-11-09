# AWS EC2 Deployment Plan for Market Data Application

## Overview
Deploy the Docker-based market data application to AWS EC2 instance that runs during market hours (9am-3:30pm IST) and syncs data daily to laptop via incremental database dumps.

## Phase 1: AWS Infrastructure Setup

### 1.1 EC2 Instance Configuration
- **Instance Type**: t3.medium (2 vCPU, 4GB RAM) - sufficient for Docker containers
  - Cost: ~$30/month (can use t3.small ~$15/month if budget is tight)
  - Region: ap-south-1 (Mumbai) for lowest latency to Indian markets
- **Storage**: 30GB gp3 EBS volume (for PostgreSQL data)
- **AMI**: Amazon Linux 2023 or Ubuntu 22.04 LTS
- **Security Group**: 
  - Inbound: SSH (22) from your IP, HTTP (8000) from your IP only
  - Outbound: All traffic

### 1.2 IAM Role & Permissions
- Create IAM role for EC2 with S3 access (for backup storage)
- Attach policy: `AmazonS3FullAccess` or custom policy for specific bucket

### 1.3 S3 Bucket Setup
- Create S3 bucket: `market-data-backups-<your-name>`
- Enable versioning for backup retention
- Lifecycle policy: Delete backups older than 30 days
- Cost: ~$0.023/GB/month (minimal for daily dumps)

## Phase 2: EC2 Instance Setup

### 2.1 Initial Server Configuration
- Install Docker and Docker Compose
- Install PostgreSQL client tools (pg_dump, pg_restore)
- Install AWS CLI v2
- Configure timezone: `Asia/Kolkata`
- Set up swap space (2GB) for memory management

### 2.2 Application Deployment
- Clone/upload application code to `/opt/market-app`
- Copy `docker-installer/` directory structure
- Build Docker images on EC2
- Configure environment variables (`.env` file)
- Set up Docker volumes for persistent data

### 2.3 Database Configuration
- Modify `docker-compose.yml` for production:
  - Remove port mappings (except for local access)
  - Add volume mounts for PostgreSQL data persistence
  - Configure PostgreSQL for better performance
- Initialize database schema using existing `Schema.sql`

## Phase 3: Scheduled Execution (Market Hours)

### 3.1 Systemd Service for Docker Compose
- Create systemd service: `/etc/systemd/system/market-app.service`
- Service starts Docker Compose at 8:55 AM IST
- Service stops Docker Compose at 3:35 PM IST
- Auto-restart on failure during market hours

### 3.2 Cron Jobs for Scheduling
- Use systemd timers or cron for:
  - Start: 8:55 AM IST daily (Monday-Friday)
  - Stop: 3:35 PM IST daily (Monday-Friday)
  - Skip weekends and market holidays

### 3.3 Application Scripts Modification
- Modify `start.sh` to handle graceful shutdown
- Update `KiteWS.py` to stop at 3:30 PM IST
- Ensure all background scripts respect market hours

## Phase 4: Daily Incremental Database Sync

### 4.1 Backup Script on EC2
- Create script: `/opt/market-app/scripts/daily_backup.sh`
- Uses `pg_dump` with `--format=custom` for incremental efficiency
- Dumps only data from last 24 hours (using `run_date >= CURRENT_DATE - 1`)
- Compresses dump file
- Uploads to S3: `s3://market-data-backups/daily/YYYY-MM-DD.dump`
- Runs daily at 4:00 PM IST (after market closes)

### 4.2 Incremental Sync Strategy
- **Option A**: Daily incremental dump (recommended)
  - Dump tables filtered by `run_date >= CURRENT_DATE - 1`
  - Tables: `ticks`, `holdings`, `positions`, `market_structure`, `rt_intraday_price`, etc.
  - Use `pg_dump --table=my_schema.table_name --data-only --where="run_date >= CURRENT_DATE - 1"`

- **Option B**: Full schema + incremental data
  - Weekly full dump (Sunday)
  - Daily incremental dumps (Monday-Friday)

### 4.3 Backup Script Implementation
```bash
#!/bin/bash
# /opt/market-app/scripts/daily_backup.sh
DATE=$(date +%Y-%m-%d)
BACKUP_DIR="/opt/market-app/backups"
S3_BUCKET="s3://market-data-backups/daily"

# Create backup directory
mkdir -p $BACKUP_DIR

# Dump incremental data (last 24 hours)
pg_dump -h localhost -U postgres -d mydb \
  --format=custom \
  --schema=my_schema \
  --table=my_schema.ticks \
  --table=my_schema.holdings \
  --table=my_schema.positions \
  --table=my_schema.market_structure \
  --table=my_schema.rt_intraday_price \
  --data-only \
  --file="$BACKUP_DIR/incremental_$DATE.dump"

# Compress
gzip "$BACKUP_DIR/incremental_$DATE.dump"

# Upload to S3
aws s3 cp "$BACKUP_DIR/incremental_$DATE.dump.gz" \
  "$S3_BUCKET/incremental_$DATE.dump.gz"

# Cleanup local backup (keep last 3 days)
find $BACKUP_DIR -name "*.dump.gz" -mtime +3 -delete
```

## Phase 5: Laptop-Side Sync Scripts

### 5.1 Sync Script for Laptop
- Create script: `sync_from_aws.sh` (Windows) or `sync_from_aws.py` (cross-platform)
- Downloads latest incremental dump from S3
- Restores to local PostgreSQL using `pg_restore`
- Handles conflicts (ON CONFLICT DO UPDATE)
- Logs sync status

### 5.2 Python Sync Script (Recommended)
- Create: `scripts/sync_from_aws.py`
- Uses boto3 for S3 access
- Downloads latest daily backup
- Restores using psycopg2 or subprocess pg_restore
- Updates sync timestamp file

### 5.3 Manual Sync Process
- User runs sync script on laptop when needed
- Script checks for new backups since last sync
- Downloads and restores incrementally
- Reports sync statistics

## Phase 6: Cost Optimization

### 6.1 EC2 Instance Scheduling
- Use AWS Instance Scheduler or Lambda to:
  - Start instance at 8:50 AM IST (Monday-Friday)
  - Stop instance at 4:00 PM IST (Monday-Friday)
  - Keep stopped on weekends
- **Savings**: ~60% cost reduction (~$12/month instead of $30)

### 6.2 EBS Volume Optimization
- Use gp3 instead of gp2 (20% cheaper)
- Enable EBS snapshot lifecycle (delete after 7 days)
- Use smaller volume size (30GB sufficient)

### 6.3 S3 Storage Optimization
- Enable S3 Intelligent-Tiering
- Compress backups (gzip)
- Delete old backups after 30 days

### 6.4 Estimated Monthly Costs
- EC2 t3.medium (scheduled): ~$12/month
- EBS 30GB gp3: ~$3/month
- S3 storage (5GB): ~$0.12/month
- Data transfer: ~$1/month
- **Total: ~$16-20/month**

## Phase 7: Monitoring & Maintenance

### 7.1 CloudWatch Monitoring
- Set up CloudWatch alarms for:
  - EC2 instance status
  - Disk space usage
  - Application health (HTTP endpoint)
- Email notifications for failures

### 7.2 Logging
- Configure CloudWatch Logs for Docker containers
- Log rotation for application logs
- Backup script logs to S3

### 7.3 Health Checks
- Create health check endpoint in FastAPI
- Monitor application availability during market hours
- Auto-restart on failure

## Phase 8: Security & Access

### 8.1 SSH Access
- Use SSH key pairs (no password)
- Restrict SSH to your IP only
- Use AWS Systems Manager Session Manager (optional, more secure)

### 8.2 Database Security
- Change default PostgreSQL passwords
- Use environment variables for secrets
- Restrict database access to localhost only

### 8.3 S3 Access
- Use IAM roles (not access keys)
- Enable S3 bucket encryption
- Use bucket policies for access control

## Implementation Files to Create/Modify

1. **EC2 Setup Script**: `scripts/ec2_setup.sh`
2. **Systemd Service**: `scripts/market-app.service`
3. **Backup Script**: `scripts/daily_backup.sh`
4. **Sync Script**: `scripts/sync_from_aws.py`
5. **Modified docker-compose.yml**: Production version
6. **Environment Config**: `.env.production`
7. **Start/Stop Scripts**: `scripts/start_market_hours.sh`, `scripts/stop_market_hours.sh`

## Testing Checklist

- [ ] EC2 instance starts/stops on schedule
- [ ] Docker containers run during market hours
- [ ] Application collects data correctly
- [ ] Daily backups are created and uploaded to S3
- [ ] Laptop sync script downloads and restores data
- [ ] Incremental sync handles conflicts correctly
- [ ] Cost monitoring shows expected expenses
- [ ] Health checks work correctly
- [ ] Logs are accessible and useful

## Rollback Plan

- Keep local Docker setup functional
- Maintain database backups before migration
- Test sync process before full cutover
- Can revert to local setup if needed

## Notes

- Market hours: 9:00 AM - 3:30 PM IST (Monday-Friday)
- Application should start at 8:55 AM and stop at 3:35 PM
- Daily backups run at 4:00 PM IST
- Sync can be run manually from laptop anytime
- Estimated monthly cost: $16-20 with instance scheduling

