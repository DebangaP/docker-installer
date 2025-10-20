This project would automatically pull all your profile, holdings, orders, trades, and margin data from Kite.


**Steps to install**
1. Download and install **Docker/ Docker Desktop**
2. Down and install **git**
3. Go to the folder where you want to setup the code and run '**git clone https://github.com/DebangaP/docker-installer.git**'
4. **Add a .env file to the 'app' folder with KITE_API_KEY and KITE_API_SECRET**
5. run '**docker compose up -d**'
6. **Visit http://127.0.0.1:8000 to login to Kite and generate your Access Token**
7. Once Access Token is generated, its valid for an entire day
8. Visit **http://localhost:3001/d/my-dashboard/sample-dashboard?orgId=1&from=now-90d&to=now&timezone=browser** to see your Grafana Dashboard

