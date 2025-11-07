This is a sample app to automatically pull all your profile, holdings, orders, trades, and margin data from **Zerodha Kite**.
To be noted that this is not a Production grade app.

**Steps to install**
1. Download and install **Docker/ Docker Desktop**
2. Down and install **git**
3. Go to the folder where you want to setup the code and run '**git clone https://github.com/DebangaP/docker-installer.git**'
4. **Add a .env file to the 'app' folder with KITE_API_KEY and KITE_API_SECRET**
5. **If you have anything running at ports 3001, 8000 and 5433, please change the docker-compose file accordingly**
6. run '**docker compose up -d**'
7. **Visit http://127.0.0.1:8000 to login to Kite and generate your Access Token**
   <img width="566" height="720" alt="image" src="https://github.com/user-attachments/assets/610d3acb-e13a-41d9-b7d9-8052fd8a0863" />

   Post Zerodha login
   <img width="1121" height="861" alt="image" src="https://github.com/user-attachments/assets/a7c7fc56-c0e0-4762-a817-b351951a814c" />

   <img width="1111" height="327" alt="image" src="https://github.com/user-attachments/assets/658eae55-fab1-4419-8c8f-b5bdbf9f75f8" />



9. Once Access Token is generated, its valid for an entire day
10. Visit **http://localhost:3001/d/my-dashboard/sample-dashboard?orgId=1&from=now-90d&to=now&timezone=browser** to see your Grafana Dashboard (user=admin/ password=adminpassword)

**NIFTY50 Ticks**

**Todays' Positions**

**TPO-based Derivative Suggestions**
<img width="1327" height="956" alt="image" src="https://github.com/user-attachments/assets/f0e9350f-979a-46b4-b159-d7218601a8be" />


**Latest Trades**

**Your Holdings**

**Click on any particular stock to see its daily Candlestick chart**

**Top Gainers and Losers for the last trading day**

