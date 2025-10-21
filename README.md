This is a sample app to automatically pull all your profile, holdings, orders, trades, and margin data from Kite.
To be noted that this is not a Production grade app.

**Steps to install**
1. Download and install **Docker/ Docker Desktop**
2. Down and install **git**
3. Go to the folder where you want to setup the code and run '**git clone https://github.com/DebangaP/docker-installer.git**'
4. **Add a .env file to the 'app' folder with KITE_API_KEY and KITE_API_SECRET**
5. run '**docker compose up -d**'
6. **Visit http://127.0.0.1:8000 to login to Kite and generate your Access Token**
7. Once Access Token is generated, its valid for an entire day
8. Visit **http://localhost:3001/d/my-dashboard/sample-dashboard?orgId=1&from=now-90d&to=now&timezone=browser** to see your Grafana Dashboard (user=admin/ password=adminpassword)

**NIFTY50 Ticks**
**<img width="706" height="507" alt="image" src="https://github.com/user-attachments/assets/b2866788-9859-4005-81c5-63c6228c176b" />

**Todays' Positions**
**<img width="930" height="180" alt="image" src="https://github.com/user-attachments/assets/f187eca5-32f2-400d-b5c1-cbf4f65d3c77" />

**Latest Trades**
**<img width="952" height="187" alt="image" src="https://github.com/user-attachments/assets/be7a4b30-3045-4ed5-b907-3d0a006f88a8" />

**Your Holdings**
<img width="770" height="127" alt="image" src="https://github.com/user-attachments/assets/7e61a635-7272-4e7d-bd0a-44d400deb4fe" />

**Click on any particular stock to see its daily Candlestick chart**
<img width="1080" height="386" alt="image" src="https://github.com/user-attachments/assets/333ce042-4ad8-4d6d-a868-64fdb56eebce" />

**Top Gainers and Losers for the last trading day**
<img width="1872" height="465" alt="image" src="https://github.com/user-attachments/assets/c5093877-1cbd-4146-922a-db92065f9e14" />

