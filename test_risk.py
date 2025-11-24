import sys
import os
sys.path.append('Custom-App')

from app.api.services.risk_service import RiskService
import json

service = RiskService()
result = service.get_portfolio_risk_metrics()
print('Risk metrics result:')
print(json.dumps(result, indent=2, default=str))
