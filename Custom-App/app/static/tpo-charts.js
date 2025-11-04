// TPO Chart rendering using Chart.js
class TPOChartRenderer {
    constructor() {
        this.preMarketChart = null;
        this.realTimeChart = null;
        this.currentAnalysisDate = null;
        this.isLiveMode = true;
        this.init();
    }

    init() {
        this.createChartContainers();
        this.setupEventListeners();
        this.loadTPOData();
        this.loadMarginData();
        this.loadAnalysisDateConfig();
    }

    createChartContainers() {
        // Charts are now handled by matplotlib-generated images
        // No need to create canvas containers
    }

    setupEventListeners() {
        // Backtest button
        document.getElementById('backtestBtn').addEventListener('click', () => {
            const selectedDate = document.getElementById('analysisDate').value;
            if (selectedDate) {
                this.setAnalysisDate(selectedDate);
            } else {
                alert('Please select a date for backtesting');
            }
        });

        // Live mode button
        document.getElementById('liveBtn').addEventListener('click', () => {
            this.setAnalysisDate(null);
        });

        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.loadTPOData();
        });
    }

    async loadAnalysisDateConfig() {
        try {
            const response = await fetch('/api/analysis_date');
            const config = await response.json();
            
            this.currentAnalysisDate = config.analysis_date;
            this.isLiveMode = config.is_live_mode;
            
            this.updateModeIndicator();
            this.updateAnalysisInfo(config);
            
            // Set date input to current analysis date
            if (this.currentAnalysisDate) {
                document.getElementById('analysisDate').value = this.currentAnalysisDate;
            }
        } catch (error) {
            console.error('Error loading analysis date config:', error);
        }
    }

    async setAnalysisDate(date) {
        try {
            const response = await fetch(`/api/analysis_date?date=${date || 'live'}`, {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.error) {
                alert(result.error);
                return;
            }
            
            this.currentAnalysisDate = result.analysis_date;
            this.isLiveMode = result.is_live_mode;
            
            this.updateModeIndicator();
            this.updateAnalysisInfo(result);
            
            // Reload TPO data with new date
            this.loadTPOData();
            
            // Show success message
            this.showMessage(result.message, 'success');
        } catch (error) {
            console.error('Error setting analysis date:', error);
            this.showMessage('Error setting analysis date', 'error');
        }
    }

    updateModeIndicator() {
        const indicator = document.getElementById('modeIndicator');
        if (this.isLiveMode) {
            indicator.textContent = 'LIVE MODE';
            indicator.className = 'mode-indicator mode-live';
        } else {
            indicator.textContent = 'BACKTEST MODE';
            indicator.className = 'mode-indicator mode-backtest';
        }
    }

    updateAnalysisInfo(config) {
        try {
            const infoDiv = document.getElementById('analysisInfo');
            const dateText = document.getElementById('analysisDateText');
            const modeText = document.getElementById('analysisModeText');
            const dataText = document.getElementById('dataAvailabilityText');
            
            // Check if elements exist before trying to update them
            if (!infoDiv || !dateText || !modeText || !dataText) {
                console.warn('Analysis info elements not found, skipping update');
                return;
            }
            
            if (this.isLiveMode) {
                dateText.textContent = `Analysis Date: ${config.current_date} (Live)`;
                modeText.textContent = 'Mode: Real-time analysis with live data';
                dataText.textContent = 'Data: Updating in real-time';
            } else {
                dateText.textContent = `Analysis Date: ${this.currentAnalysisDate} (Historical)`;
                modeText.textContent = 'Mode: Historical backtesting';
                dataText.textContent = 'Data: Complete market session (9:15am - 3:30pm IST)';
            }
            
            infoDiv.style.display = 'block';
        } catch (error) {
            console.error('Error in updateAnalysisInfo:', error);
        }
    }

    async loadMarginData() {
        try {
            const response = await fetch('/api/margin_data');
            const data = await response.json();
            
            if (data.error) {
                console.error('Error loading margin data:', data.error);
                return;
            }

            // Update margin data in status bar
            const balanceElement = document.getElementById('marginBalance');
            
            if (balanceElement) {
                balanceElement.textContent = `Balance: ₹${data.live_balance.toLocaleString()}`;
            }

            // Fetch today's P&L
            try {
                const pnlResponse = await fetch('/api/today_pnl');
                const pnlData = await pnlResponse.json();
                
                if (!pnlData.error && pnlData.total_pnl !== undefined) {
                    const pnlElement = document.getElementById('todayPnl');
                    const pnlColor = pnlData.total_pnl >= 0 ? 'color: #28a745;' : 'color: #dc3545;';
                    if (pnlElement) {
                        pnlElement.innerHTML = `<span style="${pnlColor}">Today's P&L: ₹${pnlData.total_pnl.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>`;
                    }
                }
            } catch (pnlError) {
                console.error('Error loading P&L data:', pnlError);
            }
        } catch (error) {
            console.error('Error loading margin data:', error);
        }
    }

    showMessage(message, type) {
        // Create a temporary message element
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            background: ${type === 'success' ? '#28a745' : '#dc3545'};
        `;
        messageDiv.textContent = message;
        document.body.appendChild(messageDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            document.body.removeChild(messageDiv);
        }, 3000);
    }

    async loadTPOData() {
        try {
            // Build URL with analysis date parameter
            let url = '/api/tpo_charts';
            if (this.currentAnalysisDate) {
                url += `?analysis_date=${this.currentAnalysisDate}`;
            }
            
            const response = await fetch(url);
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
                return;
            }

            // Display the matplotlib-generated chart image
            const chartImage = document.getElementById('tpoChartImage');
            if (chartImage && data.chart_image) {
                chartImage.src = data.chart_image;
                chartImage.style.display = 'block';
            }
        } catch (error) {
            console.error('Error loading TPO data:', error);
            this.showError('Failed to load TPO data');
        }
    }

    updateChartTitles(data) {
        // Chart titles are now handled by matplotlib
        // No need to update web chart titles
    }

    // Chart rendering methods removed - now using matplotlib-generated images

    showError(message) {
        const chartContainer = document.querySelector('.tpo-chart-container');
        chartContainer.innerHTML = `
            <div style="text-align: center; color: #f44336; padding: 20px;">
                <h4>⚠️ Error Loading Charts</h4>
                <p>${message}</p>
                <button onclick="location.reload()" style="margin-top: 10px; padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Retry
                </button>
            </div>
        `;
    }

    updateCharts() {
        this.loadTPOData();
    }
}

// Initialize TPO charts when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.tpoRenderer = new TPOChartRenderer();
});

// Update charts every 30 seconds (only in live mode)
setInterval(function() {
    if (window.tpoRenderer && window.tpoRenderer.isLiveMode) {
        window.tpoRenderer.updateCharts();
    }
}, 30000);
