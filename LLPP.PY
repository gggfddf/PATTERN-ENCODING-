import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit, differential_evolution
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class LPPLBubbleDetector:
    
    def __init__(self, symbol, lookback_days=200):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.data = None
        self.log_prices = None
        self.time_vector = None
        self.params = None
        self.bubble_detected = False
        self.results = {}
        
    def fetch_data(self):
        ticker = yf.Ticker(self.symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 30)
        df = ticker.history(start=start_date, end=end_date)
        self.data = np.asarray(df['Close'].values[-self.lookback_days:]).flatten()
        self.dates = df.index[-self.lookback_days:]
        return self
    
    def smooth_data(self, window=3):
        if len(self.data) < window:
            return self
        self.data = np.asarray(self.data).flatten()
        smoothed = pd.Series(self.data).rolling(window=window, center=True, min_periods=1).mean()
        self.data = np.asarray(smoothed.values).flatten()
        return self
    
    def transform_to_log_space(self):
        self.data = np.asarray(self.data).flatten()
        self.log_prices = np.log(self.data)
        self.time_vector = np.arange(len(self.data))
        return self
    
    @staticmethod
    def lppl_function(t, A, B, C, t_c, beta, omega, phi):
        dt = t_c - t
        return A + B * (dt ** beta) + C * (dt ** beta) * np.cos(omega * np.log(dt) + phi)
    
    def fit_lppl_model(self):
        t_max = self.time_vector[-1]
        
        def objective(params):
            A, B, C, t_c, beta, omega, phi = params
            dt = t_c - self.time_vector
            if np.any(dt <= 0):
                return 1e10
            try:
                predicted = A + B * (dt ** beta) + C * (dt ** beta) * np.cos(omega * np.log(dt) + phi)
                residuals = self.log_prices - predicted
                return np.sum(residuals ** 2)
            except:
                return 1e10
        
        bounds = [
            (self.log_prices.min() - 1, self.log_prices.max() + 2),
            (-5, -0.0001),
            (-0.5, 0.5),
            (t_max + 1, t_max + 90),
            (0.05, 0.95),
            (4, 15),
            (0, 2 * np.pi)
        ]
        
        result = differential_evolution(
            objective,
            bounds,
            seed=None,
            maxiter=500,
            popsize=15,
            atol=0,
            tol=0.01,
            workers=1,
            polish=False,
            strategy='best1bin'
        )
        
        self.params = {
            'A': result.x[0],
            'B': result.x[1],
            'C': result.x[2],
            't_c': result.x[3],
            'beta': result.x[4],
            'omega': result.x[5],
            'phi': result.x[6],
            'fit_error': result.fun / len(self.time_vector)
        }
        
        return self
    
    def calculate_goodness_of_fit(self):
        predicted = self.lppl_function(
            self.time_vector,
            self.params['A'],
            self.params['B'],
            self.params['C'],
            self.params['t_c'],
            self.params['beta'],
            self.params['omega'],
            self.params['phi']
        )
        
        ss_res = np.sum((self.log_prices - predicted) ** 2)
        ss_tot = np.sum((self.log_prices - np.mean(self.log_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((self.log_prices - predicted) ** 2))
        
        self.params['r_squared'] = r_squared
        self.params['rmse'] = rmse
        
        return self
    
    def validate_bubble_conditions(self):
        t_last = self.time_vector[-1]
        days_to_crash = self.params['t_c'] - t_last
        
        condition_beta = 0.05 < self.params['beta'] < 0.95
        condition_B_negative = self.params['B'] < -0.0001
        condition_C_reasonable = abs(self.params['C']) > 0.001
        condition_future_crash = self.params['t_c'] > t_last
        condition_window = 3 <= days_to_crash <= 90
        condition_fit_quality = self.params['r_squared'] > 0.75
        
        price_60d_ago_idx = max(0, len(self.data) - 60)
        condition_price_acceleration = (self.data[-1] / self.data[price_60d_ago_idx]) > 1.10
        
        condition_strong_signal = (
            (0.1 < self.params['beta'] < 0.8) and 
            (self.params['B'] < -0.001) and 
            (days_to_crash <= 45) and
            (self.params['r_squared'] > 0.85)
        )
        
        all_conditions = [
            condition_beta,
            condition_B_negative,
            condition_C_reasonable,
            condition_future_crash,
            condition_window,
            condition_fit_quality,
            condition_price_acceleration
        ]
        
        conditions_met = sum(all_conditions)
        confidence = (conditions_met / len(all_conditions)) * 100
        
        self.bubble_detected = (conditions_met >= 5) or condition_strong_signal
        
        crash_date = self.dates[-1] + timedelta(days=int(days_to_crash))
        
        price_change = ((self.data[-1] / self.data[price_60d_ago_idx]) - 1) * 100
        
        self.results = {
            'symbol': self.symbol,
            'bubble_detected': self.bubble_detected,
            't_c_predicted': crash_date.strftime('%Y-%m-%d'),
            'days_to_crash': int(days_to_crash),
            'beta': round(self.params['beta'], 4),
            'B': round(self.params['B'], 4),
            'C': round(self.params['C'], 4),
            'omega': round(self.params['omega'], 4),
            'A': round(self.params['A'], 4),
            'phi': round(self.params['phi'], 4),
            'r_squared': round(self.params['r_squared'], 4),
            'rmse': round(self.params['rmse'], 6),
            'fit_error': round(self.params['fit_error'], 6),
            'confidence': round(confidence, 2),
            'signal': 'CRASH WARNING' if self.bubble_detected else 'NORMAL BEHAVIOR',
            'price_change_60d': round(price_change, 2),
            'conditions': {
                'beta_valid': condition_beta,
                'B_negative': condition_B_negative,
                'C_reasonable': condition_C_reasonable,
                'future_crash': condition_future_crash,
                'window_valid': condition_window,
                'fit_quality': condition_fit_quality,
                'price_acceleration': condition_price_acceleration,
                'strong_signal': condition_strong_signal
            }
        }
        
        return self
    
    def plot_results(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        predicted = self.lppl_function(
            self.time_vector,
            self.params['A'],
            self.params['B'],
            self.params['C'],
            self.params['t_c'],
            self.params['beta'],
            self.params['omega'],
            self.params['phi']
        )
        
        ax1.plot(self.dates, self.data, 'b-', linewidth=2, label='Actual Price', alpha=0.7)
        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.set_title(f'{self.symbol} - LPPL Bubble Analysis', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.dates, self.log_prices, 'b-', linewidth=2, label='Actual Log-Price', alpha=0.7)
        ax2.plot(self.dates, predicted, 'r--', linewidth=2, label='LPPL Fit', alpha=0.8)
        
        crash_date = self.dates[-1] + timedelta(days=int(self.params['t_c'] - self.time_vector[-1]))
        ax2.axvline(crash_date, color='orange', linewidth=2, linestyle='--', label=f'Predicted Crash: {crash_date.strftime("%Y-%m-%d")}')
        
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Log Price', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        status_text = f"Bubble Detected: {self.bubble_detected}\nConfidence: {self.results['confidence']:.1f}%\n"
        status_text += f"Beta: {self.params['beta']:.4f} | B: {self.params['B']:.4f}\n"
        status_text += f"R²: {self.params['r_squared']:.4f} | Days to Crash: {self.results['days_to_crash']}"
        
        ax2.text(0.02, 0.98, status_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return self
    
    def run_analysis(self):
        self.fetch_data()
        self.smooth_data()
        self.transform_to_log_space()
        self.fit_lppl_model()
        self.calculate_goodness_of_fit()
        self.validate_bubble_conditions()
        return self.results


def scan_multiple_assets(symbols, lookback_days=200):
    results_list = []
    
    for symbol in symbols:
        detector = LPPLBubbleDetector(symbol, lookback_days)
        results = detector.run_analysis()
        results_list.append(results)
    
    df = pd.DataFrame(results_list)
    df = df.sort_values('confidence', ascending=False)
    
    return df


def run_rolling_backtest(symbol, start_date_str, end_date_str, lookback_days=200, step_days=5, debug=False):
    """
    Rolling window backtest to validate historical prediction accuracy.
    Tests whether the model would have detected known historical bubbles.
    """
    
    print(f"\n{'='*80}")
    print(f"ROLLING WINDOW BACKTEST - {symbol}")
    print(f"Period: {start_date_str} to {end_date_str}")
    print(f"Window: {lookback_days} days | Step: {step_days} days")
    print(f"{'='*80}\n")
    
    raw_data = yf.download(symbol, start=start_date_str, end=end_date_str, progress=False)
    
    if isinstance(raw_data['Close'], pd.DataFrame):
        full_data = raw_data['Close'].iloc[:, 0]
    else:
        full_data = raw_data['Close']
    
    if len(full_data) < lookback_days:
        print("Insufficient data for backtest.")
        return None
    
    all_predictions = []
    windows_tested = 0
    fit_failures = 0
    
    for i in range(lookback_days, len(full_data), step_days):
        window_data = full_data.iloc[i - lookback_days : i]
        windows_tested += 1
        
        detector = LPPLBubbleDetector(symbol, lookback_days)
        detector.data = np.asarray(window_data, dtype=np.float64).flatten()
        detector.dates = pd.DatetimeIndex(window_data.index)
        
        try:
            detector.smooth_data()
            detector.transform_to_log_space()
            detector.fit_lppl_model()
            detector.calculate_goodness_of_fit()
            detector.validate_bubble_conditions()
            
            results = detector.results
            
            if debug and windows_tested % 20 == 0:
                print(f"  Window ending {window_data.index[-1].date()}: "
                      f"β={results['beta']:.3f}, B={results['B']:.4f}, "
                      f"days={results['days_to_crash']}, R²={results['r_squared']:.3f}, "
                      f"growth={results['price_change_60d']:.1f}%")
            
            if results['bubble_detected']:
                prediction = {
                    'analysis_date': window_data.index[-1],
                    't_c_predicted': results['t_c_predicted'],
                    'days_to_crash': results['days_to_crash'],
                    'confidence': results['confidence'],
                    'beta': results['beta'],
                    'r_squared': results['r_squared'],
                    'price_change_60d': results['price_change_60d']
                }
                all_predictions.append(prediction)
                print(f"  ✓ SIGNAL: {window_data.index[-1].date()} → Crash: {results['t_c_predicted']} "
                      f"({results['days_to_crash']}d, conf={results['confidence']:.0f}%)")
                
        except (RuntimeError, ValueError, Exception) as e:
            fit_failures += 1
            if debug and fit_failures <= 3:
                print(f"  ✗ Fit failed: {window_data.index[-1].date()} - {str(e)[:80]}")
            continue
    
    success_rate = ((windows_tested - fit_failures) / windows_tested * 100) if windows_tested > 0 else 0
    print(f"\nBacktest Complete:")
    print(f"  Windows Tested: {windows_tested}")
    print(f"  Successful Fits: {windows_tested - fit_failures} ({success_rate:.1f}%)")
    print(f"  Bubble Signals: {len(all_predictions)}\n")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(full_data.index, full_data.values, 'b-', linewidth=2, label=f'{symbol} Price', alpha=0.6)
    
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df['t_c_predicted'] = pd.to_datetime(pred_df['t_c_predicted'])
        
        analysis_prices = [full_data[date] for date in pred_df['analysis_date']]
        
        ax.scatter(pred_df['analysis_date'], analysis_prices, color='red', s=50, 
                  label='Bubble Signal', zorder=5, alpha=0.8, edgecolors='darkred', linewidth=1)
        
        for idx, pred in pred_df.iterrows():
            crash_price = full_data[pred['t_c_predicted']] if pred['t_c_predicted'] in full_data.index else None
            if crash_price is not None:
                ax.plot([pred['analysis_date'], pred['t_c_predicted']],
                       [analysis_prices[idx], crash_price],
                       'orange', linestyle='--', alpha=0.4, linewidth=1.5)
    
    ax.set_title(f'{symbol} - Rolling Window LPPL Backtest', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(all_predictions) if all_predictions else None


if __name__ == "__main__":
    
    # CURRENT ANALYSIS
    symbol = "NVDA"
    
    detector = LPPLBubbleDetector(symbol, lookback_days=200)
    results = detector.run_analysis()
    detector.plot_results()
    
    print("\n" + "="*80)
    print(f"LPPL BUBBLE DETECTION RESULTS - {symbol}")
    print("="*80)
    print(f"\nBubble Detected: {results['bubble_detected']}")
    print(f"Signal: {results['signal']}")
    print(f"Confidence: {results['confidence']:.2f}%")
    print(f"Price Change (60d): {results['price_change_60d']}%")
    print(f"\nCritical Time (t_c): {results['t_c_predicted']}")
    print(f"Days to Crash: {results['days_to_crash']}")
    print(f"\nModel Parameters:")
    print(f"  Beta (β): {results['beta']:.4f}")
    print(f"  B: {results['B']:.4f}")
    print(f"  C: {results['C']:.4f}")
    print(f"  Omega (ω): {results['omega']:.4f}")
    print(f"  A: {results['A']:.4f}")
    print(f"  Phi (φ): {results['phi']:.4f}")
    print(f"\nFit Quality:")
    print(f"  R²: {results['r_squared']:.4f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"\nConditions Check:")
    for condition, status in results['conditions'].items():
        print(f"  {condition}: {'✓' if status else '✗'}")
    print("="*80)
    
    # MULTI-ASSET SCAN
    symbols = ["NVDA", "TSLA", "BTC-USD", "AAPL"]
    print("\n\nScanning multiple assets...")
    multi_results = scan_multiple_assets(symbols, lookback_days=200)
    print("\n" + multi_results[['symbol', 'bubble_detected', 't_c_predicted', 'days_to_crash', 'beta', 'confidence']].to_string(index=False))
    
    # HISTORICAL BACKTEST
    print("\n\n" + "="*80)
    print("HISTORICAL VALIDATION")
    print("="*80)
    
    backtest_results = run_rolling_backtest(
        symbol="BTC-USD",
        start_date_str="2017-01-01",
        end_date_str="2018-03-01",
        lookback_days=150,
        step_days=3,
        debug=True
    )
    
    if backtest_results is not None:
        print("\nHistorical Predictions Summary:")
        print(backtest_results[['analysis_date', 't_c_predicted', 'days_to_crash', 'confidence']].to_string(index=False))