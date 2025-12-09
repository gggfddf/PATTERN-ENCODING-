"""
Adaptive Ornstein-Uhlenbeck Mean Reversion Trading System
==========================================================
A production-grade implementation with log-price stationarity,
robust parameter estimation, and discrete trade generation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: OU PARAMETER ESTIMATION (LOG-PRICE SPACE)
# ============================================================================

def estimate_ou_params(series, dt=1):
    """
    Estimate Ornstein-Uhlenbeck parameters from LOG-PRICE time series.
    
    OU is applied to log-prices for stationarity: log(P_t) follows OU process.
    
    Uses AR(1) regression: X_t = α + β*X_{t-1} + ε_t
    Then converts to OU parameters:
        θ = -ln(β) / Δt    (mean reversion speed)
        μ = α / (1 - β)    (long-term mean in log-space)
        σ = std(ε) * sqrt(2θ / (1 - β²))  (volatility)
    
    Parameters:
    -----------
    series : array-like
        LOG-price time series
    dt : float
        Time step (default=1 for discrete intervals)
    
    Returns:
    --------
    theta : float - Mean reversion speed
    mu_log : float - Long-term equilibrium mean (in log-space)
    sigma : float - Volatility parameter
    """
    try:
        # Need at least 3 points for meaningful regression
        if len(series) < 10:
            return None, None, None
        
        # Prepare AR(1) regression data
        y = series.values[1:]  # X_t
        x = sm.add_constant(series.values[:-1])  # [1, X_{t-1}]
        
        # Fit OLS regression
        model = sm.OLS(y, x).fit()
        alpha, beta = model.params
        resid = model.resid
        
        # CRITICAL: Check for stationarity
        # β must be in (0, 1) for mean-reverting process
        if beta <= 0 or beta >= 0.999:  # Allow slight wiggle room
            return None, None, None  # Non-stationary or invalid
        
        # Convert AR(1) to OU parameters
        theta = -np.log(beta) / dt
        mu_log = alpha / (1 - beta)  # This is in log-space
        
        # Calculate volatility with numerical safeguards
        denom = 1 - beta**2
        if denom <= 0 or theta <= 0:
            return None, None, None
            
        sigma = np.std(resid) * np.sqrt(2 * theta / denom)
        
        # Enforce minimum volatility (prevents division by zero)
        if sigma < 1e-8:
            sigma = 1e-8
            
        # Sanity checks
        if not np.isfinite(theta) or not np.isfinite(mu_log) or not np.isfinite(sigma):
            return None, None, None
            
        return theta, mu_log, sigma
        
    except Exception as e:
        return None, None, None


# ============================================================================
# STEP 2: ROLLING ESTIMATION FRAMEWORK (INDEX-ALIGNED)
# ============================================================================

def compute_adaptive_ou(prices, window=500):
    """
    Compute adaptive OU parameters using rolling window on LOG-PRICES.
    Returns DataFrame aligned to original price index.
    
    Parameters:
    -----------
    prices : pd.Series
        Raw price time series with datetime index
    window : int
        Rolling window size (number of bars)
    
    Returns:
    --------
    df : pd.DataFrame
        Index-aligned DataFrame with columns: 
        price, log_price, theta, mu_log, mu_price, sigma, zscore
    """
    # Convert to log-prices for OU estimation
    log_prices = np.log(prices)
    
    # Pre-allocate arrays (faster than lists)
    n = len(prices)
    theta_arr = np.full(n, np.nan)
    mu_log_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    
    print(f"Computing adaptive OU parameters over {n} bars...")
    print(f"Using rolling window of {window} bars (log-price space)")
    print(f"Estimating parameters from bar {window} to {n}...\n")
    
    valid_count = 0
    for i in range(window, n):
        window_data = log_prices.iloc[i-window:i]
        theta, mu_log, sigma = estimate_ou_params(window_data)
        
        # Store valid estimates at index i
        if theta is not None:
            theta_arr[i] = theta
            mu_log_arr[i] = mu_log
            sigma_arr[i] = sigma
            valid_count += 1
    
    # Create results DataFrame aligned to original index
    df = pd.DataFrame({
        'price': prices,
        'log_price': log_prices,
        'theta': theta_arr,
        'mu_log': mu_log_arr,
        'sigma': sigma_arr
    }, index=prices.index)
    
    # Convert mu back to price space for plotting/trading
    # exp(mu_log) gives the geometric mean price level
    df['mu_price'] = np.exp(df['mu_log'])
    
    # Calculate Z-score in log-space: (log_price - μ_log) / σ
    df['zscore'] = (df['log_price'] - df['mu_log']) / df['sigma']
    
    # Drop rows where parameters couldn't be estimated
    df_valid = df.dropna(subset=['theta', 'mu_log', 'sigma'])
    
    print(f"Valid estimates: {len(df_valid)} / {n - window} ({100*len(df_valid)/(n-window):.1f}%)")
    print(f"Average theta (reversion speed): {df_valid['theta'].mean():.4f}")
    print(f"Average mu (log-space): {df_valid['mu_log'].mean():.4f}")
    print(f"Average mu (price-space): {df_valid['mu_price'].mean():.2f}")
    print(f"Average sigma (volatility): {df_valid['sigma'].mean():.4f}\n")
    
    return df_valid


# ============================================================================
# STEP 3: DISCRETE TRADE GENERATION WITH PROPER EXIT LOGIC
# ============================================================================

def generate_trades(df, long_entry=-1.5, short_entry=1.5, exit_threshold=0.0):
    """
    Generate discrete trades with proper entry/exit logic.
    
    Trading Rules:
    - LONG: Enter when zscore <= long_entry, exit when zscore >= exit_threshold
    - SHORT: Enter when zscore >= short_entry, exit when zscore <= exit_threshold
    - Opposite signals also close existing positions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Output from compute_adaptive_ou()
    long_entry : float
        Z-score threshold for long entry (default: -1.5)
    short_entry : float
        Z-score threshold for short entry (default: 1.5)
    exit_threshold : float
        Z-score for exit (default: 0.0, mean reversion)
    
    Returns:
    --------
    trades : pd.DataFrame
        Columns: entry_time, exit_time, direction, entry_price, exit_price, 
                 entry_zscore, exit_zscore, pnl_pct
    df : pd.DataFrame
        Original df with 'position' column (1=long, -1=short, 0=flat)
    """
    df = df.copy()
    df['position'] = 0  # 0=flat, 1=long, -1=short
    
    trades = []
    current_position = 0
    entry_idx = None
    entry_price = None
    entry_zscore = None
    
    for idx, row in df.iterrows():
        z = row['zscore']
        price = row['price']
        
        if current_position == 0:  # No position
            # Check for entry signals
            if z <= long_entry:
                current_position = 1  # Enter long
                entry_idx = idx
                entry_price = price
                entry_zscore = z
            elif z >= short_entry:
                current_position = -1  # Enter short
                entry_idx = idx
                entry_price = price
                entry_zscore = z
        
        elif current_position == 1:  # In long position
            # Exit if: (1) crosses mean, or (2) opposite signal
            if z >= exit_threshold or z >= short_entry:
                # Close long
                exit_price = price
                exit_zscore = z
                pnl_pct = 100 * (exit_price - entry_price) / entry_price
                
                trades.append({
                    'entry_time': entry_idx,
                    'exit_time': idx,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_zscore': entry_zscore,
                    'exit_zscore': exit_zscore,
                    'pnl_pct': pnl_pct
                })
                
                # Check if new short signal
                if z >= short_entry:
                    current_position = -1
                    entry_idx = idx
                    entry_price = price
                    entry_zscore = z
                else:
                    current_position = 0
        
        elif current_position == -1:  # In short position
            # Exit if: (1) crosses mean, or (2) opposite signal
            if z <= exit_threshold or z <= long_entry:
                # Close short
                exit_price = price
                exit_zscore = z
                pnl_pct = 100 * (entry_price - exit_price) / entry_price
                
                trades.append({
                    'entry_time': entry_idx,
                    'exit_time': idx,
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_zscore': entry_zscore,
                    'exit_zscore': exit_zscore,
                    'pnl_pct': pnl_pct
                })
                
                # Check if new long signal
                if z <= long_entry:
                    current_position = 1
                    entry_idx = idx
                    entry_price = price
                    entry_zscore = z
                else:
                    current_position = 0
        
        # Mark position in dataframe
        df.loc[idx, 'position'] = current_position
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        print(f"Generated {len(trades_df)} completed trades:")
        print(f"  Long trades:  {(trades_df['direction'] == 'LONG').sum()}")
        print(f"  Short trades: {(trades_df['direction'] == 'SHORT').sum()}")
        print(f"  Avg PnL:      {trades_df['pnl_pct'].mean():.3f}%")
        print(f"  Win rate:     {100 * (trades_df['pnl_pct'] > 0).mean():.1f}%")
        print(f"  Total PnL:    {trades_df['pnl_pct'].sum():.3f}%\n")
    else:
        print("No completed trades generated.\n")
    
    return trades_df, df


# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def plot_adaptive_ou(df, trades_df, instrument_name="Asset"):
    """
    Create comprehensive visualization with trade markers.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    
    # Plot 1: Price with Adaptive Mean and Trade Markers
    ax1 = axes[0]
    ax1.plot(df.index, df['price'], label='Price', color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(df.index, df['mu_price'], label='Adaptive Mean (μ)', color='blue', linewidth=2)
    
    # ±1.5σ bands (in price space: exp(μ_log ± 1.5*σ))
    upper_band = np.exp(df['mu_log'] + 1.5 * df['sigma'])
    lower_band = np.exp(df['mu_log'] - 1.5 * df['sigma'])
    ax1.fill_between(df.index, lower_band, upper_band, 
                     color='gray', alpha=0.2, label='±1.5σ Band')
    
    # Plot trade markers
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['direction'] == 'LONG' else 'red'
            marker = '^' if trade['direction'] == 'LONG' else 'v'
            
            # Entry marker
            ax1.scatter(trade['entry_time'], trade['entry_price'], 
                       color=color, marker=marker, s=100, zorder=5, alpha=0.8)
            # Exit marker
            ax1.scatter(trade['exit_time'], trade['exit_price'], 
                       color=color, marker='x', s=100, zorder=5, alpha=0.8)
    
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title(f'Adaptive OU Mean Reversion System (Log-Price) - {instrument_name}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Z-Score with Position Overlay
    ax2 = axes[1]
    ax2.plot(df.index, df['zscore'], label='Z-Score', color='purple', linewidth=1.5)
    ax2.axhline(y=1.5, color='r', linestyle='--', linewidth=1, label='Short Entry', alpha=0.7)
    ax2.axhline(y=-1.5, color='g', linestyle='--', linewidth=1, label='Long Entry', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Exit')
    ax2.fill_between(df.index, -1.5, 1.5, color='yellow', alpha=0.1)
    
    # Shade position periods
    long_periods = df['position'] == 1
    short_periods = df['position'] == -1
    ax2.fill_between(df.index, -3, 3, where=long_periods, 
                     color='green', alpha=0.15, label='Long Position')
    ax2.fill_between(df.index, -3, 3, where=short_periods, 
                     color='red', alpha=0.15, label='Short Position')
    
    ax2.set_ylabel('Z-Score', fontsize=11)
    ax2.set_ylim(-3, 3)
    ax2.set_title('Mean Reversion Signal & Positions', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Theta (Reversion Speed) - Regime Detection
    ax3 = axes[2]
    ax3.plot(df.index, df['theta'], label='θ (Reversion Speed)', 
             color='orange', linewidth=1.5)
    ax3.axhline(y=df['theta'].mean(), color='red', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Mean θ = {df["theta"].mean():.4f}')
    
    # Highlight low-theta (trending) periods
    low_theta = df['theta'] < df['theta'].quantile(0.25)
    ax3.fill_between(df.index, 0, df['theta'].max(), where=low_theta,
                     color='orange', alpha=0.2, label='Low θ (Trending?)')
    
    ax3.set_ylabel('Theta (θ)', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_title('Adaptive Reversion Speed (Regime Indicator)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_trade_analysis(trades_df):
    """
    Analyze trade performance and distributions.
    """
    if len(trades_df) == 0:
        print("No trades to analyze.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # PnL Distribution
    axes[0, 0].hist(trades_df['pnl_pct'], bins=30, color='steelblue', 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(trades_df['pnl_pct'].mean(), color='green', 
                       linestyle='--', linewidth=2, label=f"Mean: {trades_df['pnl_pct'].mean():.2f}%")
    axes[0, 0].set_title('Trade PnL Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('PnL (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative PnL
    cumulative_pnl = trades_df['pnl_pct'].cumsum()
    axes[0, 1].plot(cumulative_pnl.values, linewidth=2, color='darkgreen')
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Cumulative PnL', fontweight='bold')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Cumulative PnL (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entry Z-Score Distribution
    axes[1, 0].hist(trades_df['entry_zscore'], bins=30, color='purple', 
                    alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(-1.5, color='green', linestyle='--', linewidth=2, label='Long Threshold')
    axes[1, 0].axvline(1.5, color='red', linestyle='--', linewidth=2, label='Short Threshold')
    axes[1, 0].set_title('Entry Z-Score Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Entry Z-Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Win/Loss by Direction
    direction_pnl = trades_df.groupby('direction')['pnl_pct'].agg(['mean', 'count'])
    axes[1, 1].bar(direction_pnl.index, direction_pnl['mean'], 
                   color=['green' if x > 0 else 'red' for x in direction_pnl['mean']],
                   alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(0, color='black', linewidth=1)
    axes[1, 1].set_title('Average PnL by Direction', fontweight='bold')
    axes[1, 1].set_ylabel('Average PnL (%)')
    
    # Add count labels
    for i, (idx, row) in enumerate(direction_pnl.iterrows()):
        axes[1, 1].text(i, row['mean'], f"n={int(row['count'])}", 
                       ha='center', va='bottom' if row['mean'] > 0 else 'top')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_parameter_diagnostics(df):
    """
    Diagnostic plots for OU parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # θ distribution
    axes[0, 0].hist(df['theta'].dropna(), bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of θ (Reversion Speed)', fontweight='bold')
    axes[0, 0].set_xlabel('θ')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['theta'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # σ distribution
    axes[0, 1].hist(df['sigma'].dropna(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of σ (Volatility)', fontweight='bold')
    axes[0, 1].set_xlabel('σ')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['sigma'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z-score distribution
    axes[1, 0].hist(df['zscore'].dropna(), bins=100, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Z-Scores', fontweight='bold')
    axes[1, 0].set_xlabel('Z-Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].axvline(-1.5, color='green', linestyle='--', linewidth=2, label='Entry')
    axes[1, 0].axvline(1.5, color='red', linestyle='--', linewidth=2)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # θ vs σ relationship
    axes[1, 1].scatter(df['theta'], df['sigma'], alpha=0.3, s=10, color='darkblue')
    axes[1, 1].set_title('θ vs σ Relationship', fontweight='bold')
    axes[1, 1].set_xlabel('θ (Reversion Speed)')
    axes[1, 1].set_ylabel('σ (Volatility)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# STEP 5: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline for the adaptive OU mean reversion system.
    """
    print("=" * 70)
    print("ADAPTIVE ORNSTEIN-UHLENBECK MEAN REVERSION TRADING SYSTEM")
    print("(Log-Price Space with Discrete Trade Generation)")
    print("=" * 70)
    print()
    
    # Configuration
    TICKER = "AAPL"  # Apple stock (more reliable data)
    INTERVAL = "5m"
    PERIOD = "7d"
    WINDOW = 100  # Rolling window size (reduced for smaller dataset)
    LONG_ENTRY = -1.5
    SHORT_ENTRY = 1.5
    EXIT_THRESHOLD = 0.0
    print(f"Configuration:")
    print(f"  Instrument: {TICKER}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Period: {PERIOD}")
    print(f"  Rolling Window: {WINDOW} bars")
    print(f"  Long Entry: z-score <= {LONG_ENTRY}")
    print(f"  Short Entry: z-score >= {SHORT_ENTRY}")
    print(f"  Exit: z-score crosses {EXIT_THRESHOLD}")
    print()
    
    # Download data
    print("Downloading data from Yahoo Finance...")
    data = yf.download(TICKER, interval=INTERVAL, period=PERIOD, progress=False)
    prices = data['Close'].dropna()
    
    # Debug data shape
    print(f"Data shape: {data.shape}")
    print(f"Prices shape: {prices.shape}")
    print(f"Prices type: {type(prices)}")
    
    # Ensure prices is 1D Series
    if hasattr(prices, 'values') and len(prices.values.shape) > 1:
        prices = prices.iloc[:, 0] if prices.values.shape[1] > 0 else prices.iloc[0]
    
    print(f"Final prices shape: {prices.shape}")
    print(f"Downloaded {len(prices)} bars\n")
    
    # Compute adaptive OU parameters (on log-prices)
    df = compute_adaptive_ou(prices, window=WINDOW)
    
    # Generate discrete trades
    trades_df, df = generate_trades(df, 
                                     long_entry=LONG_ENTRY, 
                                     short_entry=SHORT_ENTRY, 
                                     exit_threshold=EXIT_THRESHOLD)
    
    # Display statistics
    print("=" * 70)
    print("PARAMETER STATISTICS")
    print("=" * 70)
    print(f"\nTheta - Reversion Speed:")
    print(f"  Mean: {df['theta'].mean():.6f}")
    print(f"  Std:  {df['theta'].std():.6f}")
    print(f"  Min:  {df['theta'].min():.6f}")
    print(f"  Max:  {df['theta'].max():.6f}")
    
    print(f"\nMu - Equilibrium Mean:")
    print(f"  Log-space mean: {df['mu_log'].mean():.4f}")
    print(f"  Price-space mean: {df['mu_price'].mean():.2f}")
    
    print(f"\nSigma - Volatility:")
    print(f"  Mean: {df['sigma'].mean():.4f}")
    print(f"  Std:  {df['sigma'].std():.4f}")
    
    print(f"\nZ-Score Distribution:")
    print(f"  Mean: {df['zscore'].mean():.4f}")
    print(f"  Std:  {df['zscore'].std():.4f}")
    print(f"  Min:  {df['zscore'].min():.4f}")
    print(f"  Max:  {df['zscore'].max():.4f}")
    print()
    
    # Visualizations
    print("Generating visualizations...\n")
    plot_adaptive_ou(df, trades_df, instrument_name=TICKER)
    
    if len(trades_df) > 0:
        plot_trade_analysis(trades_df)
    
    plot_parameter_diagnostics(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_file = f"ou_params_{TICKER.replace('=', '_')}_{timestamp}.csv"
    df.to_csv(df_file)
    print(f"Parameters saved to: {df_file}")
    
    if len(trades_df) > 0:
        trades_file = f"ou_trades_{TICKER.replace('=', '_')}_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Trades saved to: {trades_file}")
    
    return df, trades_df


# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    df_results, trades = main()
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("• OU process applied to LOG-PRICES for stationarity")
    print("• High theta -> Fast mean reversion (range-bound)")
    print("• Low theta -> Weak reversion (trending/breakout) -> fewer trades")
    print("• Z-score measures deviation in log-space")
    print("• Bands in price-space: exp(mu_log ± k*sigma)")
    print("\nNext Steps:")
    print("1. Add transaction costs (e.g., 0.02% per trade)")
    print("2. Optimize thresholds and window size")
    print("3. Position sizing based on theta and sigma")
    print("4. Stop-loss / take-profit logic")
    print("5. Walk-forward optimization")