import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MeanReversionStrategy:
    """
    Mean Reversion Strategy using Two Moving Averages
    with Percentage Difference Normalization
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, 
                 entry_threshold: float = 2.0, exit_threshold: float = 0.5,
                 exit_method: str = 'threshold', hold_candles: int = 5):
        """
        Initialize the mean reversion strategy
        
        Parameters:
        -----------
        fast_period : int
            Period for fast moving average
        slow_period : int
            Period for slow moving average
        entry_threshold : float
            Threshold for entry signal (percentage difference)
        exit_threshold : float
            Threshold for exit signal (percentage difference)
        exit_method : str
            Exit method: 'threshold' or 'hold_candles'
        hold_candles : int
            Number of candles to hold position (only used if exit_method='hold_candles')
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.exit_method = exit_method
        self.hold_candles = hold_candles
        
    def calculate_moving_averages(self, data: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calculate fast and slow moving averages
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with OHLC columns
        price_col : str
            Column name for price data
            
        Returns:
        --------
        pd.DataFrame
            Data with moving averages added
        """
        df = data.copy()
        df['MA_fast'] = df[price_col].rolling(window=self.fast_period).mean()
        df['MA_slow'] = df[price_col].rolling(window=self.slow_period).mean()
        
        return df
    
    def calculate_percentage_difference(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage difference normalization
        Zt = (MAfast(t) - MAslow(t)) / MAslow(t) * 100
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with moving averages
            
        Returns:
        --------
        pd.DataFrame
            Data with percentage difference added
        """
        df = data.copy()
        
        # Calculate percentage difference
        df['Zt'] = ((df['MA_fast'] - df['MA_slow']) / df['MA_slow']) * 100
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on percentage difference
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with percentage difference calculated
            
        Returns:
        --------
        pd.DataFrame
            Data with signals added
        """
        df = data.copy()
        
        # Initialize signal columns
        df['signal'] = 0
        df['position'] = 0
        df['entry_signal'] = 0
        df['exit_signal'] = 0
        
        # Generate entry signals
        # Long signal when Zt is below negative threshold (oversold)
        df.loc[df['Zt'] < -self.entry_threshold, 'entry_signal'] = 1
        
        # Short signal when Zt is above positive threshold (overbought)
        df.loc[df['Zt'] > self.entry_threshold, 'entry_signal'] = -1
        
        # Generate exit signals based on method
        if self.exit_method == 'threshold':
            # Exit signals when Zt returns to neutral zone
            df.loc[(df['Zt'] > -self.exit_threshold) & (df['Zt'] < self.exit_threshold), 'exit_signal'] = 1
        elif self.exit_method == 'hold_candles':
            # Exit after holding for specified number of candles
            df = self._generate_hold_candles_exit(df)
        
        # For mean reversion, we need to handle position changes properly
        # Entry signal: +1 for long, -1 for short
        # Exit signal: should close the current position
        
        # Initialize position tracking
        df['position'] = 0
        current_position = 0
        
        for i in range(len(df)):
            # Handle entry signals
            if df.iloc[i]['entry_signal'] != 0:
                current_position = df.iloc[i]['entry_signal']
            
            # Handle exit signals
            if df.iloc[i]['exit_signal'] != 0:
                current_position = 0
            
            df.iloc[i, df.columns.get_loc('position')] = current_position
        
        # Create signal column for analysis
        df['signal'] = df['entry_signal'] - df['exit_signal']
        
        return df
    
    def _generate_hold_candles_exit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate exit signals based on holding for specific number of candles
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with entry signals
            
        Returns:
        --------
        pd.DataFrame
            Data with hold-based exit signals
        """
        df = data.copy()
        df['exit_signal'] = 0
        
        # Track entry positions
        current_position = 0
        entry_candle_count = 0
        
        for i in range(len(df)):
            # Check for entry signals
            if df.iloc[i]['entry_signal'] != 0:
                current_position = df.iloc[i]['entry_signal']
                entry_candle_count = 0
            
            # Count candles since entry
            if current_position != 0:
                entry_candle_count += 1
                
                # Exit after holding for specified candles
                if entry_candle_count >= self.hold_candles:
                    # Set exit signal to 1 (will be handled in main signal logic)
                    df.iloc[i, df.columns.get_loc('exit_signal')] = 1
                    current_position = 0
                    entry_candle_count = 0
        
        return df
    
    def calculate_returns(self, data: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calculate strategy returns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with signals
        price_col : str
            Column name for price data
            
        Returns:
        --------
        pd.DataFrame
            Data with returns calculated
        """
        df = data.copy()
        
        # Calculate price returns
        df['price_return'] = df[price_col].pct_change()
        
        # Calculate strategy returns
        # For mean reversion, we want to capture the return when we have a position
        # Position = 1 means long, -1 means short, 0 means no position
        df['strategy_return'] = df['position'].shift(1) * df['price_return']
        
        # Calculate cumulative returns
        df['cumulative_price_return'] = (1 + df['price_return']).cumprod()
        df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()
        
        return df
    
    def backtest(self, data: pd.DataFrame, price_col: str = 'close') -> Dict:
        """
        Run complete backtest of the strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        price_col : str
            Column name for price data
            
        Returns:
        --------
        Dict
            Backtest results
        """
        # Step 1: Calculate moving averages
        df = self.calculate_moving_averages(data, price_col)
        
        # Step 2: Calculate percentage difference
        df = self.calculate_percentage_difference(df)
        
        # Step 3: Generate signals
        df = self.generate_signals(df)
        
        # Step 4: Calculate returns
        df = self.calculate_returns(df, price_col)
        
        # Calculate performance metrics
        total_return = df['cumulative_strategy_return'].iloc[-1] - 1
        total_trades = len(df[df['signal'] != 0])
        
        # Calculate Sharpe ratio
        strategy_returns = df['strategy_return'].dropna()
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = df['cumulative_strategy_return']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'data': df,
            'total_return': total_return,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold
            }
        }
        
        return results
    
    def compare_exit_strategies(self, data: pd.DataFrame, price_col: str = 'close', 
                               hold_candles_list: List[int] = [3, 5, 10, 15]) -> Dict:
        """
        Compare different exit strategies
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        price_col : str
            Column name for price data
        hold_candles_list : List[int]
            List of hold candle periods to test
            
        Returns:
        --------
        Dict
            Comparison results
        """
        comparison_results = {}
        
        # Test threshold-based exit
        print("Testing threshold-based exit strategy...")
        threshold_strategy = MeanReversionStrategy(
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            exit_method='threshold'
        )
        
        threshold_results = threshold_strategy.backtest(data, price_col)
        comparison_results['threshold'] = {
            'results': threshold_results,
            'strategy': threshold_strategy,
            'method': 'threshold'
        }
        
        # Test hold-candles exit strategies
        for hold_candles in hold_candles_list:
            print(f"Testing hold-candles exit strategy ({hold_candles} candles)...")
            hold_strategy = MeanReversionStrategy(
                fast_period=self.fast_period,
                slow_period=self.slow_period,
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
                exit_method='hold_candles',
                hold_candles=hold_candles
            )
            
            hold_results = hold_strategy.backtest(data, price_col)
            comparison_results[f'hold_{hold_candles}'] = {
                'results': hold_results,
                'strategy': hold_strategy,
                'method': f'hold_{hold_candles}_candles'
            }
        
        return comparison_results
    
    def plot_exit_comparison(self, comparison_results: Dict, figsize: Tuple[int, int] = (20, 12)):
        """
        Plot comparison of different exit strategies
        
        Parameters:
        -----------
        comparison_results : Dict
            Results from compare_exit_strategies
        figsize : Tuple[int, int]
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Cumulative Returns Comparison
        ax1 = axes[0, 0]
        for method, data in comparison_results.items():
            df = data['results']['data']
            ax1.plot(df.index, df['cumulative_strategy_return'], 
                    label=f"{data['method']}", alpha=0.8, linewidth=2)
        
        ax1.set_title('Cumulative Returns Comparison - Exit Strategies')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Cumulative Return')
        
        # Plot 2: Performance Metrics Bar Chart
        ax2 = axes[0, 1]
        methods = list(comparison_results.keys())
        total_returns = [comparison_results[method]['results']['total_return'] for method in methods]
        sharpe_ratios = [comparison_results[method]['results']['sharpe_ratio'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, total_returns, width, label='Total Return', alpha=0.8)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Exit Strategy')
        ax2.set_ylabel('Total Return', color='blue')
        ax2_twin.set_ylabel('Sharpe Ratio', color='orange')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([comparison_results[method]['method'] for method in methods], rotation=45)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Max Drawdown Comparison
        ax3 = axes[1, 0]
        max_drawdowns = [comparison_results[method]['results']['max_drawdown'] for method in methods]
        bars3 = ax3.bar(methods, max_drawdowns, alpha=0.8, color='red')
        ax3.set_title('Maximum Drawdown Comparison')
        ax3.set_ylabel('Maximum Drawdown')
        ax3.set_xticklabels([comparison_results[method]['method'] for method in methods], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Total Trades Comparison
        ax4 = axes[1, 1]
        total_trades = [comparison_results[method]['results']['total_trades'] for method in methods]
        bars4 = ax4.bar(methods, total_trades, alpha=0.8, color='green')
        ax4.set_title('Total Trades Comparison')
        ax4.set_ylabel('Number of Trades')
        ax4.set_xticklabels([comparison_results[method]['method'] for method in methods], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
            print("Skipping plot display, but results are still calculated.")
        
        # Print detailed comparison table
        print("\n" + "="*80)
        print("EXIT STRATEGY COMPARISON RESULTS")
        print("="*80)
        print(f"{'Strategy':<20} {'Total Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15} {'Total Trades':<15}")
        print("-"*80)
        
        for method, data in comparison_results.items():
            results = data['results']
            print(f"{data['method']:<20} {results['total_return']:<15.2%} {results['sharpe_ratio']:<15.3f} {results['max_drawdown']:<15.2%} {results['total_trades']:<15}")
        
        print("="*80)
        
        # Find best performing strategy
        best_return = max(comparison_results.items(), key=lambda x: x[1]['results']['total_return'])
        best_sharpe = max(comparison_results.items(), key=lambda x: x[1]['results']['sharpe_ratio'])
        best_drawdown = min(comparison_results.items(), key=lambda x: x[1]['results']['max_drawdown'])
        
        print(f"\nBest Total Return: {best_return[1]['method']} ({best_return[1]['results']['total_return']:.2%})")
        print(f"Best Sharpe Ratio: {best_sharpe[1]['method']} ({best_sharpe[1]['results']['sharpe_ratio']:.3f})")
        print(f"Best Max Drawdown: {best_drawdown[1]['method']} ({best_drawdown[1]['results']['max_drawdown']:.2%})")
    
    def plot_results(self, results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        figsize : Tuple[int, int]
            Figure size
        """
        df = results['data']
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Price and Moving Averages
        axes[0].plot(df.index, df['close'], label='Price', alpha=0.7)
        axes[0].plot(df.index, df['MA_fast'], label=f'MA Fast ({self.fast_period})', alpha=0.8)
        axes[0].plot(df.index, df['MA_slow'], label=f'MA Slow ({self.slow_period})', alpha=0.8)
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Percentage Difference and Signals
        axes[1].plot(df.index, df['Zt'], label='Zt (Percentage Difference)', color='blue', alpha=0.7)
        axes[1].axhline(y=self.entry_threshold, color='red', linestyle='--', alpha=0.7, label='Entry Threshold')
        axes[1].axhline(y=-self.entry_threshold, color='red', linestyle='--', alpha=0.7)
        axes[1].axhline(y=self.exit_threshold, color='green', linestyle='--', alpha=0.7, label='Exit Threshold')
        axes[1].axhline(y=-self.exit_threshold, color='green', linestyle='--', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark signals
        long_signals = df[df['signal'] == 1]
        short_signals = df[df['signal'] == -1]
        
        if not long_signals.empty:
            axes[1].scatter(long_signals.index, long_signals['Zt'], 
                           color='green', marker='^', s=50, label='Long Signal', zorder=5)
        if not short_signals.empty:
            axes[1].scatter(short_signals.index, short_signals['Zt'], 
                           color='red', marker='v', s=50, label='Short Signal', zorder=5)
        
        axes[1].set_title('Percentage Difference and Trading Signals')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Returns
        axes[2].plot(df.index, df['cumulative_price_return'], label='Buy & Hold', alpha=0.7)
        axes[2].plot(df.index, df['cumulative_strategy_return'], label='Strategy', alpha=0.7)
        axes[2].set_title('Cumulative Returns Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
            print("Skipping plot display, but results are still calculated.")
        
        # Print performance metrics
        print(f"\n=== Strategy Performance ===")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        print(f"\n=== Strategy Parameters ===")
        for key, value in results['parameters'].items():
            print(f"{key}: {value}")

def load_sample_data(file_path: str) -> pd.DataFrame:
    """
    Load sample data for testing
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure we have the required columns
        required_cols = ['close']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns. Available columns: {df.columns.tolist()}")
            return None
            
        # Convert to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Initialize strategy for comparison
    strategy = MeanReversionStrategy(
        fast_period=5,
        slow_period=10,
        entry_threshold=0.3,  # Reduced from 2.0 to 0.3 for 3-min gold
        exit_threshold=0.1    # Reduced from 0.5 to 0.1 for 3-min gold
    )
    
    # Load data (you can replace this with your data file)
    data_file = "xauusd_3min_standard.csv"
    data = load_sample_data(data_file)
    
    if data is not None:
        print("="*60)
        print("MEAN REVERSION STRATEGY - EXIT STRATEGY COMPARISON")
        print("="*60)
        
        # Compare different exit strategies
        comparison_results = strategy.compare_exit_strategies(
            data, 
            hold_candles_list=[3, 5, 10, 15, 20]
        )
        
        # Plot comparison results
        strategy.plot_exit_comparison(comparison_results)
        
        # Save comparison results
        comparison_df = pd.DataFrame({
            'strategy': [comparison_results[method]['method'] for method in comparison_results.keys()],
            'total_return': [comparison_results[method]['results']['total_return'] for method in comparison_results.keys()],
            'sharpe_ratio': [comparison_results[method]['results']['sharpe_ratio'] for method in comparison_results.keys()],
            'max_drawdown': [comparison_results[method]['results']['max_drawdown'] for method in comparison_results.keys()],
            'total_trades': [comparison_results[method]['results']['total_trades'] for method in comparison_results.keys()]
        })
        
        comparison_df.to_csv('exit_strategy_comparison.csv', index=False)
        print(f"\nComparison results saved to 'exit_strategy_comparison.csv'")
        
        # Also save individual strategy results
        for method, data_dict in comparison_results.items():
            filename = f"mean_reversion_{data_dict['method']}_results.csv"
            data_dict['results']['data'].to_csv(filename)
            print(f"Results for {data_dict['method']} saved to '{filename}'")
        
    else:
        print("Could not load data. Please check the file path and format.")
