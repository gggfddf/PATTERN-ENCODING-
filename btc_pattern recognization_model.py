import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load your data (tab-separated format)
df = pd.read_csv('btc_5min.csv', sep='\t')
# Combine DATE and TIME columns to create timestamp
df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
# Rename columns to lowercase without brackets for easier access
df = df.rename(columns={
    '<OPEN>': 'open',
    '<HIGH>': 'high', 
    '<LOW>': 'low',
    '<CLOSE>': 'close',
    '<TICKVOL>': 'volume'
})

# Configurable parameters for 5-minute intervals
pattern_len = 10      # 10 candles = 50 minutes of pattern data
move_len = 5         # 5 candles = 25 minutes prediction window
move_threshold = 0.001  # 0.1% move (higher threshold for longer timeframe)
min_accuracy = 0.8     # Slightly higher accuracy requirement
min_occurrences = 3     # More occurrences needed for reliability

# Helper: simplified & generalized encoding for patterns
def encode_pattern(candles):
    pattern_features = []
    for i, c in candles.iterrows():
        # Use direction and rounded close change for generalization
        pct_close = round((c['close'] - c['open']) / c['open'], 2)  # round to 2 decimals
        direction = 1 if c['close'] > c['open'] else 0
        pattern_features += [direction, pct_close]
    return tuple(pattern_features)

def check_move(candles):
    pct_move = (candles.iloc[-1]['close'] - candles.iloc[0]['open']) / candles.iloc[0]['open']
    if pct_move > move_threshold:
        return 'up'
    elif pct_move < -move_threshold:
        return 'down'
    else:
        return None

pattern_stats = defaultdict(lambda: {'up': 0, 'down': 0, 'total': 0})

move_up_count = 0
move_down_count = 0

# Training loop with progress bar
for i in tqdm(range(pattern_len, len(df) - move_len)):
    pattern_candles = df.iloc[i-pattern_len:i]
    move_candles = df.iloc[i:i+move_len]
    pattern = encode_pattern(pattern_candles)
    move = check_move(move_candles)
    if move:
        if move == 'up':
            move_up_count += 1
        else:
            move_down_count += 1
        pattern_stats[pattern][move] += 1
        pattern_stats[pattern]['total'] += 1

print(f"Total up moves detected: {move_up_count}")
print(f"Total down moves detected: {move_down_count}")
print(f"Unique patterns observed: {len(pattern_stats)}")

# Filter patterns by accuracy
high_acc_patterns = {}
accuracies = []
occurrences = []
for pattern, stats in pattern_stats.items():
    for direction in ['up', 'down']:
        acc = stats[direction] / stats['total'] if stats['total'] else 0
        if acc >= min_accuracy and stats['total'] >= min_occurrences:
            high_acc_patterns[(pattern, direction)] = acc
            accuracies.append(acc)
            occurrences.append(stats['total'])

print(f"Patterns kept after filtering: {len(high_acc_patterns)}")

# Save learned patterns
with open('btc_patterns.pkl', 'wb') as f:
    pickle.dump(high_acc_patterns, f)

# Visualization
if accuracies:
    plt.figure(figsize=(12,6))
    plt.hist(accuracies, bins=20, color='dodgerblue', alpha=0.7)
    plt.title('Distribution of Pattern Accuracies (Kept Patterns)')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Patterns')
    plt.show()
else:
    print("No patterns to plot accuracy distribution.")

if occurrences:
    plt.figure(figsize=(12,6))
    plt.hist(occurrences, bins=20, color='orange', alpha=0.7)
    plt.title('Distribution of Pattern Occurrences (Kept Patterns)')
    plt.xlabel('Occurrences')
    plt.ylabel('Number of Patterns')
    plt.show()
else:
    print("No patterns to plot occurrence distribution.")

# Confusion matrix (actual vs predicted)
actuals, preds = [], []
for i in range(pattern_len, len(df) - move_len):
    pattern_candles = df.iloc[i-pattern_len:i]
    move_candles = df.iloc[i:i+move_len]
    pattern = encode_pattern(pattern_candles)
    move = check_move(move_candles)
    if move:
        for direction in ['up', 'down']:
            if (pattern, direction) in high_acc_patterns:
                preds.append(direction)
                actuals.append(move)
                break

if preds:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(actuals, preds, labels=['up','down'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['up','down'])
    disp.plot()
    plt.title('Confusion Matrix: Pattern Prediction vs Actual')
    plt.show()
else:
    print("No predictions to plot confusion matrix.")

# For debug/testing: Save summary statistics to text
with open('training_summary.txt', 'w') as f:
    f.write(f"Total up moves detected: {move_up_count}\n")
    f.write(f"Total down moves detected: {move_down_count}\n")
    f.write(f"Unique patterns observed: {len(pattern_stats)}\n")
    f.write(f"Patterns kept after filtering: {len(high_acc_patterns)}\n")