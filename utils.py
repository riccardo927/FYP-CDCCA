import yfinance as yf
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].dropna()

def calculate_changes(price_data):
    return price_data.diff()#.dropna()

def calculate_daily_returns(price_data):
    return price_data.pct_change()#.dropna()

def calculate_rate_of_change(daily_returns):
    return daily_returns.diff()#.dropna()

def create_complex_numbers(real_part, imaginary_part):
    return real_part + 1j * imaginary_part

def load_data(name, start_date = '1992-01-01', end_date = '2023-01-01'):
    print("loading data ...")
    data = fetch_data(name, start_date, end_date)
    daily_returns = calculate_daily_returns(data)
    roc = calculate_rate_of_change(daily_returns)

    diff = data.pct_change()

    data_norm = (data-data.min())/(data.max()-data.min())
    diff_norm = (diff-diff.min())/(diff.max()-diff.min())

    aligned_data = pd.concat([data_norm[1:], diff_norm], axis=1).dropna()
    aligned_data.columns = ['Close', 'Change']

    complex_data = (create_complex_numbers(aligned_data['Close'], aligned_data['Change'])).to_numpy()

    return complex_data


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    data_length = len(data)
    train_end = int(data_length * train_ratio)
    val_end = int(data_length * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return torch.tensor(train_data).unsqueeze(1), torch.tensor(val_data).unsqueeze(1), torch.tensor(test_data).unsqueeze(1)

def plot_data(tick1, tick2, start, end):
    data1 = fetch_data(tick1, start, end)
    data2 = fetch_data(tick2, start, end)

    data1_norm = (data1-data1.mean())/data1.std()
    data2_norm = (data2-data2.mean())/data2.std()

    train_ratio=0.7
    val_ratio=0.15
    data_length = len(data1)
    train_end = int(data_length * train_ratio)
    val_end = int(data_length * (train_ratio + val_ratio))

    idx1 = data1.index
    idx2 = data2.index

    # Plot complex data
    plt.figure(figsize=(10,9))
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.4, wspace = 0.2)

    plt.subplot(3, 2, 1)
    plt.plot(data1)
    plt.axvline(x = idx1[train_end], alpha = 0.8, color='r', linewidth = 2, linestyle = '--')
    plt.axvline(x = idx1[val_end], alpha = 0.8, color='g', linewidth = 2, linestyle = '--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Data1')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(data2)
    plt.axvline(x = idx2[train_end], alpha = 0.8, color='r', linewidth = 2, linestyle = '--')
    plt.axvline(x = idx2[val_end], alpha = 0.8, color='g', linewidth = 2, linestyle = '--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Data2')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(data1.pct_change())
    plt.axvline(x = idx2[train_end], alpha = 0.8, color='r', linewidth = 2, linestyle = '--')
    plt.axvline(x = idx2[val_end], alpha = 0.8, color='g', linewidth = 2, linestyle = '--')
    plt.xlabel('Date')
    plt.ylabel('ROC')
    plt.title('Data1')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(data2.pct_change())
    plt.axvline(x = idx2[train_end], alpha = 0.8, color='r', linewidth = 2, linestyle = '--')
    plt.axvline(x = idx2[val_end], alpha = 0.8, color='g', linewidth = 2, linestyle = '--')
    plt.xlabel('Date')
    plt.ylabel('ROC')
    plt.title('Data2')
    plt.grid(True)

    view1 = load_data(tick1, start, end)
    view2 = load_data(tick2, start, end)

    circ_view1 = np.abs(np.mean(view1**2)/(np.mean(np.abs(view1**2)))); # Circularity coefficient of data
    plt.subplot(3, 2, 5)
    plt.scatter(view1.real, view1.imag, marker='o', alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('View 1 - Complex Data, \u03B7 ={}'.format(round(circ_view1, 4)))
    plt.grid(True)

    circ_view2 = np.abs(np.mean(view2**2)/(np.mean(np.abs(view2**2)))); # Circularity coefficient of data
    plt.subplot(3, 2, 6)
    plt.scatter(view2.real, view2.imag, marker='o', alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('View 2 - Complex Data, \u03B7 ={}'.format(round(circ_view2, 4)))
    plt.grid(True)

    plt.show()
