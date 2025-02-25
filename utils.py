import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from datetime import datetime

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def download_fred_data(series_id, start_date, end_date):
    """
    从FRED下载金融数据（使用pandas_datareader）
    
    参数:
    series_id: 需要下载的数据系列ID
    start_date: 开始日期
    end_date: 结束日期
    
    返回:
    pandas DataFrame 包含下载的数据
    """
    # 使用pandas_datareader从FRED下载数据
    df = web.DataReader(series_id, 'fred', start_date, end_date)
    
    # 重命名列为'Value'以保持与原来的接口一致
    df.columns = ['Value']
    
    # 删除缺失值
    df = df.dropna()
    
    return df

def calculate_returns(df, freq='D'):
    """
    计算收益率
    
    参数:
    df: 原始价格数据
    freq: 频率 - 'D'每日, 'W'每周, 'M'每月
    
    返回:
    收益率DataFrame
    """
    # 根据频率重采样
    if freq != 'D':
        df = df.resample(freq).last()
    
    # 计算收益率 (简单百分比变化)
    returns = df.pct_change().dropna()
    
    return returns

def plot_predictions(actual, predicted, dates):
    """
    绘制预测结果与实际值的对比图
    
    参数:
    actual: 实际值数组
    predicted: 预测值数组
    dates: 日期索引
    
    返回:
    matplotlib figure对象
    """
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 确保输入是一维数组
    if isinstance(actual, pd.DataFrame):
        actual = actual.values
    if isinstance(predicted, pd.DataFrame):
        predicted = predicted.values
        
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    # 确保预测值的长度不超过实际值的长度
    n = min(len(actual), len(predicted))
    
    # 确保dates长度足够
    if len(dates) < n:
        # 如果日期不足，只使用可用的数据点
        n = len(dates)
        actual = actual[:n]
        predicted = predicted[:n]
    
    # 绘制实际值和预测值
    ax.plot(dates[:n], actual[:n], label='实际值', marker='o', linestyle='-', color='blue')
    ax.plot(dates[:n], predicted[:n], label='预测值', marker='x', linestyle='--', color='red')
    
    # 添加图例和标签
    ax.set_title('预测结果与实际值比较')
    ax.set_xlabel('日期')
    ax.set_ylabel('收益率')
    ax.legend()
    ax.grid(True)
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def calculate_metrics(actual, predicted):
    """
    计算模型评估指标
    
    参数:
    actual: 实际值数组
    predicted: 预测值数组
    
    返回:
    包含评估指标的字典
    """
    # 确保输入是一维数组
    if isinstance(actual, pd.DataFrame):
        actual = actual.values
    if isinstance(predicted, pd.DataFrame):
        predicted = predicted.values
    
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    # 确保长度相同
    n = min(len(actual), len(predicted))
    actual = actual[:n]
    predicted = predicted[:n]
    
    # 计算指标
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # 处理零值情况以避免MAPE中的除零错误
    if np.any(np.abs(actual) < 1e-10):
        mape = np.nan
    else:
        try:
            mape = mean_absolute_percentage_error(actual, predicted) * 100
        except:
            mape = np.nan
    
    # 防止r2计算出nan或inf（如果所有实际值相同）
    try:
        r2 = r2_score(actual, predicted)
    except:
        r2 = np.nan
    
    # 计算平均绝对误差
    mae = np.mean(np.abs(actual - predicted))
    
    # 计算准确方向比例（预测变化方向与实际一致的比例）
    try:
        actual_diff = np.diff(actual)
        pred_diff = np.diff(predicted)
        if len(actual_diff) > 0:
            dir_matches = (actual_diff * pred_diff) > 0
            dir_accuracy = np.mean(dir_matches) * 100
        else:
            dir_accuracy = np.nan
    except:
        dir_accuracy = np.nan
    
    return {
        "RMSE": round(rmse, 6) if not np.isnan(rmse) else np.nan,
        "MAPE (%)": round(mape, 2) if not np.isnan(mape) else np.nan,
        "MAE": round(mae, 6) if not np.isnan(mae) else np.nan,
        "R²": round(r2, 4) if not np.isnan(r2) else np.nan,
        "方向准确率 (%)": round(dir_accuracy, 2) if not np.isnan(dir_accuracy) else np.nan
    } 