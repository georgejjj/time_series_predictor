import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

from models import LinearModel, LSTMModel, TransformerModel
from utils import download_fred_data, calculate_returns, plot_predictions, calculate_metrics

# 设置页面配置
st.set_page_config(
    page_title="金融指数预测系统",
    page_icon="📈",
    layout="wide"
)

# 设置页面标题
st.title("金融指数收益率预测系统")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("参数设置")
    
    # 选择指数
    index_options = {
        "S&P 500": "SP500",
        "VIX": "VIXCLS", 
        "道琼斯工业平均指数": "DJIA",
        "纳斯达克综合指数": "NASDAQCOM",
        "10年期国债收益率": "GS10",
        "美元指数": "DTWEXBGS"
    }
    
    selected_index = st.selectbox(
        "选择要预测的指数",
        list(index_options.keys())
    )
    
    fred_series_id = index_options[selected_index]
    
    # 获取数据时间范围
    st.subheader("数据范围")
    start_date = st.date_input("开始日期", datetime.now() - relativedelta(years=10))
    end_date = st.date_input("结束日期", datetime.now())
    
    # 选择返回类型
    return_type = st.selectbox(
        "返回类型",
        ["日收益率", "周收益率", "月收益率"]
    )
    
    # 模型选择
    st.subheader("模型选择")
    models_to_use = st.multiselect(
        "选择要使用的模型",
        ["线性模型", "LSTM", "Transformer"],
        default=["线性模型"]
    )
    
    # 参数设置
    st.subheader("训练参数")
    train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.8, 0.05)
    
    forecast_horizon = st.number_input("预测期数", 1, 30, 5)
    
    lookback_periods = st.number_input("使用多少历史数据点进行预测", 5, 100, 20)
    
    # 特定模型参数
    st.subheader("模型参数")
    
    if "线性模型" in models_to_use:
        st.markdown("#### 线性模型参数")
        linear_method = st.selectbox(
            "方法",
            ["OLS", "Ridge", "Lasso"],
            key="linear_method"
        )
    
    if "LSTM" in models_to_use:
        st.markdown("#### LSTM参数")
        lstm_units = st.slider("LSTM单元数", 10, 200, 50, 10, key="lstm_units")
        lstm_epochs = st.slider("训练轮数", 10, 500, 100, 10, key="lstm_epochs")
        lstm_batch = st.slider("批次大小", 8, 128, 32, 8, key="lstm_batch")
    
    if "Transformer" in models_to_use:
        st.markdown("#### Transformer参数")
        transformer_layers = st.slider("Transformer层数", 1, 6, 2, 1, key="transformer_layers")
        transformer_heads = st.slider("注意力头数", 1, 8, 4, 1, key="transformer_heads")
        transformer_epochs = st.slider("训练轮数", 10, 500, 100, 10, key="transformer_epochs")

# 主程序逻辑
if st.button("开始预测"):
    try:
        with st.spinner("正在从FRED下载数据..."):
            # 下载数据
            df = download_fred_data(fred_series_id, start_date, end_date)
            
            # 计算收益率
            freq_map = {"日收益率": "D", "周收益率": "W", "月收益率": "M"}
            returns_df = calculate_returns(df, freq_map[return_type])
            
            st.success(f"成功下载 {selected_index} 的数据！")
            
            # 显示原始数据
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原始价格数据")
                st.line_chart(df)
            
            with col2:
                st.subheader(f"{return_type}数据")
                st.line_chart(returns_df)
            
            # 准备训练和测试数据
            train_size = int(len(returns_df) * train_ratio)
            train_data = returns_df.iloc[:train_size]
            test_data = returns_df.iloc[train_size:]
            
            # 创建用于显示结果的列
            model_cols = st.columns(len(models_to_use))
            all_predictions = {}
            
            # 运行所选的模型
            for i, model_name in enumerate(models_to_use):
                with model_cols[i]:
                    st.subheader(f"{model_name}预测结果")
                    
                    with st.spinner(f"正在训练{model_name}..."):
                        if model_name == "线性模型":
                            model = LinearModel(method=linear_method, lookback=lookback_periods)
                        elif model_name == "LSTM":
                            model = LSTMModel(
                                lookback=lookback_periods,
                                units=lstm_units,
                                epochs=lstm_epochs,
                                batch_size=lstm_batch
                            )
                        elif model_name == "Transformer":
                            model = TransformerModel(
                                lookback=lookback_periods,
                                layers=transformer_layers,
                                heads=transformer_heads,
                                epochs=transformer_epochs
                            )
                        
                        # 训练模型
                        model.fit(train_data.values)
                        
                        # 在测试集上预测
                        y_pred = model.predict(returns_df.values, train_size, forecast_horizon)
                        
                        # 保存预测结果
                        all_predictions[model_name] = y_pred
                        
                        # 显示预测图表
                        fig = plot_predictions(test_data.values, y_pred, test_data.index)
                        st.pyplot(fig)
                        
                        # 计算并显示评估指标
                        metrics = calculate_metrics(test_data.values[:len(y_pred)], y_pred)
                        
                        metrics_df = pd.DataFrame({
                            "指标": list(metrics.keys()),
                            "值": list(metrics.values())
                        })
                        
                        st.table(metrics_df)
            
            # 模型比较（如果选择了多个模型）
            if len(models_to_use) > 1:
                st.subheader("模型比较")
                
                # 创建比较表格
                comparison_metrics = {}
                
                for model_name in models_to_use:
                    pred = all_predictions[model_name]
                    metrics = calculate_metrics(test_data.values[:len(pred)], pred)
                    comparison_metrics[model_name] = metrics
                
                # 转换为DataFrame进行显示
                comparison_df = pd.DataFrame(comparison_metrics)
                st.table(comparison_df)
                
                # 创建模型比较图
                st.subheader("预测结果比较")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 确保预测长度不超过测试数据可用长度
                plot_length = min(forecast_horizon, len(test_data))
                
                # 绘制实际值
                ax.plot(test_data.index[:plot_length], test_data.values[:plot_length].flatten(), 
                        label='实际值', color='black', linestyle='--')
                
                # 绘制各模型预测值
                colors = ['blue', 'red', 'green']
                for i, model_name in enumerate(models_to_use):
                    # 确保预测结果的长度不超过可视化所需的长度
                    model_pred = all_predictions[model_name]
                    if len(model_pred) > plot_length:
                        model_pred = model_pred[:plot_length]
                    
                    ax.plot(test_data.index[:len(model_pred)], model_pred, 
                            label=f'{model_name}预测', color=colors[i % len(colors)])
                
                ax.set_title('模型预测比较')
                ax.set_xlabel('日期')
                ax.set_ylabel(return_type)
                ax.legend()
                ax.grid(True)
                
                # 旋转x轴标签以避免重叠
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"发生错误: {str(e)}")

else:
    # 显示使用说明
    st.info("""
    ### 使用说明:
    1. 在侧边栏选择要预测的金融指数
    2. 设置数据日期范围和收益率类型
    3. 选择一个或多个预测模型
    4. 调整模型参数
    5. 点击"开始预测"按钮
    """) 