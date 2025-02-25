import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 基础模型类
class BaseModel:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.scaler = StandardScaler()
        
    def create_sequences(self, data):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback])
        return np.array(X), np.array(y)
    
    def fit(self, data):
        """训练模型"""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, data, start_idx, horizon):
        """预测未来值"""
        raise NotImplementedError("Subclasses must implement predict()")

# 线性模型
class LinearModel(BaseModel):
    def __init__(self, method='OLS', lookback=20):
        super().__init__(lookback)
        self.method = method
        
        # 基于方法选择模型
        if method == 'OLS':
            self.model = LinearRegression()
        elif method == 'Ridge':
            self.model = Ridge(alpha=1.0)
        elif method == 'Lasso':
            self.model = Lasso(alpha=0.1)
        else:
            self.model = LinearRegression()
            
    def fit(self, data):
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        
        # 重塑为线性模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 训练模型
        self.model.fit(X_reshaped, y)
        
    def predict(self, data, start_idx, horizon):
        # 数据标准化
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # 存储预测
        predictions = []
        
        # 开始索引
        idx = start_idx - self.lookback
        
        # 预测未来horizon个值
        for _ in range(horizon):
            # 获取输入序列
            input_seq = scaled_data[idx:idx+self.lookback]
            input_seq_reshaped = input_seq.reshape(1, -1)
            
            # 进行预测
            pred = self.model.predict(input_seq_reshaped)[0]
            
            # 添加到预测列表
            predictions.append(pred)
            
            # 更新输入数据以进行下一次预测
            scaled_data = np.append(scaled_data, pred)
            idx += 1
            
        # 反向标准化预测
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions

# LSTM模型
class LSTMModel(BaseModel):
    def __init__(self, lookback=20, units=50, epochs=100, batch_size=32):
        super().__init__(lookback)
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=self.units//2),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def fit(self, data):
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        
        # 重塑为LSTM输入 [样本, 时间步, 特征]
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        # 构建模型
        self.model = self.build_model((self.lookback, 1))
        
        # 早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 训练模型
        self.model.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
    def predict(self, data, start_idx, horizon):
        # 数据标准化
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # 存储预测
        predictions = []
        
        # 开始索引
        idx = start_idx - self.lookback
        
        # 预测未来horizon个值
        for _ in range(horizon):
            # 获取输入序列
            input_seq = scaled_data[idx:idx+self.lookback]
            input_seq_reshaped = input_seq.reshape(1, self.lookback, 1)
            
            # 进行预测
            pred = self.model.predict(input_seq_reshaped, verbose=0)[0][0]
            
            # 添加到预测列表
            predictions.append(pred)
            
            # 更新输入数据以进行下一次预测
            scaled_data = np.append(scaled_data, pred)
            idx += 1
            
        # 反向标准化预测
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions

# 改进的Transformer模型
class TransformerModel(BaseModel):
    def __init__(self, lookback=20, layers=2, heads=4, epochs=100):
        super().__init__(lookback)
        self.layers = layers
        self.heads = heads
        self.epochs = epochs
        self.model = None
        
    def positional_encoding(self, length, depth):
        """
        为Transformer模型创建位置编码
        """
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # 多头注意力
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = Dropout(dropout)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # 前馈网络
        ff_output = Dense(ff_dim, activation="relu")(attention_output)
        ff_output = Dense(inputs.shape[-1])(ff_output)
        ff_output = Dropout(dropout)(ff_output)
        return LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
    
    def build_model(self):
        # 输入层
        inputs = Input(shape=(self.lookback, 1))
        
        # 嵌入层，将输入从1维扩展到64维
        embedding_dim = 64
        x = Dense(embedding_dim)(inputs)
        
        # 添加位置编码
        positions = self.positional_encoding(self.lookback, embedding_dim)
        positions = tf.reshape(positions, (1, self.lookback, embedding_dim))
        x = x + positions
        
        # 添加多个Transformer层
        for _ in range(self.layers):
            x = self.transformer_encoder(
                x, 
                head_size=16, 
                num_heads=self.heads, 
                ff_dim=embedding_dim * 2, 
                dropout=0.1
            )
        
        # 全局平均池化代替只取最后一个时间步，捕获整个序列的信息
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 添加几个全连接层，逐渐减小维度
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.1)(x)
        
        # 输出层
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse"
        )
        return model
        
    def fit(self, data):
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        
        # 重塑为Transformer输入 [样本, 时间步, 特征]
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        # 构建模型
        self.model = self.build_model()
        
        # 早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # 增加patience值
            restore_best_weights=True
        )
        
        # 学习率调度器
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,
            patience=5, 
            min_lr=0.0001
        )
        
        # 训练模型
        history = self.model.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
    def predict(self, data, start_idx, horizon):
        # 数据标准化
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # 存储预测
        predictions = []
        
        # 开始索引
        idx = start_idx - self.lookback
        
        # 预测未来horizon个值
        for _ in range(horizon):
            # 获取输入序列
            input_seq = scaled_data[idx:idx+self.lookback]
            input_seq_reshaped = input_seq.reshape(1, self.lookback, 1)
            
            # 进行预测
            pred = self.model.predict(input_seq_reshaped, verbose=0)[0][0]
            
            # 添加到预测列表
            predictions.append(pred)
            
            # 更新输入数据以进行下一次预测
            scaled_data = np.append(scaled_data, pred)
            idx += 1
            
        # 反向标准化预测
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions 