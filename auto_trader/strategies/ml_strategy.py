"""
机器学习策略模块

基于简单机器学习模型的交易策略：
- 使用技术指标作为特征
- 简单的线性回归和分类模型
- 在线学习和模型更新
- 特征工程和数据预处理
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 尝试导入scikit-learn，如果没有则使用简单的自实现模型
try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderType


class SimpleMLModel:
    """简单的机器学习模型实现（当sklearn不可用时）"""
    
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.weights = None
        self.bias = 0.0
        self.scaler_mean = None
        self.scaler_std = None
    
    def _sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _standardize(self, X):
        """标准化特征"""
        if self.scaler_mean is None:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-8
        
        return (X - self.scaler_mean) / self.scaler_std
    
    def fit(self, X, y):
        """训练模型"""
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0 or len(y) == 0:
            return self
        
        # 标准化特征
        X_scaled = self._standardize(X)
        
        # 初始化权重
        if self.weights is None:
            self.weights = np.random.normal(0, 0.01, X_scaled.shape[1])
        
        # 简单的梯度下降
        learning_rate = 0.01
        epochs = 100
        
        for _ in range(epochs):
            if self.model_type == 'logistic':
                # 逻辑回归
                predictions = self._sigmoid(np.dot(X_scaled, self.weights) + self.bias)
                error = y - predictions
                
                # 更新权重和偏置
                self.weights += learning_rate * np.dot(X_scaled.T, error) / len(X)
                self.bias += learning_rate * np.mean(error)
                
            else:
                # 线性回归
                predictions = np.dot(X_scaled, self.weights) + self.bias
                error = y - predictions
                
                # 更新权重和偏置
                self.weights += learning_rate * np.dot(X_scaled.T, error) / len(X)
                self.bias += learning_rate * np.mean(error)
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.weights is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        if self.model_type == 'logistic':
            predictions = self._sigmoid(np.dot(X_scaled, self.weights) + self.bias)
            return (predictions > 0.5).astype(int)
        else:
            return np.dot(X_scaled, self.weights) + self.bias
    
    def predict_proba(self, X):
        """预测概率"""
        if self.weights is None:
            return np.zeros((len(X), 2))
        
        X = np.array(X)
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        if self.model_type == 'logistic':
            pos_prob = self._sigmoid(np.dot(X_scaled, self.weights) + self.bias)
            return np.column_stack([1 - pos_prob, pos_prob])
        else:
            predictions = np.dot(X_scaled, self.weights) + self.bias
            probs = self._sigmoid(predictions)
            return np.column_stack([1 - probs, probs])


class MLTradingStrategy(Strategy):
    """
    机器学习交易策略实现
    
    策略逻辑：
    1. 提取技术指标作为特征
    2. 使用历史数据训练预测模型
    3. 预测未来价格方向或价格变化
    4. 基于预测结果生成交易信号
    5. 定期重新训练模型
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化机器学习策略
        
        Args:
            config: 策略配置对象
        """
        super().__init__(config)
        
        # 策略参数
        self.lookback_period = self.parameters.get('lookback_period', 100)    # 训练数据回望周期
        self.prediction_horizon = self.parameters.get('prediction_horizon', 1)  # 预测时间范围
        self.retrain_interval = self.parameters.get('retrain_interval', 50)   # 重新训练间隔
        self.min_training_samples = self.parameters.get('min_training_samples', 50)  # 最小训练样本数
        self.confidence_threshold = self.parameters.get('confidence_threshold', 0.6)  # 预测置信度阈值
        self.position_size = self.parameters.get('position_size', 0.15)       # 仓位大小
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.03)       # 止损百分比
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.06)   # 止盈百分比
        
        # 模型参数
        self.model_type = self.parameters.get('model_type', 'logistic')       # 模型类型
        self.use_price_prediction = self.parameters.get('use_price_prediction', False)  # 是否预测价格变化
        self.feature_list = self.parameters.get('features', [
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'bb_position', 
            'volume_ratio', 'price_change', 'volatility'
        ])
        
        # 内部状态
        self.model = None                                                      # 预测模型
        self.scaler = None                                                     # 特征缩放器
        self.training_data = []                                                # 训练数据缓存
        self.last_retrain_time = None                                          # 最后重训练时间
        self.feature_data = pd.DataFrame()                                     # 特征数据
        self.prediction_accuracy = 0.0                                         # 预测准确率
        self.entry_price = 0.0                                                # 入场价格
        self.last_prediction = None                                            # 最后预测结果
        self.prediction_confidence = 0.0                                       # 预测置信度
        
        # 技术指标缓存
        self.indicators_cache = {}
        
        print(f"初始化机器学习策略: {self.name}")
        print(f"交易对: {self.symbol}")
        print(f"模型类型: {self.model_type}, sklearn可用: {SKLEARN_AVAILABLE}")
        print(f"特征列表: {self.feature_list}")
        print(f"训练周期: {self.lookback_period}, 重训练间隔: {self.retrain_interval}")
    
    def initialize(self) -> None:
        """初始化策略"""
        self.is_initialized = True
        print("机器学习策略初始化完成")
    
    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标特征
        
        Args:
            data: OHLCV数据
            
        Returns:
            pd.DataFrame: 包含技术指标特征的DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        # 价格特征
        features['price'] = data['close']
        features['price_change'] = data['close'].pct_change()
        features['price_change_5'] = data['close'].pct_change(5)
        features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        
        # 移动平均线
        features['sma_5'] = data['close'].rolling(window=5).mean()
        features['sma_10'] = data['close'].rolling(window=10).mean()
        features['sma_20'] = data['close'].rolling(window=20).mean()
        features['sma_50'] = data['close'].rolling(window=50).mean()
        
        # 指数移动平均线
        features['ema_12'] = data['close'].ewm(span=12).mean()
        features['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_line = features['ema_12'] - features['ema_26']
        features['macd'] = macd_line
        features['macd_signal'] = macd_line.ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 布林带
        bb_middle = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
        
        # 成交量特征
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_ma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma']
            features['volume_change'] = data['volume'].pct_change()
        else:
            features['volume'] = 1.0
            features['volume_ma'] = 1.0
            features['volume_ratio'] = 1.0
            features['volume_change'] = 0.0
        
        # 波动率特征
        features['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
        features['volatility_5'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
        
        # 动量特征
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        # 相对价格位置
        features['price_percentile_20'] = data['close'].rolling(window=20).rank() / 20
        features['price_percentile_50'] = data['close'].rolling(window=50).rank() / 50
        
        # ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(window=14).mean()
        features['atr_ratio'] = features['atr'] / data['close']
        
        # 价格与各移动平均线的关系
        features['price_vs_sma5'] = data['close'] / features['sma_5'] - 1
        features['price_vs_sma20'] = data['close'] / features['sma_20'] - 1
        features['price_vs_ema12'] = data['close'] / features['ema_12'] - 1
        
        return features
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """
        创建目标变量
        
        Args:
            data: OHLCV数据
            
        Returns:
            pd.Series: 目标变量
        """
        if self.use_price_prediction:
            # 预测价格变化百分比
            future_return = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
            return future_return
        else:
            # 预测价格方向（分类）
            future_return = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
            # 转换为分类标签：0=下跌，1=上涨
            target = (future_return > 0.001).astype(int)  # 0.1%阈值
            return target
    
    def prepare_training_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            features: 特征数据
            target: 目标变量
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标向量)
        """
        # 选择指定的特征
        available_features = [f for f in self.feature_list if f in features.columns]
        if not available_features:
            return np.array([]), np.array([])
        
        # 提取特征和目标
        X = features[available_features].copy()
        y = target.copy()
        
        # 删除包含NaN的行
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 转换为numpy数组
        X_array = X.values
        y_array = y.values
        
        return X_array, y_array
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            
        Returns:
            bool: 训练是否成功
        """
        if len(X) < self.min_training_samples:
            return False
        
        try:
            if SKLEARN_AVAILABLE:
                # 使用sklearn模型
                if self.use_price_prediction:
                    # 回归模型
                    self.model = LinearRegression()
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                    self.model.fit(X_scaled, y)
                else:
                    # 分类模型
                    if self.model_type == 'random_forest':
                        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
                        self.model.fit(X, y)
                    else:
                        self.model = LogisticRegression(random_state=42, max_iter=1000)
                        self.scaler = StandardScaler()
                        X_scaled = self.scaler.fit_transform(X)
                        self.model.fit(X_scaled, y)
            else:
                # 使用简单自实现模型
                model_type = 'linear' if self.use_price_prediction else 'logistic'
                self.model = SimpleMLModel(model_type=model_type)
                self.model.fit(X, y)
            
            # 计算训练集准确率（仅分类模型）
            if not self.use_price_prediction:
                predictions = self.predict(X)
                if len(predictions) > 0:
                    self.prediction_accuracy = np.mean(predictions == y)
                    print(f"[ML] 模型训练完成，训练准确率: {self.prediction_accuracy:.3f}")
            else:
                print(f"[ML] 回归模型训练完成，样本数: {len(X)}")
            
            self.last_retrain_time = datetime.now()
            return True
            
        except Exception as e:
            print(f"[ML] 模型训练失败: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.model is None or len(X) == 0:
            return np.array([])
        
        try:
            if SKLEARN_AVAILABLE and hasattr(self.model, 'predict'):
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict(X_scaled)
                else:
                    return self.model.predict(X)
            else:
                return self.model.predict(X)
        except Exception as e:
            print(f"[ML] 预测失败: {e}")
            return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.model is None or len(X) == 0:
            return np.array([])
        
        try:
            if SKLEARN_AVAILABLE and hasattr(self.model, 'predict_proba'):
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict_proba(X_scaled)
                else:
                    return self.model.predict_proba(X)
            else:
                return self.model.predict_proba(X)
        except Exception as e:
            print(f"[ML] 概率预测失败: {e}")
            return np.array([])
    
    def should_retrain(self) -> bool:
        """
        判断是否需要重新训练模型
        
        Returns:
            bool: 是否需要重新训练
        """
        if self.model is None:
            return True
        
        if self.last_retrain_time is None:
            return True
        
        # 检查时间间隔
        if len(self.feature_data) >= self.retrain_interval:
            return True
        
        return False
    
    def generate_trading_signal(self, current_features: pd.Series) -> Optional[SignalType]:
        """
        基于模型预测生成交易信号
        
        Args:
            current_features: 当前特征
            
        Returns:
            Optional[SignalType]: 交易信号
        """
        if self.model is None:
            return None
        
        # 准备特征
        available_features = [f for f in self.feature_list if f in current_features.index]
        if not available_features:
            return None
        
        feature_values = current_features[available_features].values.reshape(1, -1)
        
        # 检查是否有NaN值
        if np.isnan(feature_values).any():
            return None
        
        try:
            if self.use_price_prediction:
                # 回归预测
                prediction = self.predict(feature_values)[0]
                self.last_prediction = prediction
                self.prediction_confidence = min(1.0, abs(prediction) / 0.02)  # 基于预测幅度计算置信度
                
                # 根据预测的价格变化生成信号
                if prediction > 0.005:  # 预测上涨超过0.5%
                    return SignalType.BUY
                elif prediction < -0.005:  # 预测下跌超过0.5%
                    return SignalType.SELL
            
            else:
                # 分类预测
                prediction = self.predict(feature_values)[0]
                probabilities = self.predict_proba(feature_values)[0]
                
                self.last_prediction = prediction
                self.prediction_confidence = max(probabilities)  # 最大概率作为置信度
                
                # 检查置信度
                if self.prediction_confidence < self.confidence_threshold:
                    return None
                
                # 根据分类结果生成信号
                if prediction == 1:  # 预测上涨
                    return SignalType.BUY
                elif prediction == 0:  # 预测下跌
                    return SignalType.SELL
        
        except Exception as e:
            print(f"[ML] 信号生成失败: {e}")
        
        return None
    
    def should_exit_position(self, data: pd.DataFrame) -> Optional[SignalType]:
        """
        判断是否应该平仓
        
        Args:
            data: OHLCV数据
            
        Returns:
            Optional[SignalType]: 平仓信号类型或None
        """
        if self.position == 0:
            return None
        
        current_price = data['close'].iloc[-1]
        
        # 止损止盈检查
        if self.entry_price > 0:
            if self.position > 0:  # 多仓
                # 止损
                if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                    return SignalType.SELL
                
                # 止盈
                if current_price >= self.entry_price * (1 + self.take_profit_pct):
                    return SignalType.SELL
                
            else:  # 空仓
                # 止损
                if current_price >= self.entry_price * (1 + self.stop_loss_pct):
                    return SignalType.BUY
                
                # 止盈
                if current_price <= self.entry_price * (1 - self.take_profit_pct):
                    return SignalType.BUY
        
        # 基于新预测的平仓检查
        if len(self.feature_data) > 0:
            current_features = self.feature_data.iloc[-1]
            new_signal = self.generate_trading_signal(current_features)
            
            # 如果预测方向与当前仓位相反，则平仓
            if new_signal:
                if self.position > 0 and new_signal == SignalType.SELL:
                    return SignalType.SELL
                elif self.position < 0 and new_signal == SignalType.BUY:
                    return SignalType.BUY
        
        return None
    
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        处理新数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[TradeSignal]: 生成的交易信号列表
        """
        if len(data) < self.lookback_period:
            return []
        
        # 计算技术特征
        features = self.calculate_technical_features(data)
        
        # 更新特征数据缓存
        self.feature_data = features.copy()
        
        # 获取当前价格
        current_price = data['close'].iloc[-1]
        
        signals = []
        
        # 检查是否需要重新训练模型
        if self.should_retrain():
            print("[ML] 开始重新训练模型...")
            
            # 创建目标变量
            target = self.create_target_variable(data)
            
            # 准备训练数据
            X, y = self.prepare_training_data(features, target)
            
            if len(X) >= self.min_training_samples:
                # 使用最近的数据进行训练
                train_size = min(len(X), self.lookback_period)
                X_train = X[-train_size:]
                y_train = y[-train_size:]
                
                success = self.train_model(X_train, y_train)
                if success:
                    print(f"[ML] 模型重新训练成功，使用{len(X_train)}个样本")
                else:
                    print("[ML] 模型重新训练失败")
        
        # 检查平仓信号
        exit_signal = self.should_exit_position(data)
        if exit_signal:
            if exit_signal == SignalType.BUY:
                signal_type = SignalType.CLOSE_SHORT if self.position < 0 else SignalType.BUY
            else:
                signal_type = SignalType.CLOSE_LONG if self.position > 0 else SignalType.SELL
            
            # 创建平仓信号
            signal = self.create_signal(
                signal_type=signal_type,
                quantity_percent=abs(self.position) / current_price if self.entry_price > 0 else self.position_size,
                confidence=0.8,
                metadata={
                    'action': 'exit',
                    'exit_reason': 'stop_loss_or_model_prediction',
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'last_prediction': self.last_prediction,
                    'prediction_confidence': self.prediction_confidence,
                    'pnl_percent': ((current_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0
                }
            )
            signals.append(signal)
            
            print(f"[ML] 平仓信号: {signal_type.value} @ {current_price:.2f}")
        
        # 检查开仓信号
        elif self.position == 0 and len(features) > 0:
            current_features = features.iloc[-1]
            enter_signal = self.generate_trading_signal(current_features)
            
            if enter_signal and self.prediction_confidence >= self.confidence_threshold:
                # 创建开仓信号
                signal = self.create_signal(
                    signal_type=enter_signal,
                    quantity_percent=self.position_size,
                    confidence=self.prediction_confidence,
                    metadata={
                        'action': 'enter',
                        'prediction': self.last_prediction,
                        'prediction_confidence': self.prediction_confidence,
                        'model_type': self.model_type,
                        'use_price_prediction': self.use_price_prediction,
                        'prediction_accuracy': self.prediction_accuracy,
                        'features_used': len([f for f in self.feature_list if f in current_features.index])
                    }
                )
                signals.append(signal)
                
                print(f"[ML] 开仓信号: {enter_signal.value} @ {current_price:.2f}, 置信度: {self.prediction_confidence:.3f}, 预测: {self.last_prediction}")
        
        return signals
    
    def on_order_fill(self, order_event) -> Optional[List[TradeSignal]]:
        """
        订单成交事件处理
        
        Args:
            order_event: 订单成交事件
            
        Returns:
            Optional[List[TradeSignal]]: 可选的额外交易信号
        """
        super().on_order_fill(order_event)
        
        # 更新入场价格
        if order_event.side == "BUY":
            if self.position > 0:  # 新开多仓
                self.entry_price = order_event.price
                print(f"[ML] 多仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
        
        elif order_event.side == "SELL":
            if self.position < 0:  # 新开空仓
                self.entry_price = order_event.price
                print(f"[ML] 空仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
            
            elif self.position == 0:  # 平仓
                print(f"[ML] 平仓完成，PnL: {self.realized_pnl:.2f}")
                self.entry_price = 0
        
        return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略状态信息
        """
        info = {
            'strategy_type': 'MachineLearning',
            'position': self.position,
            'entry_price': self.entry_price,
            'last_prediction': self.last_prediction,
            'prediction_confidence': self.prediction_confidence,
            'prediction_accuracy': self.prediction_accuracy,
            'model_trained': self.model is not None,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'parameters': {
                'model_type': self.model_type,
                'use_price_prediction': self.use_price_prediction,
                'lookback_period': self.lookback_period,
                'retrain_interval': self.retrain_interval,
                'confidence_threshold': self.confidence_threshold,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'sklearn_available': SKLEARN_AVAILABLE
            }
        }
        
        # 添加特征信息
        if len(self.feature_data) > 0:
            current_features = self.feature_data.iloc[-1]
            available_features = [f for f in self.feature_list if f in current_features.index]
            info['features'] = {
                'total_features': len(self.feature_list),
                'available_features': len(available_features),
                'feature_list': available_features[:10]  # 只显示前10个特征
            }
        
        return info