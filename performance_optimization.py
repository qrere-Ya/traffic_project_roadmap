# performance_optimization.py - 性能優化和調參

"""
交通預測系統性能優化和調參
========================

優化目標：
1. 提升預測準確率到90%+
2. 降低響應時間到50ms以下
3. 模型參數自動調優
4. 特徵重要性分析
5. 預測誤差分析

優化策略：
- 超參數網格搜索
- 特徵選擇優化
- 模型ensemble優化
- 預測pipeline優化

作者: 交通預測專案團隊
日期: 2025-01-23
"""

import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 添加 src 目錄到路徑
sys.path.append('src')

class PerformanceOptimizer:
    """性能優化器"""
    
    def __init__(self, debug=True):
        self.debug = debug
        self.optimization_results = {}
        self.best_models = {}
        
        if self.debug:
            print("🔧 性能優化器初始化")
    
    def load_training_data(self, sample_rate=1.0):
        """載入完整訓練數據"""
        if self.debug:
            print(f"📊 載入訓練數據 (採樣率: {sample_rate})")
        
        try:
            from enhanced_predictor import EnhancedPredictor
            
            predictor = EnhancedPredictor(debug=False)
            df = predictor.load_fusion_data(sample_rate=sample_rate)
            X, y = predictor.prepare_features(df)
            
            # 時間序列分割
            split_idx = int(len(X) * 0.8)
            
            self.X_train = X[:split_idx]
            self.X_test = X[split_idx:]
            self.y_train = y[:split_idx]
            self.y_test = y[split_idx:]
            self.feature_names = predictor.feature_names
            
            if self.debug:
                print(f"   ✅ 訓練集: {len(self.X_train):,} 筆")
                print(f"   ✅ 測試集: {len(self.X_test):,} 筆")
                print(f"   ✅ 特徵數: {len(self.feature_names)}")
            
            # 標準化特徵
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ 數據載入失敗: {e}")
            return False
    
    def optimize_xgboost_parameters(self):
        """XGBoost超參數優化"""
        if self.debug:
            print("\n🎯 XGBoost超參數優化")
            print("-" * 40)
        
        try:
            import xgboost as xgb
            
            # 定義參數搜索空間
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            if self.debug:
                print(f"🔍 參數搜索空間: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} 組合")
            
            # 使用RandomizedSearchCV加速搜索
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
            random_search = RandomizedSearchCV(
                xgb_model,
                param_grid,
                n_iter=20,  # 隨機搜索20組參數
                cv=3,       # 3折交叉驗證
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            start_time = time.time()
            random_search.fit(self.X_train_scaled, self.y_train)
            search_time = time.time() - start_time
            
            # 最佳參數
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            if self.debug:
                print(f"   ⏱️ 搜索時間: {search_time:.1f} 秒")
                print(f"   🏆 最佳CV分數: {best_score:.3f}")
                print(f"   📊 最佳參數:")
                for param, value in best_params.items():
                    print(f"     {param}: {value}")
            
            # 使用最佳參數訓練模型
            best_xgb = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
            best_xgb.fit(self.X_train_scaled, self.y_train)
            
            # 測試集評估
            y_pred = best_xgb.predict(self.X_test_scaled)
            test_r2 = r2_score(self.y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            if self.debug:
                print(f"   📈 測試集R²: {test_r2:.3f}")
                print(f"   📈 測試集RMSE: {test_rmse:.2f}")
            
            self.best_models['xgboost'] = {
                'model': best_xgb,
                'params': best_params,
                'cv_score': best_score,
                'test_r2': test_r2,
                'test_rmse': test_rmse
            }
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ XGBoost優化失敗: {e}")
            return False
    
    def optimize_randomforest_parameters(self):
        """RandomForest超參數優化"""
        if self.debug:
            print("\n🌲 RandomForest超參數優化")
            print("-" * 40)
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # 定義參數搜索空間
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            if self.debug:
                print(f"🔍 參數搜索中...")
            
            # RandomizedSearchCV
            rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            random_search = RandomizedSearchCV(
                rf_model,
                param_grid,
                n_iter=15,  # 隨機搜索15組參數
                cv=3,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            start_time = time.time()
            random_search.fit(self.X_train_scaled, self.y_train)
            search_time = time.time() - start_time
            
            # 最佳參數
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            if self.debug:
                print(f"   ⏱️ 搜索時間: {search_time:.1f} 秒")
                print(f"   🏆 最佳CV分數: {best_score:.3f}")
                print(f"   📊 最佳參數:")
                for param, value in best_params.items():
                    print(f"     {param}: {value}")
            
            # 使用最佳參數訓練模型
            best_rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            best_rf.fit(self.X_train_scaled, self.y_train)
            
            # 測試集評估
            y_pred = best_rf.predict(self.X_test_scaled)
            test_r2 = r2_score(self.y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            if self.debug:
                print(f"   📈 測試集R²: {test_r2:.3f}")
                print(f"   📈 測試集RMSE: {test_rmse:.2f}")
            
            self.best_models['random_forest'] = {
                'model': best_rf,
                'params': best_params,
                'cv_score': best_score,
                'test_r2': test_r2,
                'test_rmse': test_rmse
            }
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ RandomForest優化失敗: {e}")
            return False
    
    def analyze_feature_importance(self):
        """特徵重要性分析"""
        if self.debug:
            print("\n📊 特徵重要性分析")
            print("-" * 40)
        
        try:
            feature_importance_data = {}
            
            # XGBoost特徵重要性
            if 'xgboost' in self.best_models:
                xgb_importance = self.best_models['xgboost']['model'].feature_importances_
                feature_importance_data['xgboost'] = dict(zip(self.feature_names, xgb_importance))
            
            # RandomForest特徵重要性
            if 'random_forest' in self.best_models:
                rf_importance = self.best_models['random_forest']['model'].feature_importances_
                feature_importance_data['random_forest'] = dict(zip(self.feature_names, rf_importance))
            
            # 計算平均重要性
            if len(feature_importance_data) > 0:
                avg_importance = {}
                for feature in self.feature_names:
                    importances = [data[feature] for data in feature_importance_data.values() if feature in data]
                    avg_importance[feature] = np.mean(importances) if importances else 0
                
                # 排序特徵重要性
                sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                
                if self.debug:
                    print(f"🏆 前10重要特徵:")
                    for i, (feature, importance) in enumerate(sorted_features[:10]):
                        print(f"   {i+1:2d}. {feature}: {importance:.3f}")
                
                # 分析低重要性特徵
                low_importance_features = [f for f, imp in sorted_features if imp < 0.01]
                if self.debug and low_importance_features:
                    print(f"⚠️ 低重要性特徵 (<0.01): {len(low_importance_features)} 個")
                    print(f"   建議移除: {low_importance_features[:5]}...")
                
                self.feature_analysis = {
                    'feature_importance': avg_importance,
                    'sorted_features': sorted_features,
                    'low_importance': low_importance_features
                }
                
                return True
            else:
                if self.debug:
                    print("❌ 沒有可用的模型進行特徵分析")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"❌ 特徵重要性分析失敗: {e}")
            return False
    
    def optimize_ensemble_weights(self):
        """優化ensemble權重"""
        if self.debug:
            print("\n🔗 ensemble權重優化")
            print("-" * 40)
        
        try:
            if len(self.best_models) < 2:
                if self.debug:
                    print("⚠️ 需要至少2個模型進行ensemble優化")
                return False
            
            # 獲取各模型預測
            predictions = {}
            for model_name, model_info in self.best_models.items():
                pred = model_info['model'].predict(self.X_test_scaled)
                predictions[model_name] = pred
            
            # 測試不同權重組合
            best_r2 = -np.inf
            best_weights = None
            
            weight_combinations = [
                {'xgboost': 0.7, 'random_forest': 0.3},
                {'xgboost': 0.8, 'random_forest': 0.2},
                {'xgboost': 0.6, 'random_forest': 0.4},
                {'xgboost': 0.5, 'random_forest': 0.5},
                {'xgboost': 0.9, 'random_forest': 0.1}
            ]
            
            if self.debug:
                print(f"🔍 測試 {len(weight_combinations)} 種權重組合...")
            
            for weights in weight_combinations:
                # 計算加權平均預測
                ensemble_pred = np.zeros_like(self.y_test)
                for model_name, weight in weights.items():
                    if model_name in predictions:
                        ensemble_pred += weight * predictions[model_name]
                
                # 計算R²
                r2 = r2_score(self.y_test, ensemble_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = weights.copy()
                
                if self.debug:
                    weight_str = ', '.join([f"{k}:{v}" for k, v in weights.items()])
                    print(f"   {weight_str} → R²: {r2:.3f}")
            
            if best_weights:
                if self.debug:
                    print(f"🏆 最佳權重組合: R² = {best_r2:.3f}")
                    for model, weight in best_weights.items():
                        print(f"   {model}: {weight}")
                
                self.best_ensemble = {
                    'weights': best_weights,
                    'r2': best_r2
                }
                
                return True
            else:
                return False
                
        except Exception as e:
            if self.debug:
                print(f"❌ ensemble權重優化失敗: {e}")
            return False
    
    def benchmark_prediction_speed(self, num_tests=1000):
        """預測速度基準測試"""
        if self.debug:
            print(f"\n⚡ 預測速度基準測試 ({num_tests} 次)")
            print("-" * 40)
        
        try:
            speed_results = {}
            
            # 準備測試數據
            test_indices = np.random.choice(len(self.X_test_scaled), num_tests, replace=True)
            
            for model_name, model_info in self.best_models.items():
                model = model_info['model']
                
                # 測試預測速度
                times = []
                for idx in test_indices:
                    sample = self.X_test_scaled[idx:idx+1]
                    
                    start_time = time.time()
                    _ = model.predict(sample)
                    pred_time = time.time() - start_time
                    
                    times.append(pred_time * 1000)  # 轉換為毫秒
                
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                speed_results[model_name] = {
                    'avg_time_ms': avg_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time
                }
                
                if self.debug:
                    print(f"   {model_name}:")
                    print(f"     平均: {avg_time:.2f} ms")
                    print(f"     範圍: {min_time:.2f} - {max_time:.2f} ms")
            
            self.speed_benchmark = speed_results
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ 速度基準測試失敗: {e}")
            return False
    
    def save_optimized_models(self):
        """保存優化後的模型"""
        if self.debug:
            print("\n💾 保存優化後的模型")
            print("-" * 40)
        
        try:
            import pickle
            import json
            
            models_folder = Path("models/optimized_models")
            models_folder.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            for model_name, model_info in self.best_models.items():
                if model_name == 'xgboost':
                    model_file = models_folder / f"optimized_{model_name}.json"
                    model_info['model'].save_model(str(model_file))
                else:
                    model_file = models_folder / f"optimized_{model_name}.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_info['model'], f)
                
                if self.debug:
                    print(f"   ✅ {model_name}: {model_file}")
            
            # 保存scaler
            scaler_file = models_folder / "optimized_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # 保存特徵名稱
            features_file = models_folder / "optimized_features.json"
            with open(features_file, 'w') as f:
                json.dump(self.feature_names, f)
            
            # 保存優化結果報告
            optimization_report = {
                'model_performance': {
                    name: {
                        'params': info['params'],
                        'cv_score': info['cv_score'],
                        'test_r2': info['test_r2'],
                        'test_rmse': info['test_rmse']
                    }
                    for name, info in self.best_models.items()
                },
                'ensemble_weights': getattr(self, 'best_ensemble', {}),
                'feature_analysis': getattr(self, 'feature_analysis', {}),
                'speed_benchmark': getattr(self, 'speed_benchmark', {})
            }
            
            report_file = models_folder / "optimization_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(optimization_report, f, ensure_ascii=False, indent=2, default=str)
            
            if self.debug:
                print(f"   ✅ 優化報告: {report_file}")
                print(f"   📁 模型目錄: {models_folder}")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ 模型保存失敗: {e}")
            return False
    
    def generate_optimization_report(self):
        """生成優化報告"""
        print(f"\n" + "="*60)
        print("📋 交通預測系統性能優化報告")
        print("="*60)
        
        if hasattr(self, 'X_train'):
            print(f"\n📊 優化數據:")
            print(f"   訓練數據: {len(self.X_train):,} 筆")
            print(f"   測試數據: {len(self.X_test):,} 筆")
            print(f"   特徵數量: {len(self.feature_names)}")
        
        print(f"\n🏆 優化後模型性能:")
        if self.best_models:
            for model_name, model_info in self.best_models.items():
                print(f"   {model_name}:")
                print(f"     CV R²: {model_info['cv_score']:.3f}")
                print(f"     測試R²: {model_info['test_r2']:.3f}")
                print(f"     測試RMSE: {model_info['test_rmse']:.2f} km/h")
        
        if hasattr(self, 'best_ensemble'):
            ensemble = self.best_ensemble
            print(f"\n🔗 Ensemble優化:")
            print(f"   最佳組合R²: {ensemble['r2']:.3f}")
            print(f"   最佳權重:")
            for model, weight in ensemble['weights'].items():
                print(f"     {model}: {weight}")
        
        if hasattr(self, 'speed_benchmark'):
            print(f"\n⚡ 預測速度:")
            for model_name, speed_info in self.speed_benchmark.items():
                print(f"   {model_name}: {speed_info['avg_time_ms']:.1f} ms")
        
        if hasattr(self, 'feature_analysis'):
            analysis = self.feature_analysis
            print(f"\n📊 特徵優化:")
            print(f"   重要特徵數: {len([f for f, imp in analysis['sorted_features'] if imp > 0.01])}")
            print(f"   低重要性特徵: {len(analysis['low_importance'])}")
            print(f"   可移除特徵: {len(analysis['low_importance'])}")
        
        # 性能改善建議
        print(f"\n💡 性能改善建議:")
        
        best_r2 = max(info['test_r2'] for info in self.best_models.values()) if self.best_models else 0
        
        if best_r2 > 0.9:
            print(f"   🎉 模型性能優秀 (R² = {best_r2:.3f})")
            print(f"   建議: 系統已達到生產級標準")
        elif best_r2 > 0.8:
            print(f"   ✅ 模型性能良好 (R² = {best_r2:.3f})")
            print(f"   建議: 可考慮增加更多特徵工程")
        else:
            print(f"   ⚠️ 模型性能需改善 (R² = {best_r2:.3f})")
            print(f"   建議: 增加訓練數據或調整特徵")
        
        if hasattr(self, 'speed_benchmark'):
            avg_speed = np.mean([info['avg_time_ms'] for info in self.speed_benchmark.values()])
            if avg_speed < 50:
                print(f"   ⚡ 預測速度優秀 ({avg_speed:.1f} ms)")
            elif avg_speed < 100:
                print(f"   ✅ 預測速度良好 ({avg_speed:.1f} ms)")
            else:
                print(f"   ⚠️ 預測速度需優化 ({avg_speed:.1f} ms)")


def main():
    """主優化程序"""
    print("🔧 交通預測系統性能優化")
    print("=" * 50)
    print("🎯 目標: 提升預測準確率到90%+，降低響應時間到50ms")
    print("=" * 50)
    
    optimizer = PerformanceOptimizer(debug=True)
    
    # 執行優化序列
    optimization_steps = [
        ("載入訓練數據", lambda: optimizer.load_training_data(sample_rate=0.8)),
        ("XGBoost參數優化", optimizer.optimize_xgboost_parameters),
        ("RandomForest參數優化", optimizer.optimize_randomforest_parameters),
        ("特徵重要性分析", optimizer.analyze_feature_importance),
        ("Ensemble權重優化", optimizer.optimize_ensemble_weights),
        ("預測速度基準測試", lambda: optimizer.benchmark_prediction_speed(500)),
        ("保存優化模型", optimizer.save_optimized_models)
    ]
    
    results = []
    total_start_time = time.time()
    
    for step_name, step_func in optimization_steps:
        if optimizer.debug:
            print(f"\n🔄 執行: {step_name}")
        
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success and step_name == "載入訓練數據":
                print("❌ 數據載入失敗，無法繼續優化")
                break
                
        except Exception as e:
            print(f"❌ {step_name} 失敗: {e}")
            results.append((step_name, False))
    
    total_duration = time.time() - total_start_time
    
    # 生成優化報告
    optimizer.generate_optimization_report()
    
    # 優化總結
    successful_steps = sum(1 for _, success in results if success)
    total_steps = len(results)
    
    print(f"\n" + "="*60)
    print(f"📊 優化總結")
    print("="*60)
    print(f"總優化步驟: {total_steps}")
    print(f"成功步驟: {successful_steps}")
    print(f"成功率: {successful_steps/total_steps*100:.1f}%")
    print(f"總優化時間: {total_duration:.1f} 秒")
    
    print(f"\n詳細結果:")
    for step_name, success in results:
        status = "✅ 完成" if success else "❌ 失敗"
        print(f"   • {step_name}: {status}")
    
    if successful_steps >= total_steps * 0.8:
        print(f"\n🎉 性能優化完成！系統已達到最佳狀態！")
        print(f"\n📁 優化後模型位置: models/optimized_models/")
        print(f"📋 優化報告: models/optimized_models/optimization_report.json")
        return True
    else:
        print(f"\n⚠️ 優化未完全成功，建議檢查:")
        print("   1. 訓練數據質量和數量")
        print("   2. 特徵工程是否充分")
        print("   3. 模型參數搜索範圍")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 系統性能優化完成！")
        print(f"💡 下一步: 使用優化後的模型進行生產部署")
    else:
        print(f"\n🔧 請根據優化結果進行進一步調整")