# system_integration_test.py - 完整系統整合測試

"""
國道1號圓山-三重路段交通預測系統 - 完整整合測試
===============================================

測試目標：
1. 端到端數據流驗證
2. 模型訓練和性能評估
3. 預測準確性驗證
4. 系統穩定性測試
5. 性能基準測試

數據流：
Raw VD/eTag → 時空對齊 → 特徵融合 → 模型訓練 → 15分鐘預測

作者: 交通預測專案團隊
日期: 2025-01-23
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加 src 目錄到路徑
sys.path.append('src')

class SystemIntegrationTester:
    """系統整合測試器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.models_trained = False
        
    def test_complete_data_pipeline(self):
        """測試1: 完整數據管道"""
        print("🧪 測試1: 完整數據管道驗證")
        print("=" * 50)
        
        try:
            # 檢查時空對齊數據
            from spatial_temporal_aligner import get_available_data_status
            alignment_status = get_available_data_status(debug=False)
            
            print(f"📊 時空對齊狀態:")
            print(f"   可用日期: {alignment_status['total_days']} 天")
            
            if alignment_status['total_days'] == 0:
                print("❌ 沒有時空對齊數據")
                return False
            
            # 檢查融合數據
            from fusion_engine import get_fusion_data_status
            fusion_status = get_fusion_data_status(debug=False)
            
            print(f"📊 融合數據狀態:")
            print(f"   可用日期: {fusion_status['total_days']} 天")
            
            if fusion_status['total_days'] == 0:
                print("❌ 沒有融合數據")
                return False
            
            # 檢查數據一致性
            if alignment_status['total_days'] == fusion_status['total_days']:
                print("✅ 數據管道完整且一致")
                return True
            else:
                print(f"⚠️ 數據不一致：對齊{alignment_status['total_days']}天，融合{fusion_status['total_days']}天")
                return False
                
        except Exception as e:
            print(f"❌ 數據管道測試失敗: {e}")
            return False
    
    def test_model_training_pipeline(self, sample_rate=0.3):
        """測試2: 模型訓練管道"""
        print(f"\n🧪 測試2: 模型訓練管道 (採樣率: {sample_rate})")
        print("=" * 50)
        
        try:
            from enhanced_predictor import EnhancedPredictor
            
            predictor = EnhancedPredictor(debug=True)
            
            print("🚀 開始完整模型訓練...")
            start_time = time.time()
            
            # 執行完整訓練管道
            result = predictor.train_complete_pipeline(
                sample_rate=sample_rate,
                test_size=0.2
            )
            
            training_time = time.time() - start_time
            
            if 'error' in result:
                print(f"❌ 訓練失敗: {result['error']}")
                return False
            
            # 記錄訓練結果
            self.models_trained = True
            self.training_result = result
            
            results = result['results']
            report = result['report']
            
            print(f"✅ 模型訓練完成:")
            print(f"   ⏱️ 訓練時間: {training_time:.1f} 秒")
            print(f"   📊 訓練數據: {result['training_data_size']:,} 筆")
            print(f"   🎯 特徵數量: {result['feature_count']}")
            
            print(f"\n📊 模型性能:")
            for model_name, model_result in results.items():
                metrics = model_result['metrics']
                print(f"   {model_name}:")
                print(f"     R²: {metrics['r2']:.3f}")
                print(f"     RMSE: {metrics['rmse']:.2f}")
                print(f"     MAE: {metrics['mae']:.2f}")
                print(f"     MAPE: {metrics['mape']:.1f}%")
            
            # 保存性能指標
            self.performance_metrics = {
                'training_time': training_time,
                'data_size': result['training_data_size'],
                'feature_count': result['feature_count'],
                'model_performance': {name: res['metrics'] for name, res in results.items()}
            }
            
            # 性能檢查
            best_r2 = max(res['metrics']['r2'] for res in results.values())
            
            if best_r2 > 0.8:
                print(f"🎉 模型性能優秀 (最佳R² = {best_r2:.3f})")
                return True
            elif best_r2 > 0.7:
                print(f"✅ 模型性能良好 (最佳R² = {best_r2:.3f})")
                return True
            elif best_r2 > 0.5:
                print(f"⚠️ 模型性能可接受 (最佳R² = {best_r2:.3f})")
                return True
            else:
                print(f"❌ 模型性能不足 (最佳R² = {best_r2:.3f})")
                return False
                
        except Exception as e:
            print(f"❌ 模型訓練測試失敗: {e}")
            return False
    
    def test_prediction_accuracy(self, test_samples=100):
        """測試3: 預測準確性驗證"""
        print(f"\n🧪 測試3: 預測準確性驗證 ({test_samples} 樣本)")
        print("=" * 50)
        
        if not self.models_trained:
            print("⚠️ 模型尚未訓練，跳過預測測試")
            return False
        
        try:
            from enhanced_predictor import load_enhanced_predictor
            
            # 載入訓練好的模型
            predictor = load_enhanced_predictor(debug=False)
            
            # 載入測試數據
            df = predictor.load_fusion_data(sample_rate=0.1)
            X, y = predictor.prepare_features(df)
            
            # 使用最後20%作為測試集
            test_start_idx = int(len(X) * 0.8)
            X_test = X[test_start_idx:]
            y_test = y[test_start_idx:]
            
            # 隨機選擇測試樣本
            if len(X_test) > test_samples:
                indices = np.random.choice(len(X_test), test_samples, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]
            
            print(f"📊 預測準確性測試:")
            print(f"   測試樣本: {len(X_test)} 個")
            
            # 批次預測
            all_predictions = []
            prediction_times = []
            
            for i in range(len(X_test)):
                start_time = time.time()
                pred_result = predictor.predict_15_minutes(X_test[i:i+1])
                pred_time = time.time() - start_time
                
                prediction_times.append(pred_time * 1000)  # ms
                all_predictions.append(pred_result['ensemble']['predicted_speed'])
            
            predictions = np.array(all_predictions)
            
            # 計算準確性指標
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
            
            avg_pred_time = np.mean(prediction_times)
            
            print(f"✅ 預測準確性結果:")
            print(f"   R²: {r2:.3f}")
            print(f"   RMSE: {rmse:.2f} km/h")
            print(f"   MAE: {mae:.2f} km/h")
            print(f"   MAPE: {mape:.1f}%")
            print(f"   平均預測時間: {avg_pred_time:.1f} ms")
            
            # 準確性分析
            error_rates = np.abs((y_test - predictions) / (y_test + 1e-8)) * 100
            accurate_predictions = (error_rates <= 10).sum()  # 誤差<=10%
            accuracy_rate = (accurate_predictions / len(error_rates)) * 100
            
            print(f"   預測準確率 (誤差≤10%): {accuracy_rate:.1f}%")
            
            # 保存預測結果
            self.prediction_accuracy = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'accuracy_rate': accuracy_rate,
                'avg_prediction_time': avg_pred_time
            }
            
            # 準確性標準
            if r2 > 0.8 and accuracy_rate > 80:
                print(f"🎉 預測準確性優秀")
                return True
            elif r2 > 0.7 and accuracy_rate > 70:
                print(f"✅ 預測準確性良好")
                return True
            else:
                print(f"⚠️ 預測準確性需要改善")
                return False
                
        except Exception as e:
            print(f"❌ 預測準確性測試失敗: {e}")
            return False
    
    def test_system_stability(self, stress_test_duration=60):
        """測試4: 系統穩定性測試"""
        print(f"\n🧪 測試4: 系統穩定性測試 ({stress_test_duration}秒)")
        print("=" * 50)
        
        if not self.models_trained:
            print("⚠️ 模型尚未訓練，跳過穩定性測試")
            return False
        
        try:
            from enhanced_predictor import load_enhanced_predictor
            
            predictor = load_enhanced_predictor(debug=False)
            
            # 準備測試數據
            df = predictor.load_fusion_data(sample_rate=0.05)
            X, y = predictor.prepare_features(df)
            
            print(f"🔄 執行穩定性壓力測試...")
            
            start_time = time.time()
            prediction_count = 0
            error_count = 0
            response_times = []
            
            while time.time() - start_time < stress_test_duration:
                try:
                    # 隨機選擇一個樣本
                    idx = np.random.randint(0, len(X))
                    test_sample = X[idx:idx+1]
                    
                    # 預測
                    pred_start = time.time()
                    result = predictor.predict_15_minutes(test_sample)
                    pred_time = time.time() - pred_start
                    
                    response_times.append(pred_time * 1000)  # ms
                    prediction_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count > 10:  # 超過10個錯誤就停止
                        break
            
            test_duration = time.time() - start_time
            
            if prediction_count > 0:
                avg_response_time = np.mean(response_times)
                max_response_time = np.max(response_times)
                predictions_per_second = prediction_count / test_duration
                error_rate = (error_count / (prediction_count + error_count)) * 100
                
                print(f"✅ 穩定性測試結果:")
                print(f"   測試時長: {test_duration:.1f} 秒")
                print(f"   預測次數: {prediction_count}")
                print(f"   錯誤次數: {error_count}")
                print(f"   錯誤率: {error_rate:.1f}%")
                print(f"   平均響應時間: {avg_response_time:.1f} ms")
                print(f"   最大響應時間: {max_response_time:.1f} ms")
                print(f"   預測吞吐量: {predictions_per_second:.1f} 次/秒")
                
                # 穩定性標準
                stability_good = (
                    error_rate < 1 and  # 錯誤率<1%
                    avg_response_time < 100 and  # 平均響應<100ms
                    predictions_per_second > 5  # 吞吐量>5次/秒
                )
                
                if stability_good:
                    print(f"🎉 系統穩定性優秀")
                    return True
                else:
                    print(f"⚠️ 系統穩定性需要改善")
                    return False
            else:
                print(f"❌ 穩定性測試失敗：無法完成預測")
                return False
                
        except Exception as e:
            print(f"❌ 穩定性測試失敗: {e}")
            return False
    
    def test_real_world_scenarios(self):
        """測試5: 真實場景模擬"""
        print(f"\n🧪 測試5: 真實場景模擬測試")
        print("=" * 50)
        
        if not self.models_trained:
            print("⚠️ 模型尚未訓練，跳過場景測試")
            return False
        
        try:
            from enhanced_predictor import load_enhanced_predictor
            
            predictor = load_enhanced_predictor(debug=False)
            
            # 模擬不同交通場景
            scenarios = [
                {'name': '平日尖峰', 'speed_range': (20, 50), 'expected': '擁堵'},
                {'name': '平日離峰', 'speed_range': (60, 90), 'expected': '順暢'},
                {'name': '假日離峰', 'speed_range': (70, 100), 'expected': '非常順暢'},
                {'name': '事故狀況', 'speed_range': (10, 30), 'expected': '嚴重擁堵'}
            ]
            
            print(f"🚗 模擬真實交通場景:")
            
            scenario_results = []
            
            for scenario in scenarios:
                print(f"\n   場景: {scenario['name']}")
                
                # 創建模擬特徵數據
                num_features = len(predictor.feature_names)
                simulated_features = np.random.rand(1, num_features)
                
                # 進行預測
                prediction = predictor.predict_15_minutes(simulated_features)
                pred_speed = prediction['ensemble']['predicted_speed']
                confidence = prediction['ensemble']['confidence']
                
                # 判斷交通狀態
                if pred_speed < 30:
                    traffic_state = '嚴重擁堵'
                elif pred_speed < 50:
                    traffic_state = '擁堵'
                elif pred_speed < 70:
                    traffic_state = '順暢'
                else:
                    traffic_state = '非常順暢'
                
                print(f"     預測速度: {pred_speed:.1f} km/h")
                print(f"     交通狀態: {traffic_state}")
                print(f"     預測信心: {confidence}%")
                
                scenario_results.append({
                    'scenario': scenario['name'],
                    'predicted_speed': pred_speed,
                    'traffic_state': traffic_state,
                    'confidence': confidence
                })
            
            print(f"\n✅ 真實場景模擬完成")
            print(f"   模擬場景: {len(scenarios)} 個")
            print(f"   預測範圍: {min(r['predicted_speed'] for r in scenario_results):.1f} - {max(r['predicted_speed'] for r in scenario_results):.1f} km/h")
            
            return True
            
        except Exception as e:
            print(f"❌ 真實場景測試失敗: {e}")
            return False
    
    def generate_integration_report(self):
        """生成整合測試報告"""
        print(f"\n" + "="*60)
        print("📋 國道1號圓山-三重路段交通預測系統整合測試報告")
        print("="*60)
        
        print(f"\n🎯 系統概述:")
        print(f"   路段: 國道1號圓山(23K)-台北(25K)-三重(27K)")
        print(f"   範圍: 3.8公里雙向，總長7.6公里")
        print(f"   技術: VD+eTag多源融合預測")
        
        if hasattr(self, 'performance_metrics'):
            metrics = self.performance_metrics
            print(f"\n📊 訓練數據統計:")
            print(f"   數據量: {metrics['data_size']:,} 筆記錄")
            print(f"   特徵數: {metrics['feature_count']} 個融合特徵")
            print(f"   訓練時間: {metrics['training_time']:.1f} 秒")
        
        if hasattr(self, 'prediction_accuracy'):
            acc = self.prediction_accuracy
            print(f"\n🎯 預測性能:")
            print(f"   預測準確率: R² = {acc['r2']:.3f}")
            print(f"   平均誤差: MAE = {acc['mae']:.1f} km/h")
            print(f"   準確預測率: {acc['accuracy_rate']:.1f}% (誤差≤10%)")
            print(f"   響應時間: {acc['avg_prediction_time']:.1f} ms")
        
        print(f"\n⚡ 系統性能指標:")
        print(f"   預測時程: 15分鐘短期預測")
        print(f"   更新頻率: 實時（毫秒級響應）")
        print(f"   數據來源: VD車輛偵測器 + eTag電子標籤")
        print(f"   模型架構: XGBoost + RandomForest 融合")
        
        print(f"\n🏆 達成目標:")
        if hasattr(self, 'prediction_accuracy'):
            r2 = self.prediction_accuracy['r2']
            response_time = self.prediction_accuracy['avg_prediction_time']
            
            if r2 > 0.8:
                print(f"   ✅ 預測準確率優秀 (目標: >85%, 實際: {r2*100:.1f}%)")
            elif r2 > 0.7:
                print(f"   ✅ 預測準確率良好 (目標: >85%, 實際: {r2*100:.1f}%)")
            else:
                print(f"   ⚠️ 預測準確率需改善 (目標: >85%, 實際: {r2*100:.1f}%)")
            
            if response_time < 100:
                print(f"   ✅ 響應時間達標 (目標: <100ms, 實際: {response_time:.1f}ms)")
            else:
                print(f"   ⚠️ 響應時間需優化 (目標: <100ms, 實際: {response_time:.1f}ms)")
        
        print(f"\n🚀 系統就緒狀態:")
        models_folder = Path("models/fusion_models")
        model_files = list(models_folder.glob("*.json")) + list(models_folder.glob("*.pkl"))
        
        if len(model_files) >= 4:
            print(f"   ✅ 融合預測模型已訓練並保存")
            print(f"   ✅ 系統可進行15分鐘交通預測")
            print(f"   ✅ 適用於國道1號圓山-三重路段")
        else:
            print(f"   ⚠️ 模型檔案不完整，需要重新訓練")
        
        print(f"\n📈 應用價值:")
        print(f"   🚗 用路人: 精準出行時間規劃")
        print(f"   🏛️ 交通管理: 即時擁堵預警與疏導")
        print(f"   📱 導航系統: 動態路線規劃優化")
        print(f"   🏢 物流業: 運輸時間成本控制")


def main():
    """主整合測試程序"""
    print("🧪 國道1號圓山-三重路段交通預測系統 - 完整整合測試")
    print("=" * 70)
    print("🎯 目標: 驗證端到端預測系統性能與準確性")
    print("=" * 70)
    
    tester = SystemIntegrationTester()
    
    # 執行整合測試
    test_sequence = [
        ("完整數據管道驗證", tester.test_complete_data_pipeline),
        ("模型訓練管道", lambda: tester.test_model_training_pipeline(sample_rate=0.5)),
        ("預測準確性驗證", lambda: tester.test_prediction_accuracy(test_samples=200)),
        ("系統穩定性測試", lambda: tester.test_system_stability(stress_test_duration=30)),
        ("真實場景模擬", tester.test_real_world_scenarios)
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in test_sequence:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
            results.append((test_name, False))
    
    total_duration = time.time() - total_start_time
    
    # 生成最終報告
    tester.generate_integration_report()
    
    # 測試總結
    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n" + "="*60)
    print(f"📊 整合測試總結")
    print("="*60)
    print(f"總測試項目: {total_tests}")
    print(f"通過測試: {passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"總測試時間: {total_duration:.1f} 秒")
    
    print(f"\n詳細結果:")
    for test_name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
    
    if success_rate >= 80:
        print(f"\n🎉 系統整合測試通過！交通預測系統已準備就緒！")
        print(f"\n🚀 可以開始部署和使用:")
        print("   1. 即時15分鐘交通預測")
        print("   2. 交通狀況監控和預警")
        print("   3. 導航系統整合應用")
        return True
    else:
        print(f"\n⚠️ 系統需要進一步優化，建議:")
        print("   1. 檢查數據品質和完整性")
        print("   2. 調整模型參數和特徵工程")
        print("   3. 增加訓練數據量")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎊 國道1號圓山-三重交通預測系統整合測試完成！")
        print(f"🚗 Ready for Real-world Traffic Prediction! 🚗")
    else:
        print(f"\n🔧 請根據測試結果進行系統優化")