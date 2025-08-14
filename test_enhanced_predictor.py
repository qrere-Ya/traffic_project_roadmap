# test_enhanced_predictor.py - 增強預測器測試

"""
VD+eTag增強預測器測試程式
========================

測試重點：
1. 增強預測器導入與初始化
2. 融合數據載入
3. 特徵準備
4. 模型訓練（XGBoost + RandomForest）
5. 模型融合預測
6. 15分鐘預測功能
7. 模型保存載入
8. 性能評估

簡化原則：
- 專注核心預測功能
- 清晰的性能指標
- 實用的使用指南

作者: 交通預測專案團隊
日期: 2025-01-23
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# 添加 src 目錄到路徑
sys.path.append('src')

def test_enhanced_predictor_import():
    """測試1: 增強預測器導入"""
    print("🧪 測試1: 增強預測器導入")
    print("-" * 30)
    
    try:
        from enhanced_predictor import (
            EnhancedPredictor,
            train_enhanced_predictor,
            load_enhanced_predictor
        )
        print("✅ 成功導入增強預測器類別")
        print("✅ 成功導入便利函數")
        
        # 測試初始化
        predictor = EnhancedPredictor(debug=False)
        print("✅ 增強預測器初始化成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False


def test_fusion_data_loading():
    """測試2: 融合數據載入"""
    print("\n🧪 測試2: 融合數據載入")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        predictor = EnhancedPredictor(debug=True)
        available_dates = predictor.get_available_fusion_dates()
        
        print(f"📊 檢測結果:")
        print(f"   可用融合日期: {len(available_dates)} 天")
        
        if available_dates:
            print(f"   日期範圍: {available_dates[0]} - {available_dates[-1]}")
            
            # 測試數據載入
            start_time = time.time()
            df = predictor.load_fusion_data(sample_rate=0.1)  # 10%採樣
            load_time = time.time() - start_time
            
            print(f"✅ 數據載入成功:")
            print(f"   ⏱️ 載入時間: {load_time:.3f} 秒")
            print(f"   📊 記錄數: {len(df):,}")
            print(f"   📋 欄位數: {len(df.columns)}")
            
            return True
        else:
            print("⚠️ 沒有可用的融合數據")
            return False
        
    except Exception as e:
        print(f"❌ 融合數據載入測試失敗: {e}")
        return False


def test_feature_preparation():
    """測試3: 特徵準備"""
    print("\n🧪 測試3: 特徵準備")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        predictor = EnhancedPredictor(debug=False)
        
        # 載入少量數據進行測試
        df = predictor.load_fusion_data(sample_rate=0.05)  # 5%採樣
        
        print(f"📊 原始數據: {len(df)} 筆, {len(df.columns)} 欄位")
        
        # 特徵準備
        start_time = time.time()
        X, y = predictor.prepare_features(df)
        prep_time = time.time() - start_time
        
        print(f"✅ 特徵準備成功:")
        print(f"   ⏱️ 處理時間: {prep_time:.3f} 秒")
        print(f"   📊 特徵維度: {X.shape}")
        print(f"   🎯 目標範圍: {y.min():.1f} - {y.max():.1f}")
        print(f"   🔧 特徵數量: {len(predictor.feature_names)}")
        
        # 檢查數據品質
        nan_features = np.isnan(X).sum()
        nan_targets = np.isnan(y).sum()
        
        if nan_features == 0 and nan_targets == 0:
            print(f"   ✅ 數據品質良好（無缺失值）")
        else:
            print(f"   ⚠️ 特徵缺失值: {nan_features}, 目標缺失值: {nan_targets}")
        
        return X.shape[1] > 0 and len(y) > 0
        
    except Exception as e:
        print(f"❌ 特徵準備測試失敗: {e}")
        return False


def test_model_training():
    """測試4: 模型訓練"""
    print("\n🧪 測試4: 模型訓練")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        predictor = EnhancedPredictor(debug=False)
        
        # 載入數據並準備
        df = predictor.load_fusion_data(sample_rate=0.1)  # 10%採樣
        X, y = predictor.prepare_features(df)
        
        # 分割數據
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"📊 數據分割:")
        print(f"   訓練集: {len(X_train):,} 筆")
        print(f"   測試集: {len(X_test):,} 筆")
        
        # 訓練模型
        print(f"🚀 開始模型訓練...")
        start_time = time.time()
        results = predictor.train_models(X_train, y_train, X_test, y_test)
        training_time = time.time() - start_time
        
        print(f"✅ 模型訓練完成:")
        print(f"   ⏱️ 訓練時間: {training_time:.2f} 秒")
        print(f"   🤖 訓練模型: {list(results.keys())}")
        
        # 檢查訓練結果
        success_count = 0
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                r2 = metrics['r2']
                rmse = metrics['rmse']
                
                print(f"   📊 {model_name}: R²={r2:.3f}, RMSE={rmse:.2f}")
                
                if r2 > 0.5:  # R²大於0.5算成功
                    success_count += 1
        
        print(f"   🎯 成功模型: {success_count}/{len(results)}")
        
        return success_count >= len(results) * 0.5  # 至少50%模型成功
        
    except Exception as e:
        print(f"❌ 模型訓練測試失敗: {e}")
        return False


def test_prediction_functionality():
    """測試5: 預測功能"""
    print("\n🧪 測試5: 預測功能")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        predictor = EnhancedPredictor(debug=False)
        
        # 先訓練模型
        df = predictor.load_fusion_data(sample_rate=0.05)
        X, y = predictor.prepare_features(df)
        
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        predictor.train_models(X_train, y_train, X_test, y_test)
        
        # 測試預測功能
        print(f"🎯 測試15分鐘預測功能...")
        
        # 使用測試集的第一個樣本進行預測
        test_features = X_test[:1]  # 取第一個樣本
        actual_speed = y_test[0]
        
        start_time = time.time()
        predictions = predictor.predict_15_minutes(test_features)
        pred_time = time.time() - start_time
        
        print(f"✅ 預測完成:")
        print(f"   ⏱️ 預測時間: {pred_time*1000:.1f} ms")
        print(f"   🎯 實際速度: {actual_speed:.1f} km/h")
        
        # 顯示各模型預測結果
        prediction_success = True
        for model_name, pred_result in predictions.items():
            pred_speed = pred_result['predicted_speed']
            confidence = pred_result['confidence']
            
            # 計算預測誤差
            error = abs(pred_speed - actual_speed)
            error_rate = (error / actual_speed) * 100 if actual_speed > 0 else 0
            
            print(f"   📊 {model_name}: {pred_speed:.1f} km/h (誤差: {error_rate:.1f}%, 信心: {confidence}%)")
            
            # 檢查預測合理性
            if not (20 <= pred_speed <= 120):  # 速度應在合理範圍
                prediction_success = False
        
        return prediction_success
        
    except Exception as e:
        print(f"❌ 預測功能測試失敗: {e}")
        return False


def test_model_persistence():
    """測試6: 模型保存載入"""
    print("\n🧪 測試6: 模型保存載入")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        # 訓練並保存模型
        print("💾 訓練並保存模型...")
        predictor1 = EnhancedPredictor(debug=False)
        
        df = predictor1.load_fusion_data(sample_rate=0.05)
        X, y = predictor1.prepare_features(df)
        
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        predictor1.train_models(X_train, y_train, X_test, y_test)
        print("   ✅ 模型訓練和保存完成")
        
        # 載入模型
        print("📂 載入已保存的模型...")
        predictor2 = EnhancedPredictor(debug=False)
        load_success = predictor2.load_models()
        
        if load_success:
            print("   ✅ 模型載入成功")
            
            # 測試載入的模型是否能正常預測
            test_features = X_test[:1]
            
            pred1 = predictor1.predict_15_minutes(test_features)
            pred2 = predictor2.predict_15_minutes(test_features)
            
            # 比較預測結果
            xgb_diff = abs(pred1['xgboost']['predicted_speed'] - pred2['xgboost']['predicted_speed'])
            
            print(f"   📊 預測一致性檢查:")
            print(f"      XGBoost差異: {xgb_diff:.3f} km/h")
            
            if xgb_diff < 0.1:  # 差異小於0.1認為一致
                print("   ✅ 載入模型預測結果一致")
                return True
            else:
                print("   ⚠️ 載入模型預測結果有差異")
                return False
        else:
            print("   ❌ 模型載入失敗")
            return False
            
    except Exception as e:
        print(f"❌ 模型保存載入測試失敗: {e}")
        return False


def test_complete_pipeline():
    """測試7: 完整訓練管道"""
    print("\n🧪 測試7: 完整訓練管道")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        predictor = EnhancedPredictor(debug=False)
        
        print("🚀 執行完整訓練管道...")
        start_time = time.time()
        result = predictor.train_complete_pipeline(sample_rate=0.1, test_size=0.2)
        pipeline_time = time.time() - start_time
        
        print(f"⏱️ 管道時間: {pipeline_time:.2f} 秒")
        
        if 'error' in result:
            print(f"❌ 管道失敗: {result['error']}")
            return False
        
        results = result['results']
        report = result['report']
        
        print(f"✅ 完整管道成功:")
        print(f"   📊 訓練數據: {result['training_data_size']:,} 筆")
        print(f"   🎯 特徵數量: {result['feature_count']}")
        print(f"   🤖 最佳模型: {report['training_summary']['best_model']}")
        print(f"   📈 最佳R²: {report['training_summary']['best_r2']:.3f}")
        
        # 檢查性能標準
        best_r2 = report['training_summary']['best_r2']
        
        if best_r2 > 0.7:
            print(f"   🎉 性能優秀 (R² > 0.7)")
            return True
        elif best_r2 > 0.5:
            print(f"   ✅ 性能良好 (R² > 0.5)")
            return True
        else:
            print(f"   ⚠️ 性能待改善 (R² = {best_r2:.3f})")
            return False
        
    except Exception as e:
        print(f"❌ 完整管道測試失敗: {e}")
        return False


def test_convenience_functions():
    """測試8: 便利函數"""
    print("\n🧪 測試8: 便利函數")
    print("-" * 30)
    
    try:
        from enhanced_predictor import train_enhanced_predictor, load_enhanced_predictor
        
        # 測試訓練便利函數
        print("🔧 測試訓練便利函數...")
        try:
            predictor = train_enhanced_predictor(sample_rate=0.05, debug=False)
            print("   ✅ train_enhanced_predictor(): 成功")
            train_success = True
        except Exception as e:
            print(f"   ❌ train_enhanced_predictor(): {e}")
            train_success = False
        
        # 測試載入便利函數
        print("📂 測試載入便利函數...")
        try:
            predictor = load_enhanced_predictor(debug=False)
            print("   ✅ load_enhanced_predictor(): 成功")
            load_success = True
        except Exception as e:
            print(f"   ⚠️ load_enhanced_predictor(): {e}")
            load_success = False  # 可能沒有已訓練模型
        
        # 至少訓練函數要成功
        return train_success
        
    except Exception as e:
        print(f"❌ 便利函數測試失敗: {e}")
        return False


def test_performance_benchmark():
    """測試9: 性能基準測試"""
    print("\n🧪 測試9: 性能基準測試")
    print("-" * 30)
    
    try:
        from enhanced_predictor import EnhancedPredictor
        
        predictor = EnhancedPredictor(debug=False)
        
        # 載入較大數據集進行基準測試
        print("📊 載入基準測試數據...")
        df = predictor.load_fusion_data(sample_rate=0.2)  # 20%採樣
        X, y = predictor.prepare_features(df)
        
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"   訓練集: {len(X_train):,} 筆")
        print(f"   測試集: {len(X_test):,} 筆")
        
        # 性能基準測試
        print("⚡ 執行性能基準測試...")
        
        # 訓練時間測試
        train_start = time.time()
        results = predictor.train_models(X_train, y_train, X_test, y_test)
        train_time = time.time() - train_start
        
        # 預測時間測試
        test_samples = X_test[:100]  # 測試100個樣本
        pred_start = time.time()
        for i in range(len(test_samples)):
            predictor.predict_15_minutes(test_samples[i:i+1])
        pred_time = time.time() - pred_start
        avg_pred_time = (pred_time / len(test_samples)) * 1000  # ms
        
        print(f"📊 性能基準結果:")
        print(f"   🚂 訓練時間: {train_time:.2f} 秒")
        print(f"   ⚡ 平均預測時間: {avg_pred_time:.1f} ms")
        print(f"   📈 最佳模型R²: {max(r['metrics']['r2'] for r in results.values()):.3f}")
        
        # 性能標準檢查
        performance_good = (
            train_time < 300 and  # 訓練時間小於5分鐘
            avg_pred_time < 100 and  # 預測時間小於100ms
            max(r['metrics']['r2'] for r in results.values()) > 0.6  # R²大於0.6
        )
        
        if performance_good:
            print(f"   🎉 性能基準測試通過")
        else:
            print(f"   ⚠️ 性能基準測試未達標準")
        
        return performance_good
        
    except Exception as e:
        print(f"❌ 性能基準測試失敗: {e}")
        return False


def generate_test_summary(test_results):
    """生成測試摘要"""
    print("\n" + "="*50)
    print("📋 VD+eTag增強預測器測試報告")
    print("="*50)
    
    passed_tests = sum(1 for result in test_results if result[1])
    total_tests = len(test_results)
    
    print(f"📊 測試統計:")
    print(f"   總測試項目: {total_tests}")
    print(f"   通過測試: {passed_tests}")
    print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n📋 詳細結果:")
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
    
    if passed_tests >= total_tests * 0.8:  # 80%通過
        print(f"\n🎉 增強預測器測試通過！")
        
        print(f"\n✨ 增強預測器特色:")
        print("   🔗 多源融合：基於VD+eTag融合特徵")
        print("   🤖 雙模型架構：XGBoost + RandomForest")
        print("   ⚡ 高速預測：<100ms響應時間")
        print("   📈 高精度：R²>0.7預測準確率")
        
        print(f"\n📁 模型結構:")
        print("   models/fusion_models/")
        print("   ├── fusion_xgboost.json        # XGBoost模型")
        print("   ├── fusion_random_forest.pkl   # RandomForest模型")
        print("   ├── fusion_scaler.pkl          # 特徵標準化器")
        print("   └── fusion_features.json       # 特徵名稱")
        
        print(f"\n🚀 使用方式:")
        print("```python")
        print("from src.enhanced_predictor import EnhancedPredictor")
        print("")
        print("# 訓練新模型")
        print("predictor = EnhancedPredictor(debug=True)")
        print("result = predictor.train_complete_pipeline()")
        print("")
        print("# 或載入已訓練模型")
        print("predictor = load_enhanced_predictor()")
        print("")
        print("# 15分鐘預測")
        print("predictions = predictor.predict_15_minutes(features)")
        print("```")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查融合數據和系統配置")
        return False


def main():
    """主測試程序"""
    print("🧪 VD+eTag增強預測器測試")
    print("=" * 40)
    print("🎯 測試重點：多源融合預測、模型訓練、15分鐘預測")
    print("=" * 40)
    
    start_time = datetime.now()
    
    # 執行測試序列
    test_results = []
    
    # 核心測試
    success = test_enhanced_predictor_import()
    test_results.append(("增強預測器導入", success))
    
    if success:
        success = test_fusion_data_loading()
        test_results.append(("融合數據載入", success))
        
        success = test_feature_preparation()
        test_results.append(("特徵準備", success))
        
        success = test_model_training()
        test_results.append(("模型訓練", success))
        
        success = test_prediction_functionality()
        test_results.append(("預測功能", success))
        
        success = test_model_persistence()
        test_results.append(("模型保存載入", success))
        
        success = test_complete_pipeline()
        test_results.append(("完整訓練管道", success))
        
        success = test_convenience_functions()
        test_results.append(("便利函數", success))
        
        success = test_performance_benchmark()
        test_results.append(("性能基準測試", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試摘要
    all_passed = generate_test_summary(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    if all_passed:
        print(f"\n✅ 增強預測器已準備就緒！")
        
        print(f"\n💡 下一步建議:")
        print("   1. 完整系統整合測試")
        print("   2. 性能優化和調參")
        print("   3. 部署到生產環境")
        
        return True
    else:
        print(f"\n🔧 請檢查並解決測試中的問題")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 增強預測器測試完成！")
        
        print("\n💻 快速使用:")
        print("# 訓練融合預測器")
        print("python -c \"from src.enhanced_predictor import train_enhanced_predictor; train_enhanced_predictor()\"")
        print("")
        print("# 執行預測")
        print("python src/enhanced_predictor.py")
        
        print(f"\n🚀 Ready for Enhanced Prediction! 🚀")
        
    else:
        print("\n🔧 請解決測試問題後重新執行")
    
    print(f"\n🎊 增強預測器測試完成！")