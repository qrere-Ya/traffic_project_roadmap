"""
簡化版交通流量分析器測試程式
============================

專注核心測試功能：
1. 🧪 分析器導入和初始化
2. 📊 數據載入和特性分析
3. 🤖 AI模型評估測試
4. 📋 報告生成測試
5. ⚡ 性能測試

作者: 交通預測專案團隊
日期: 2025-07-21 (簡化核心版)
"""

import sys
import os
import time
import psutil
from datetime import datetime
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append('src')

def test_analyzer_import():
    """測試1: 分析器導入"""
    print("🧪 測試1: 分析器導入")
    print("-" * 50)
    
    try:
        from flow_analyzer import SimplifiedTrafficAnalyzer, quick_analyze
        print("✅ 成功導入 SimplifiedTrafficAnalyzer")
        print("✅ 成功導入 quick_analyze 函數")
        
        # 檢查關鍵方法
        analyzer = SimplifiedTrafficAnalyzer()
        required_methods = [
            'load_data', 'analyze_data_characteristics',
            'evaluate_ai_model_suitability', 'generate_comprehensive_report'
        ]
        
        missing_methods = []
        for method in required_methods:
            if hasattr(analyzer, method):
                print(f"✅ 方法 {method} 存在")
            else:
                print(f"❌ 方法 {method} 缺失")
                missing_methods.append(method)
        
        # 檢查預測配置
        config = analyzer.prediction_config
        print(f"✅ 預測配置:")
        print(f"   目標欄位: {config['target_columns']}")
        print(f"   LSTM最小記錄: {config['min_records_for_lstm']:,}")
        
        return len(missing_methods) == 0
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他錯誤: {e}")
        return False


def test_data_loading():
    """測試2: 數據載入功能"""
    print("\n🧪 測試2: 數據載入功能")
    print("-" * 50)
    
    try:
        from flow_analyzer import SimplifiedTrafficAnalyzer
        
        analyzer = SimplifiedTrafficAnalyzer()
        
        # 檢查是否有可用數據
        if not analyzer.available_dates:
            print("⚠️ 未發現清理後數據，跳過數據載入測試")
            return True
        
        print(f"🔍 發現 {len(analyzer.available_dates)} 個清理後日期")
        
        # 測試數據載入
        start_time = time.time()
        success = analyzer.load_data(sample_rate=0.5)  # 使用50%採樣加速測試
        load_time = time.time() - start_time
        
        if success:
            total_records = sum(len(df) for df in analyzer.datasets.values())
            print(f"✅ 數據載入成功")
            print(f"   載入時間: {load_time:.2f} 秒")
            print(f"   總記錄數: {total_records:,}")
            print(f"   數據集數: {len(analyzer.datasets)}")
            
            # 檢查數據結構
            for name, df in analyzer.datasets.items():
                print(f"   {name}: {len(df):,} 筆記錄")
            
            return True
        else:
            print("❌ 數據載入失敗")
            return False
            
    except Exception as e:
        print(f"❌ 數據載入測試失敗: {e}")
        return False


def test_data_analysis():
    """測試3: 數據特性分析"""
    print("\n🧪 測試3: 數據特性分析")
    print("-" * 50)
    
    try:
        from flow_analyzer import SimplifiedTrafficAnalyzer
        
        analyzer = SimplifiedTrafficAnalyzer()
        
        if not analyzer.load_data(sample_rate=0.3):  # 使用30%採樣
            print("⚠️ 無數據可分析，跳過此測試")
            return True
        
        # 執行數據特性分析
        start_time = time.time()
        characteristics = analyzer.analyze_data_characteristics()
        analysis_time = time.time() - start_time
        
        print(f"✅ 數據特性分析完成")
        print(f"   分析時間: {analysis_time:.2f} 秒")
        
        # 檢查分析結果結構
        required_keys = ['data_summary', 'quality_metrics', 'time_coverage', 'prediction_readiness']
        for key in required_keys:
            if key in characteristics:
                print(f"   ✅ {key} 分析完成")
            else:
                print(f"   ❌ {key} 分析缺失")
                return False
        
        # 顯示關鍵指標
        time_coverage = characteristics.get('time_coverage', {})
        quality_metrics = characteristics.get('quality_metrics', {})
        prediction_readiness = characteristics.get('prediction_readiness', {})
        
        print(f"📊 關鍵指標:")
        print(f"   總記錄數: {time_coverage.get('total_records', 0):,}")
        print(f"   時間跨度: {time_coverage.get('date_span_days', 0)} 天")
        print(f"   整體品質: {quality_metrics.get('overall_quality', 0):.1f}/100")
        print(f"   VD站點數: {prediction_readiness.get('unique_vd_stations', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 數據分析測試失敗: {e}")
        return False


def test_ai_model_evaluation():
    """測試4: AI模型評估功能"""
    print("\n🧪 測試4: AI模型評估功能")
    print("-" * 50)
    
    try:
        from flow_analyzer import SimplifiedTrafficAnalyzer
        
        analyzer = SimplifiedTrafficAnalyzer()
        
        if not analyzer.load_data(sample_rate=0.3):
            print("⚠️ 無數據可評估，跳過此測試")
            return True
        
        # 先進行數據特性分析（AI評估的前提）
        analyzer.analyze_data_characteristics()
        
        # 執行AI模型評估
        start_time = time.time()
        ai_evaluation = analyzer.evaluate_ai_model_suitability()
        eval_time = time.time() - start_time
        
        print(f"✅ AI模型評估完成")
        print(f"   評估時間: {eval_time:.2f} 秒")
        
        # 檢查評估結果
        required_keys = ['model_suitability', 'recommendations', 'data_readiness']
        for key in required_keys:
            if key in ai_evaluation:
                print(f"   ✅ {key} 評估完成")
            else:
                print(f"   ❌ {key} 評估缺失")
                return False
        
        # 顯示模型推薦結果
        recommendations = ai_evaluation.get('recommendations', [])
        print(f"🤖 AI模型推薦結果:")
        
        if recommendations:
            for rec in recommendations:
                print(f"   {rec['priority']} {rec['model']}: {rec['score']:.1f}分")
                print(f"      預期準確率: {rec['expected_accuracy']}")
        else:
            print("   ⚠️ 未生成模型推薦")
        
        # 檢查數據就緒度
        data_readiness = ai_evaluation.get('data_readiness', {})
        lstm_ready = data_readiness.get('lstm_ready', False)
        xgboost_ready = data_readiness.get('xgboost_ready', False)
        rf_ready = data_readiness.get('rf_ready', False)
        
        print(f"📈 模型就緒度:")
        print(f"   LSTM就緒: {'✅ 是' if lstm_ready else '❌ 否'}")
        print(f"   XGBoost就緒: {'✅ 是' if xgboost_ready else '❌ 否'}")
        print(f"   隨機森林就緒: {'✅ 是' if rf_ready else '❌ 否'}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI模型評估測試失敗: {e}")
        return False


def test_report_generation():
    """測試5: 報告生成功能"""
    print("\n🧪 測試5: 報告生成功能")
    print("-" * 50)
    
    try:
        from flow_analyzer import SimplifiedTrafficAnalyzer
        
        analyzer = SimplifiedTrafficAnalyzer()
        
        if not analyzer.load_data(sample_rate=0.3):
            print("⚠️ 無數據可生成報告，跳過此測試")
            return True
        
        # 執行完整分析並生成報告
        start_time = time.time()
        report = analyzer.generate_comprehensive_report()
        report_time = time.time() - start_time
        
        print(f"✅ 綜合報告生成完成")
        print(f"   生成時間: {report_time:.2f} 秒")
        
        # 檢查報告結構
        required_sections = [
            'metadata', 'data_summary', 'quality_assessment',
            'time_coverage', 'prediction_readiness', 'ai_model_evaluation',
            'key_insights', 'actionable_recommendations'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section in report:
                print(f"   ✅ {section} 章節完成")
            else:
                print(f"   ❌ {section} 章節缺失")
                missing_sections.append(section)
        
        # 顯示報告摘要
        metadata = report.get('metadata', {})
        insights_count = metadata.get('total_insights', 0)
        
        print(f"📋 報告摘要:")
        print(f"   分析時間: {metadata.get('analysis_date', 'N/A')}")
        print(f"   洞察數量: {insights_count}")
        print(f"   版本: {metadata.get('analyzer_version', 'N/A')}")
        
        # 測試報告保存
        try:
            test_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            analyzer.save_report(report, test_filename)
            print(f"   ✅ 報告保存成功: {test_filename}")
        except Exception as save_error:
            print(f"   ❌ 報告保存失敗: {save_error}")
            return False
        
        return len(missing_sections) == 0
        
    except Exception as e:
        print(f"❌ 報告生成測試失敗: {e}")
        return False


def test_quick_analyze_function():
    """測試6: 快速分析函數"""
    print("\n🧪 測試6: 快速分析函數")
    print("-" * 50)
    
    try:
        from flow_analyzer import quick_analyze
        
        print("🚀 執行快速分析...")
        start_time = time.time()
        result = quick_analyze(sample_rate=0.2)  # 使用20%採樣加速
        total_time = time.time() - start_time
        
        if result:
            print(f"✅ 快速分析成功")
            print(f"   總執行時間: {total_time:.2f} 秒")
            
            # 檢查返回結果
            required_keys = ['metadata', 'ai_model_evaluation', 'prediction_readiness']
            
            missing_keys = []
            for key in required_keys:
                if key in result:
                    print(f"   ✅ {key} 結果完整")
                else:
                    print(f"   ❌ {key} 結果缺失")
                    missing_keys.append(key)
            
            # 顯示關鍵結果
            ai_eval = result.get('ai_model_evaluation', {})
            if ai_eval:
                recommendations = ai_eval.get('recommendations', [])
                if recommendations:
                    top_model = recommendations[0]
                    print(f"🎯 快速分析結果:")
                    print(f"   推薦模型: {top_model.get('model', 'N/A')}")
                    print(f"   推薦評分: {top_model.get('score', 0):.1f}")
                    print(f"   預期準確率: {top_model.get('expected_accuracy', 'N/A')}")
            
            return len(missing_keys) == 0
        else:
            print("❌ 快速分析返回空結果")
            return False
            
    except Exception as e:
        print(f"❌ 快速分析測試失敗: {e}")
        return False


def test_performance_benchmark():
    """測試7: 性能基準測試"""
    print("\n🧪 測試7: 性能基準測試")
    print("-" * 50)
    
    try:
        from flow_analyzer import SimplifiedTrafficAnalyzer
        
        print("⏱️ 執行性能基準測試...")
        
        # 記錄初始記憶體
        initial_memory = psutil.virtual_memory().percent
        print(f"   初始記憶體: {initial_memory:.1f}%")
        
        # 測試多次運行的一致性和速度
        times = []
        
        for i in range(3):
            print(f"   第 {i+1} 輪測試...")
            start_time = time.time()
            
            analyzer = SimplifiedTrafficAnalyzer()
            if analyzer.load_data(sample_rate=0.2):  # 使用20%採樣加速
                analyzer.analyze_data_characteristics()
                analyzer.evaluate_ai_model_suitability()
                report = analyzer.generate_comprehensive_report()
            
            run_time = time.time() - start_time
            times.append(run_time)
            print(f"      執行時間: {run_time:.2f} 秒")
            
            # 清理
            del analyzer
            if 'report' in locals():
                del report
        
        # 性能統計
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"📊 性能統計:")
            print(f"   平均執行時間: {avg_time:.2f} 秒")
            print(f"   最快執行時間: {min_time:.2f} 秒")
            print(f"   最慢執行時間: {max_time:.2f} 秒")
            print(f"   性能穩定性: {((max_time - min_time) / avg_time * 100):.1f}% 變異")
            
            # 性能評級
            if avg_time < 10:
                performance_grade = "🚀 優秀"
            elif avg_time < 20:
                performance_grade = "✅ 良好"
            elif avg_time < 40:
                performance_grade = "⚡ 可接受"
            else:
                performance_grade = "⚠️ 需優化"
            
            print(f"   性能評級: {performance_grade}")
            
            return avg_time < 60  # 60秒內完成算通過
        else:
            print("❌ 無法完成性能測試")
            return False
            
    except Exception as e:
        print(f"❌ 性能測試失敗: {e}")
        return False


def generate_test_summary(test_results):
    """生成測試摘要"""
    print("\n" + "="*60)
    print("📋 簡化版交通分析器測試摘要")
    print("="*60)
    
    passed_tests = sum(1 for result in test_results if result[1])
    total_tests = len(test_results)
    
    print(f"📊 測試統計:")
    print(f"   總測試項目: {total_tests}")
    print(f"   通過測試: {passed_tests}")
    print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
    
    # 系統狀態
    memory = psutil.virtual_memory()
    print(f"\n💻 當前系統狀態:")
    print(f"   記憶體使用: {memory.percent:.1f}%")
    print(f"   可用記憶體: {memory.available/(1024**3):.1f}GB")
    
    # 詳細結果
    print(f"\n📋 詳細結果:")
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有測試通過！簡化版分析器運行正常！")
        
        print(f"\n🔧 簡化版特色:")
        print("   ✅ 程式碼大幅簡化，保留核心功能")
        print("   ✅ AI模型智能推薦系統")
        print("   ✅ 快速分析功能")
        print("   ✅ 性能優化")
        
        print(f"\n🎯 推薦使用方式:")
        print("   1. 快速分析: quick_analyze()")
        print("   2. 詳細分析: SimplifiedTrafficAnalyzer()")
        print("   3. 模型選擇: 參考AI模型推薦結果")
        
        print(f"\n📈 下一步建議:")
        print("   1. 使用推薦的AI模型開始訓練")
        print("   2. 開發預測系統")
        print("   3. 建立監控和評估機制")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查相關功能後再使用")
        return False


def main():
    """主測試程序"""
    print("🧪 簡化版交通流量分析器測試")
    print("=" * 60)
    print("🎯 核心測試內容:")
    print("• 數據載入和分析")
    print("• AI模型評估和推薦") 
    print("• 報告生成")
    print("• 性能基準測試")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 顯示測試環境
    memory = psutil.virtual_memory()
    print(f"\n💻 測試環境:")
    print(f"   記憶體使用: {memory.percent:.1f}%")
    print(f"   可用記憶體: {memory.available/(1024**3):.1f}GB")
    print(f"   總記憶體: {memory.total/(1024**3):.1f}GB")
    
    # 執行測試序列
    test_results = []
    
    # 基礎功能測試
    success = test_analyzer_import()
    test_results.append(("分析器導入", success))
    
    if success:
        # 核心功能測試
        success = test_data_loading()
        test_results.append(("數據載入", success))
        
        success = test_data_analysis()
        test_results.append(("數據特性分析", success))
        
        success = test_ai_model_evaluation()
        test_results.append(("AI模型評估", success))
        
        success = test_report_generation()
        test_results.append(("報告生成", success))
        
        success = test_quick_analyze_function()
        test_results.append(("快速分析函數", success))
        
        success = test_performance_benchmark()
        test_results.append(("性能基準測試", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試摘要
    all_passed = generate_test_summary(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    if all_passed:
        print(f"\n✅ 簡化版分析器已準備就緒！")
        print(f"\n🚀 可以開始使用以下功能:")
        print(f"   • 快速分析: python -c \"from src.flow_analyzer import quick_analyze; quick_analyze()\"")
        print(f"   • 查看AI模型推薦")
        print(f"   • 生成分析報告")
        return True
    else:
        print(f"\n❌ 測試未完全通過，請檢查相關功能")
        return False


if __name__ == "__main__":
    main()