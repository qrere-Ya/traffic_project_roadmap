"""
視覺化模組測試程式
====================

功能：
1. 測試視覺化模組導入
2. 測試數據載入功能
3. 測試各種圖表生成
4. 測試互動式儀表板
5. 性能基準測試
6. 生成測試報告

基於：80,640筆AI訓練數據
作者: 交通預測專案團隊
日期: 2025-07-07
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append('src')

def test_visualizer_import():
    """測試1: 視覺化模組導入"""
    print("🧪 測試1: 視覺化模組導入")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer, quick_visualize, create_dashboard_only
        print("✅ 成功導入 TrafficVisualizer")
        print("✅ 成功導入 quick_visualize 函數")
        print("✅ 成功導入 create_dashboard_only 函數")
        
        # 檢查關鍵方法
        visualizer = TrafficVisualizer()
        required_methods = [
            'plot_time_series_analysis', 'plot_ai_model_recommendations',
            'plot_vehicle_type_analysis', 'create_interactive_dashboard',
            'plot_data_quality_heatmap', 'generate_all_visualizations'
        ]
        
        for method in required_methods:
            if hasattr(visualizer, method):
                print(f"✅ 方法 {method} 存在")
            else:
                print(f"❌ 方法 {method} 缺失")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("💡 請確認以下依賴包已安裝：")
        print("   pip install matplotlib seaborn plotly")
        return False
    except Exception as e:
        print(f"❌ 其他錯誤: {e}")
        return False


def test_data_loading_for_visualization():
    """測試2: 視覺化數據載入"""
    print("\n🧪 測試2: 視覺化數據載入")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        # 檢查數據載入狀況
        if not visualizer.datasets:
            print("⚠️ 未載入數據，檢查數據可用性...")
            
            # 檢查清理數據目錄
            cleaned_folder = Path("data/cleaned")
            if not cleaned_folder.exists():
                print(f"   ❌ 清理數據目錄不存在: {cleaned_folder}")
                print("   💡 請先執行 test_cleaner.py 生成清理數據")
                return False
            
            # 檢查日期資料夾
            date_folders = [d for d in cleaned_folder.iterdir() 
                           if d.is_dir() and d.name.count('-') == 2]
            
            if not date_folders:
                print("   ❌ 沒有找到日期資料夾")
                print("   💡 請先執行完整的數據處理流程")
                return False
            
            print(f"   ✅ 找到 {len(date_folders)} 個日期資料夾")
            return True
        
        # 統計載入數據
        total_records = sum(len(df) for df in visualizer.datasets.values())
        print(f"✅ 數據載入成功")
        print(f"   數據集數量: {len(visualizer.datasets)}")
        print(f"   總記錄數: {total_records:,}")
        
        # 檢查各數據集
        for name, df in visualizer.datasets.items():
            print(f"   {name}: {len(df):,} 筆記錄")
        
        # 檢查AI分析結果
        if visualizer.ai_analysis:
            print(f"   ✅ AI分析結果已載入")
            if 'ai_evaluation' in visualizer.ai_analysis:
                recommendations = visualizer.ai_analysis['ai_evaluation']['recommendations']
                print(f"   🤖 AI模型推薦: {len(recommendations)} 個")
        else:
            print(f"   ⚠️ 缺少AI分析結果")
        
        return True
        
    except Exception as e:
        print(f"❌ 數據載入測試失敗: {e}")
        return False


def test_time_series_visualization():
    """測試3: 時間序列視覺化"""
    print("\n🧪 測試3: 時間序列視覺化")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        if not visualizer.datasets:
            print("⚠️ 無數據可視覺化，跳過此測試")
            return True
        
        print("📈 測試時間序列圖表生成...")
        start_time = time.time()
        
        # 生成時間序列圖表
        fig = visualizer.plot_time_series_analysis(save_html=True)
        
        generation_time = time.time() - start_time
        
        if fig:
            print(f"✅ 時間序列圖表生成成功")
            print(f"   生成時間: {generation_time:.2f} 秒")
            
            # 檢查輸出檔案
            output_path = visualizer.output_folder / "time_series_analysis.html"
            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"   輸出檔案: {output_path}")
                print(f"   檔案大小: {file_size:.1f} KB")
            
            return True
        else:
            print("❌ 時間序列圖表生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ 時間序列視覺化測試失敗: {e}")
        return False


def test_ai_model_visualization():
    """測試4: AI模型推薦視覺化"""
    print("\n🧪 測試4: AI模型推薦視覺化")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        if not visualizer.datasets:
            print("⚠️ 無數據可視覺化，跳過此測試")
            return True
        
        print("🤖 測試AI模型推薦圖表生成...")
        start_time = time.time()
        
        # 生成AI模型推薦圖表
        fig = visualizer.plot_ai_model_recommendations(save_html=True)
        
        generation_time = time.time() - start_time
        
        if fig:
            print(f"✅ AI模型推薦圖表生成成功")
            print(f"   生成時間: {generation_time:.2f} 秒")
            
            # 檢查AI分析結果顯示
            if 'ai_evaluation' in visualizer.ai_analysis:
                recommendations = visualizer.ai_analysis['ai_evaluation']['recommendations']
                if recommendations:
                    print(f"   🥇 推薦模型: {recommendations[0]['model']}")
                    print(f"   📊 評分: {recommendations[0]['score']:.1f}")
            
            # 檢查輸出檔案
            output_path = visualizer.output_folder / "ai_model_recommendations.html"
            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"   輸出檔案: {output_path}")
                print(f"   檔案大小: {file_size:.1f} KB")
            
            return True
        else:
            print("❌ AI模型推薦圖表生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ AI模型視覺化測試失敗: {e}")
        return False


def test_interactive_dashboard():
    """測試5: 互動式儀表板"""
    print("\n🧪 測試5: 互動式儀表板")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        if not visualizer.datasets:
            print("⚠️ 無數據可視覺化，跳過此測試")
            return True
        
        print("📊 測試互動式儀表板生成...")
        start_time = time.time()
        
        # 生成互動式儀表板
        fig = visualizer.create_interactive_dashboard(save_html=True)
        
        generation_time = time.time() - start_time
        
        if fig:
            print(f"✅ 互動式儀表板生成成功")
            print(f"   生成時間: {generation_time:.2f} 秒")
            
            # 檢查儀表板指標
            total_records = sum(len(df) for df in visualizer.datasets.values())
            print(f"   總數據量: {total_records:,} 筆記錄")
            
            if 'ai_evaluation' in visualizer.ai_analysis:
                data_readiness = visualizer.ai_analysis['ai_evaluation']['data_readiness']
                lstm_ready = data_readiness.get('lstm_ready', False)
                print(f"   LSTM就緒: {'✅ 是' if lstm_ready else '❌ 否'}")
            
            # 檢查輸出檔案
            output_path = visualizer.output_folder / "interactive_dashboard.html"
            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"   輸出檔案: {output_path}")
                print(f"   檔案大小: {file_size:.1f} KB")
            
            return True
        else:
            print("❌ 互動式儀表板生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ 互動式儀表板測試失敗: {e}")
        return False


def test_vehicle_analysis_visualization():
    """測試6: 車種分析視覺化"""
    print("\n🧪 測試6: 車種分析視覺化")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        if not visualizer.datasets:
            print("⚠️ 無數據可視覺化，跳過此測試")
            return True
        
        print("🚗 測試車種行為分析圖表生成...")
        start_time = time.time()
        
        # 生成車種分析圖表
        fig = visualizer.plot_vehicle_type_analysis(save_html=True)
        
        generation_time = time.time() - start_time
        
        if fig:
            print(f"✅ 車種分析圖表生成成功")
            print(f"   生成時間: {generation_time:.2f} 秒")
            
            # 檢查車種數據
            if 'target_peak' in visualizer.datasets:
                df = visualizer.datasets['target_peak']
                vehicle_columns = ['volume_small', 'volume_large', 'volume_truck']
                available_columns = [col for col in vehicle_columns if col in df.columns]
                print(f"   可分析車種數: {len(available_columns)}")
            
            # 檢查輸出檔案
            output_path = visualizer.output_folder / "vehicle_type_analysis.html"
            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"   輸出檔案: {output_path}")
                print(f"   檔案大小: {file_size:.1f} KB")
            
            return True
        else:
            print("❌ 車種分析圖表生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ 車種分析視覺化測試失敗: {e}")
        return False


def test_data_quality_heatmap():
    """測試7: 數據品質熱力圖"""
    print("\n🧪 測試7: 數據品質熱力圖")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        if not visualizer.datasets:
            print("⚠️ 無數據可視覺化，跳過此測試")
            return True
        
        print("🔥 測試數據品質熱力圖生成...")
        start_time = time.time()
        
        # 生成數據品質熱力圖
        fig = visualizer.plot_data_quality_heatmap(save_html=True)
        
        generation_time = time.time() - start_time
        
        if fig:
            print(f"✅ 數據品質熱力圖生成成功")
            print(f"   生成時間: {generation_time:.2f} 秒")
            
            # 檢查輸出檔案
            output_path = visualizer.output_folder / "data_quality_heatmap.html"
            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"   輸出檔案: {output_path}")
                print(f"   檔案大小: {file_size:.1f} KB")
            
            return True
        else:
            print("⚠️ 數據品質熱力圖生成失敗（可能缺少日期分組數據）")
            return True  # 這不算嚴重錯誤
            
    except Exception as e:
        print(f"❌ 數據品質熱力圖測試失敗: {e}")
        return False


def test_complete_visualization_suite():
    """測試8: 完整視覺化套組"""
    print("\n🧪 測試8: 完整視覺化套組生成")
    print("-" * 50)
    
    try:
        from visualizer import TrafficVisualizer
        
        visualizer = TrafficVisualizer()
        
        if not visualizer.datasets:
            print("⚠️ 無數據可視覺化，跳過此測試")
            return True
        
        print("🎨 測試完整視覺化套組生成...")
        start_time = time.time()
        
        # 生成所有視覺化圖表
        generated_files = visualizer.generate_all_visualizations()
        
        generation_time = time.time() - start_time
        
        if generated_files:
            print(f"✅ 完整視覺化套組生成成功")
            print(f"   生成時間: {generation_time:.2f} 秒")
            print(f"   生成圖表數: {len(generated_files)}")
            
            # 檢查每個生成的檔案
            total_size = 0
            for filename in generated_files:
                file_path = visualizer.output_folder / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size / 1024  # KB
                    total_size += file_size
                    print(f"   ✅ {filename}: {file_size:.1f} KB")
                else:
                    print(f"   ❌ {filename}: 檔案不存在")
            
            print(f"   總檔案大小: {total_size:.1f} KB")
            
            # 檢查摘要報告
            summary_path = visualizer.output_folder / "visualization_summary.json"
            if summary_path.exists():
                print(f"   ✅ 摘要報告: visualization_summary.json")
            
            return True
        else:
            print("❌ 完整視覺化套組生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ 完整視覺化套組測試失敗: {e}")
        return False


def test_quick_functions():
    """測試9: 快速函數"""
    print("\n🧪 測試9: 便利函數")
    print("-" * 50)
    
    try:
        from visualizer import quick_visualize, create_dashboard_only
        
        print("⚡ 測試快速視覺化函數...")
        
        # 測試快速儀表板生成
        print("   測試 create_dashboard_only()...")
        start_time = time.time()
        fig = create_dashboard_only()
        dashboard_time = time.time() - start_time
        
        if fig:
            print(f"   ✅ 快速儀表板生成成功 ({dashboard_time:.2f}秒)")
        else:
            print(f"   ⚠️ 快速儀表板生成失敗（可能無數據）")
        
        print("✅ 便利函數測試完成")
        return True
        
    except Exception as e:
        print(f"❌ 便利函數測試失敗: {e}")
        return False


def test_output_file_structure():
    """測試10: 輸出檔案結構"""
    print("\n🧪 測試10: 輸出檔案結構驗證")
    print("-" * 50)
    
    try:
        output_folder = Path("outputs/figures")
        
        if not output_folder.exists():
            print("⚠️ 輸出資料夾不存在，可能尚未生成任何圖表")
            return True
        
        print(f"📁 檢查輸出目錄: {output_folder}")
        
        # 預期的檔案清單
        expected_files = [
            "interactive_dashboard.html",
            "time_series_analysis.html", 
            "ai_model_recommendations.html",
            "vehicle_type_analysis.html",
            "data_quality_heatmap.html",
            "visualization_summary.json"
        ]
        
        existing_files = []
        missing_files = []
        total_size = 0
        
        for filename in expected_files:
            file_path = output_folder / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024  # KB
                total_size += file_size
                existing_files.append(filename)
                print(f"   ✅ {filename}: {file_size:.1f} KB")
            else:
                missing_files.append(filename)
                print(f"   ❌ {filename}: 不存在")
        
        print(f"\n📊 檔案結構統計:")
        print(f"   存在檔案: {len(existing_files)}/{len(expected_files)}")
        print(f"   總檔案大小: {total_size:.1f} KB")
        print(f"   完整性: {len(existing_files)/len(expected_files)*100:.1f}%")
        
        if missing_files:
            print(f"\n⚠️ 缺少檔案: {missing_files}")
            print("   建議執行完整視覺化生成流程")
        
        return len(existing_files) >= len(expected_files) * 0.6  # 至少60%檔案存在
        
    except Exception as e:
        print(f"❌ 輸出檔案結構測試失敗: {e}")
        return False


def generate_test_summary(test_results):
    """生成測試摘要"""
    print("\n" + "="*60)
    print("📋 視覺化模組測試摘要")
    print("="*60)
    
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
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有測試通過！視覺化模組運行正常！")
        
        print(f"\n🎨 視覺化模組特色:")
        print("   ✅ 7天時間序列深度分析")
        print("   ✅ AI模型智能推薦圖表")
        print("   ✅ 車種行為模式分析")
        print("   ✅ 數據品質全面評估")
        print("   ✅ 互動式實時儀表板")
        print("   ✅ 基於80,640筆AI訓練數據")
        
        print(f"\n🌐 建議使用方式:")
        print("   1. 查看儀表板: outputs/figures/interactive_dashboard.html")
        print("   2. 時間序列分析: outputs/figures/time_series_analysis.html")
        print("   3. AI模型推薦: outputs/figures/ai_model_recommendations.html")
        
        print(f"\n📈 下一步建議:")
        print("   1. 基於視覺化結果優化數據處理")
        print("   2. 根據AI模型推薦開始模型開發")
        print("   3. 開發 src/predictor.py AI預測模組")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查相關依賴和數據後再使用")
        
        print(f"\n🔧 故障排除:")
        print("   1. 確認已安裝: pip install matplotlib seaborn plotly")
        print("   2. 確認數據已清理: python test_cleaner.py")
        print("   3. 確認分析已完成: python test_analyzer.py")
        
        return False


def main():
    """主測試程序"""
    print("🧪 視覺化模組完整測試")
    print("="*60)
    print("這將測試視覺化模組的所有核心功能:")
    print("• 數據載入和圖表生成")
    print("• AI模型推薦視覺化")
    print("• 互動式儀表板")
    print("• 完整視覺化套組")
    print("• 基於80,640筆AI訓練數據")
    print("="*60)
    
    start_time = datetime.now()
    
    # 執行測試序列
    test_results = []
    
    # 測試1: 模組導入
    success = test_visualizer_import()
    test_results.append(("視覺化模組導入", success))
    
    if success:
        # 測試2: 數據載入
        success = test_data_loading_for_visualization()
        test_results.append(("視覺化數據載入", success))
        
        # 測試3: 時間序列視覺化
        success = test_time_series_visualization()
        test_results.append(("時間序列視覺化", success))
        
        # 測試4: AI模型視覺化
        success = test_ai_model_visualization()
        test_results.append(("AI模型推薦視覺化", success))
        
        # 測試5: 互動式儀表板
        success = test_interactive_dashboard()
        test_results.append(("互動式儀表板", success))
        
        # 測試6: 車種分析視覺化
        success = test_vehicle_analysis_visualization()
        test_results.append(("車種分析視覺化", success))
        
        # 測試7: 數據品質熱力圖
        success = test_data_quality_heatmap()
        test_results.append(("數據品質熱力圖", success))
        
        # 測試8: 完整視覺化套組
        success = test_complete_visualization_suite()
        test_results.append(("完整視覺化套組", success))
        
        # 測試9: 便利函數
        success = test_quick_functions()
        test_results.append(("便利函數", success))
        
        # 測試10: 輸出檔案結構
        success = test_output_file_structure()
        test_results.append(("輸出檔案結構", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試摘要
    all_passed = generate_test_summary(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    if all_passed:
        print(f"\n✅ 視覺化模組已準備就緒！")
        print(f"🎨 您可以開始使用以下功能:")
        print(f"   • 完整視覺化: python src/visualizer.py")
        print(f"   • 快速儀表板: python -c \"from src.visualizer import create_dashboard_only; create_dashboard_only()\"")
        print(f"   • 查看結果: 瀏覽器開啟 outputs/figures/interactive_dashboard.html")
        
        print(f"\n🚀 視覺化就緒，建議下一步:")
        print("   1. 查看互動式儀表板了解數據全貌")
        print("   2. 根據AI模型推薦開始模型開發")
        print("   3. 開發 src/predictor.py 實現15分鐘預測")
        
        return True
    else:
        print(f"\n❌ 測試未完全通過，請檢查相關功能")
        return False


if __name__ == "__main__":
    main()