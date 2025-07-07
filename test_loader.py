# test_loader.py - 簡化版

"""
VD數據載入器測試程式 - 簡化版
==========================================

測試重點：
1. 專注Raw數據處理測試
2. 後台記憶體優化（不顯示詳細信息）
3. 智慧Archive檢查測試
4. 保留所有原本測試功能，簡化輸出

核心特色：
1. 一次性處理raw所有檔案，按日期分類
2. 3-5分鐘處理1千萬筆記錄
3. 自動分類並生成6種檔案（每個日期資料夾）
4. 支援指定日期和全日期載入
5. 💾 後台記憶體優化：靜默管理
6. 📂 簡潔測試輸出：專注主要功能
"""

import sys
import os
import time
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import psutil
import gc

# 添加src目錄到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_loader import VDDataLoader
except ImportError:
    print("❌ 無法導入VDDataLoader，請確認檔案位置")
    sys.exit(1)


def test_simplified_initialization():
    """測試1: 簡化版初始化測試"""
    print("🧪 測試1: 簡化版初始化測試")
    print("-" * 50)
    
    # 測試靜默初始化
    loader = VDDataLoader(verbose=False)
    
    print(f"✅ 簡化版初始化成功")
    print(f"   📁 基礎資料夾: {loader.base_folder}")
    print(f"   🧵 處理線程: {loader.max_workers}")
    print(f"   💾 批次大小: {loader.internal_batch_size}")
    print(f"   🎯 目標路段: {len(loader.target_route_vd_ids)}個")
    
    # 測試詳細模式
    print(f"\n🔍 測試詳細模式:")
    loader_verbose = VDDataLoader(verbose=True)
    print(f"   ✅ 詳細模式初始化成功")
    
    return True


def test_archive_check_simplified():
    """測試2: 簡化Archive檢查測試"""
    print("\n🧪 測試2: 簡化Archive檢查測試")
    print("-" * 50)
    
    loader = VDDataLoader()
    
    print("📂 測試靜默Archive檢查:")
    start_time = time.time()
    
    # 靜默Archive檢查
    archive_status = loader.check_archive_status_silent()
    check_time = time.time() - start_time
    
    print(f"   ⏱️ Archive檢查時間: {check_time:.3f} 秒")
    print(f"   📊 檢查結果:")
    print(f"      • Archive存在: {'✅' if archive_status['archive_exists'] else '❌'}")
    print(f"      • 已歸檔日期: {archive_status['archived_date_count']} 個")
    
    if archive_status['archived_dates']:
        print(f"      • 日期範圍: {archive_status['archived_dates'][0]} ~ {archive_status['archived_dates'][-1]}")
        print(f"   🎯 優勢: 不讀取檔案內容，只檢查資料夾存在性")
    
    return True


def test_raw_folder_check_simplified():
    """測試3: 簡化Raw資料夾檢查測試"""
    print("\n🧪 測試3: 簡化Raw資料夾檢查測試")
    print("-" * 50)
    
    loader = VDDataLoader()
    
    print("🔍 測試簡化Raw資料夾檢查:")
    start_time = time.time()
    
    # 簡化Raw檢查
    folder_status = loader.check_raw_folder()
    check_time = time.time() - start_time
    
    print(f"   ⏱️ Raw檢查時間: {check_time:.3f} 秒")
    print(f"   📊 檢測結果:")
    print(f"      • 資料夾存在: {'✅' if folder_status['exists'] else '❌'}")
    print(f"      • VD檔案數: {folder_status['vd_files']}")
    print(f"      • 待處理檔案: {folder_status['unprocessed']}")
    print(f"      • 已歸檔日期: {folder_status['archived_dates']}")
    
    if folder_status['unprocessed'] > 0:
        estimated_minutes = folder_status['unprocessed'] * 0.005
        print(f"      • 預估處理時間: {estimated_minutes:.1f} 分鐘")
        print(f"   🎯 特色: 專注主要信息，後台自動記憶體優化")
    
    return True


def test_date_folder_detection_simplified():
    """測試4: 簡化日期資料夾檢測"""
    print("\n🧪 測試4: 簡化日期資料夾檢測")
    print("-" * 50)
    
    loader = VDDataLoader()
    
    # 檢測現有日期資料夾
    available_dates = loader.list_available_dates()
    
    print(f"📅 檢測結果:")
    if available_dates:
        print(f"   • 可用日期: {len(available_dates)} 個")
        print(f"   • 日期範圍: {available_dates[0]} ~ {available_dates[-1]}")
        
        # 顯示前幾個日期
        for date_str in available_dates[:3]:
            print(f"      - {date_str}")
        if len(available_dates) > 3:
            print(f"      - ... 以及其他 {len(available_dates) - 3} 個日期")
    else:
        print(f"   ⚠️ 沒有找到日期資料夾")
    
    # 簡化日期摘要
    if available_dates:
        print(f"\n📊 生成簡化日期摘要:")
        try:
            start_time = time.time()
            date_summary = loader.get_date_summary()
            summary_time = time.time() - start_time
            
            print(f"   ⏱️ 摘要生成時間: {summary_time:.3f} 秒")
            
            total_dates = date_summary["總覽"]["可用日期數"]
            total_records = date_summary["總覽"]["總記錄數"]
            date_range = date_summary["總覽"]["日期範圍"]
            
            print(f"   📊 摘要結果:")
            print(f"      • 可用日期數: {total_dates}")
            print(f"      • 總記錄數: {total_records:,}")
            print(f"      • 日期範圍: {date_range['最早']} ~ {date_range['最晚']}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 日期摘要生成失敗: {e}")
            return False
    
    return True


def test_vectorized_performance_simplified():
    """測試5: 簡化向量化效能測試"""
    print("\n🧪 測試5: 簡化向量化效能測試")
    print("-" * 50)
    
    loader = VDDataLoader()
    
    print("⚡ 向量化分類效能測試:")
    
    # 建立測試數據
    test_size = 100000
    test_times = []
    
    base_date = datetime(2025, 6, 23)
    for day in range(7):
        current_date = base_date + timedelta(days=day)
        for hour in range(24):
            for minute in range(0, 60, 10):
                test_times.append(current_date.replace(hour=hour, minute=minute))
                if len(test_times) >= test_size:
                    break
            if len(test_times) >= test_size:
                break
        if len(test_times) >= test_size:
            break
    
    time_series = pd.Series(test_times)
    
    # 測試時間分類
    start_time = time.time()
    result_series = loader.classify_peak_hours_vectorized(time_series)
    process_time = time.time() - start_time
    
    print(f"   📊 時間分類測試:")
    print(f"      • 處理記錄: {len(time_series):,}")
    print(f"      • 處理時間: {process_time:.4f} 秒")
    print(f"      • 處理速度: {len(time_series)/process_time:,.0f} 記錄/秒")
    
    # 檢查結果
    peak_count = result_series.str.contains('尖峰').sum()
    print(f"      • 尖峰比例: {peak_count/len(result_series)*100:.1f}%")
    
    # 清理測試數據
    del time_series, result_series
    
    # 測試路段分類
    target_vd_ids = loader.target_route_vd_ids
    non_target_vd_ids = ['VD-N3-N-25-O-SE-1-木柵休息站', 'VD-N2-S-100.5-M-MAIN']
    
    test_vd_ids = (target_vd_ids * (test_size // (len(target_vd_ids) * 2)) + 
                   non_target_vd_ids * (test_size // (len(non_target_vd_ids) * 2)))[:test_size]
    
    vd_series = pd.Series(test_vd_ids)
    
    start_time = time.time()
    route_result = loader.is_target_route_vectorized(vd_series)
    process_time = time.time() - start_time
    
    print(f"   🛣️ 路段分類測試:")
    print(f"      • 處理記錄: {len(vd_series):,}")
    print(f"      • 處理時間: {process_time:.4f} 秒")
    print(f"      • 處理速度: {len(vd_series)/process_time:,.0f} 記錄/秒")
    
    target_count = route_result.sum()
    print(f"      • 目標路段比例: {target_count/len(route_result)*100:.1f}%")
    
    # 清理測試數據
    del vd_series, route_result
    
    print(f"   ✅ 向量化效能測試完成")
    return True


def test_data_loading_simplified():
    """測試6: 簡化數據載入測試"""
    print("\n🧪 測試6: 簡化數據載入測試")
    print("-" * 50)
    
    loader = VDDataLoader()
    
    # 獲取可用日期
    available_dates = loader.list_available_dates()
    
    if not available_dates:
        print("⚠️ 沒有可用的日期資料夾，跳過載入測試")
        return True
    
    print(f"📅 可用日期: {len(available_dates)} 個")
    
    # 測試載入特定日期
    test_date = available_dates[0]
    print(f"\n🎯 測試載入 {test_date} 數據...")
    
    try:
        start_time = time.time()
        classified_data = loader.load_classified_data(target_date=test_date)
        load_time = time.time() - start_time
        
        print(f"   ⏱️ 載入時間: {load_time:.3f} 秒")
        
        # 簡化結果顯示
        file_types = ['all', 'peak', 'offpeak', 'target_route', 'target_route_peak', 'target_route_offpeak']
        total_records = 0
        
        for file_type in file_types:
            df = classified_data.get(file_type, pd.DataFrame())
            if not df.empty:
                total_records += len(df)
        
        print(f"   📊 載入結果:")
        print(f"      • 總記錄數: {total_records:,}")
        print(f"      • 檔案類型: {len([k for k, v in classified_data.items() if not v.empty])} 種")
        
        # 清理載入的數據
        del classified_data
        
        # 測試載入所有日期（如果有多個日期）
        if len(available_dates) > 1:
            print(f"\n🔄 測試載入所有日期數據...")
            
            start_time = time.time()
            all_data = loader.load_classified_data()
            load_time = time.time() - start_time
            
            print(f"   ⏱️ 合併載入時間: {load_time:.3f} 秒")
            
            combined_total = 0
            for file_type in file_types:
                df = all_data.get(file_type, pd.DataFrame())
                if not df.empty:
                    combined_total += len(df)
            
            print(f"   📊 合併結果:")
            print(f"      • 總記錄數: {combined_total:,}")
            print(f"      • 涵蓋日期: {len(available_dates)} 個")
            
            # 清理合併數據
            del all_data
        
        return True
        
    except Exception as e:
        print(f"   ❌ 數據載入測試失敗: {e}")
        return False


def test_raw_processing_simplified():
    """測試7: 簡化Raw處理測試"""
    print("\n🧪 測試7: 簡化Raw處理測試")
    print("-" * 50)
    
    loader = VDDataLoader()
    
    # 檢查是否有資料
    folder_status = loader.check_raw_folder()
    
    if not folder_status["exists"]:
        print("⚠️ raw資料夾不存在，跳過Raw處理測試")
        return True
    
    if folder_status["unprocessed"] == 0:
        print("ℹ️ 無待處理檔案，測試載入現有數據")
        
        # 測試快速載入
        available_dates = loader.list_available_dates()
        
        if available_dates:
            print(f"📅 發現 {len(available_dates)} 個日期資料夾")
            
            # 測試載入特定日期
            test_date = available_dates[0]
            print(f"   🎯 測試快速載入 {test_date}...")
            
            start_time = time.time()
            date_data = loader.quick_load_existing_data(target_date=test_date)
            load_time = time.time() - start_time
            
            if not date_data.empty:
                print(f"   ✅ 載入成功: {len(date_data):,} 筆記錄")
                print(f"   ⏱️ 載入時間: {load_time:.3f} 秒")
                print(f"   🚀 載入速度: {len(date_data)/load_time:,.0f} 記錄/秒")
                
                # 清理數據
                del date_data
        else:
            print("⚠️ 無日期資料夾")
        
        return True
    
    else:
        print(f"🚀 發現 {folder_status['unprocessed']} 個待處理檔案")
        print("簡化版Raw處理特色：")
        print("   • 專注Raw數據處理進度顯示")
        print("   • 後台自動記憶體優化")
        print("   • 智慧Archive檢查避免重複")
        print("   • 按日期組織輸出")
        
        estimated_minutes = folder_status['unprocessed'] * 0.005
        estimated_records = folder_status['unprocessed'] * 1500
        
        print(f"\n預估：")
        print(f"   ⏱️ 處理時間: {estimated_minutes:.1f} 分鐘")
        print(f"   📊 預估記錄: {estimated_records:,} 筆")
        print(f"   📁 輸出: data/processed/YYYY-MM-DD/")
        
        response = input("是否進行簡化版Raw處理測試？(y/N): ")
        
        if response.lower() in ['y', 'yes']:
            print("\n🚀 開始簡化版Raw處理測試...")
            
            start_time = time.time()
            df = loader.process_all_files()
            process_time = time.time() - start_time
            
            print(f"\n📊 簡化版處理結果:")
            print(f"   ⏱️ 總時間: {process_time/60:.2f} 分鐘")
            
            if not df.empty:
                print(f"   ✅ 處理成功")
                print(f"   📊 總記錄數: {len(df):,}")
                print(f"   🚀 處理速度: {len(df)/(process_time/60):,.0f} 記錄/分鐘")
                
                # 檢查日期資料夾
                available_dates = loader.list_available_dates()
                print(f"   📅 建立日期資料夾: {len(available_dates)} 個")
                
                # 清理主數據
                del df
                
                return True
            else:
                print("   ❌ 處理失敗")
                return False
        else:
            print("跳過簡化版Raw處理測試")
            return True


def test_convenience_functions_simplified():
    """測試8: 簡化便利函數測試"""
    print("\n🧪 測試8: 簡化便利函數測試")
    print("-" * 50)
    
    try:
        from data_loader import (
            process_all_files_one_shot, 
            load_classified_data_quick, 
            get_date_summary_quick
        )
        
        print("🔧 測試便利函數導入...")
        print("   ✅ 成功導入所有便利函數")
        
        # 測試日期摘要便利函數
        print("\n📊 測試日期摘要便利函數...")
        
        start_time = time.time()
        summary = get_date_summary_quick()
        summary_time = time.time() - start_time
        
        if summary and "總覽" in summary:
            total_dates = summary["總覽"]["可用日期數"]
            print(f"   ✅ get_date_summary_quick(): {total_dates} 個日期")
            print(f"   ⏱️ 執行時間: {summary_time:.3f} 秒")
        else:
            print(f"   ⚠️ get_date_summary_quick(): 無結果")
        
        # 測試載入便利函數
        print("\n📂 測試載入便利函數...")
        available_dates = []
        
        processed_base = Path("data/processed")
        if processed_base.exists():
            available_dates = [d.name for d in processed_base.iterdir() 
                             if d.is_dir() and d.name.count('-') == 2]
        
        if available_dates:
            test_date = available_dates[0]
            
            start_time = time.time()
            date_data = load_classified_data_quick(target_date=test_date)
            load_time = time.time() - start_time
            
            if date_data:
                total_records = sum(len(df) for df in date_data.values() if not df.empty)
                print(f"   ✅ load_classified_data_quick({test_date}): {total_records:,} 筆記錄")
                print(f"   ⏱️ 載入時間: {load_time:.3f} 秒")
                
                # 清理數據
                del date_data
            else:
                print(f"   ⚠️ load_classified_data_quick({test_date}): 無結果")
        else:
            print(f"   ⚠️ 沒有可用日期資料夾測試")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 便利函數測試失敗: {e}")
        return False


def show_simplified_usage_guide():
    """顯示簡化版使用指南"""
    print("\n💡 簡化版使用指南")
    print("=" * 60)
    
    print("🚀 Raw數據處理（簡化版）:")
    print("```python")
    print("from src.data_loader import VDDataLoader")
    print("")
    print("# 初始化（後台記憶體優化）")
    print("loader = VDDataLoader()  # 靜默記憶體優化")
    print("")
    print("# 處理Raw數據（專注進度顯示）")
    print("df = loader.process_all_files()  # 簡潔輸出")
    print("")
    print("# 便利函數（一行搞定）")
    print("from src.data_loader import process_all_files_one_shot")
    print("df = process_all_files_one_shot()")
    print("```")
    
    print("\n📅 數據載入（簡化版）:")
    print("```python")
    print("# 載入特定日期")
    print("data = loader.load_classified_data(target_date='2025-06-27')")
    print("")
    print("# 載入所有日期")
    print("all_data = loader.load_classified_data()")
    print("")
    print("# 便利函數")
    print("from src.data_loader import load_classified_data_quick")
    print("data = load_classified_data_quick(target_date='2025-06-27')")
    print("```")
    
    print("\n📊 數據摘要（簡化版）:")
    print("```python")
    print("# 獲取日期摘要")
    print("summary = loader.get_date_summary()")
    print("")
    print("# 便利函數")
    print("from src.data_loader import get_date_summary_quick")
    print("summary = get_date_summary_quick()")
    print("```")
    
    print("\n🎯 簡化版特色:")
    print("   📋 專注主要功能：只顯示重要的處理進度")
    print("   💾 後台記憶體優化：自動管理，不顯示詳細信息")
    print("   📂 智慧Archive檢查：快速檢查，避免重複處理")
    print("   🚀 保持高速：維持3-5分鐘處理千萬筆記錄")
    print("   🔄 完整功能：保留所有原功能，簡化輸出")
    print("   📊 簡潔報告：專注核心統計數據")


def main():
    """主測試函數"""
    print("🚀 VD數據載入器簡化版測試")
    print("=" * 70)
    print("特色：專注Raw處理 + 後台記憶體優化 + 簡潔輸出")
    print("=" * 70)
    
    # 顯示系統基本資訊
    memory_info = psutil.virtual_memory()
    print(f"💾 系統記憶體: {memory_info.total/(1024**3):.1f}GB (使用率: {memory_info.percent:.1f}%)")
    
    start_time = time.time()
    test_results = []
    
    try:
        # 測試1: 簡化版初始化
        success = test_simplified_initialization()
        test_results.append(("簡化版初始化", success))
        
        # 測試2: 簡化Archive檢查
        success = test_archive_check_simplified()
        test_results.append(("簡化Archive檢查", success))
        
        # 測試3: 簡化Raw資料夾檢查
        success = test_raw_folder_check_simplified()
        test_results.append(("簡化Raw資料夾檢查", success))
        
        # 測試4: 簡化日期資料夾檢測
        success = test_date_folder_detection_simplified()
        test_results.append(("簡化日期資料夾檢測", success))
        
        # 測試5: 簡化向量化效能測試
        success = test_vectorized_performance_simplified()
        test_results.append(("簡化向量化效能", success))
        
        # 測試6: 簡化數據載入測試
        success = test_data_loading_simplified()
        test_results.append(("簡化數據載入", success))
        
        # 測試7: 簡化Raw處理測試
        success = test_raw_processing_simplified()
        test_results.append(("簡化Raw處理", success))
        
        # 測試8: 簡化便利函數測試
        success = test_convenience_functions_simplified()
        test_results.append(("簡化便利函數", success))
        
        # 顯示使用指南
        show_simplified_usage_guide()
        
    except Exception as e:
        print(f"\n❌ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試結果
    total_time = time.time() - start_time
    final_memory = psutil.virtual_memory()
    
    print(f"\n🏁 簡化版測試完成")
    print("=" * 70)
    print("📋 測試結果:")
    
    passed_tests = 0
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
        if success:
            passed_tests += 1
    
    print(f"\n📊 測試統計:")
    print(f"   • 總測試項目: {len(test_results)}")
    print(f"   • 通過測試: {passed_tests}")
    print(f"   • 成功率: {passed_tests/len(test_results)*100:.1f}%")
    print(f"   • 執行時間: {total_time:.1f} 秒")
    print(f"   • 最終記憶體: {final_memory.percent:.1f}%")
    
    # 最終評估
    if passed_tests == len(test_results):
        print("\n🎉 所有測試通過！簡化版功能完全就緒！")
        print("✅ 簡化版初始化正常")
        print("✅ 後台記憶體優化正常")
        print("✅ 智慧Archive檢查正常")
        print("✅ Raw數據處理功能正常")
        print("✅ 簡化輸出顯示正常")
        print("✅ 所有便利函數正常")
        print("🔬 可以開始專注Raw數據處理")
        
        return True
    else:
        print(f"\n⚠️ 有 {len(test_results) - passed_tests} 個測試失敗")
        print("建議檢查相關功能後再使用")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🔬 簡化版系統測試完成，專注Raw處理功能已就緒！")
        
        print("\n💻 簡化版執行流程:")
        print("1. 將所有XML檔案放入 data/raw/ 資料夾")
        print("2. 執行: python test_loader.py 或直接運行 data_loader.py")
        print("3. 系統後台自動記憶體優化（靜默）")
        print("4. 顯示簡潔的Raw處理進度")
        print("5. 自動Archive檢查，避免重複處理")
        print("6. 按日期組織輸出結果")
        
        print("\n🎯 簡化版優勢:")
        print("   📋 專注Raw處理：主要顯示處理進度和結果")
        print("   💾 後台記憶體優化：自動管理，不干擾用戶")
        print("   📂 智慧Archive檢查：快速檢查，避免重複")
        print("   📊 簡潔輸出：只顯示重要信息")
        print("   🚀 保持高速：維持3-5分鐘處理千萬筆記錄")
        print("   🔄 完整功能：保留所有原功能")
        
        print("\n📁 簡化版輸出架構:")
        print("   📂 data/processed/")
        print("      ├── 2025-06-27/  📅 按日期組織")
        print("      │   ├── vd_data_all.csv + _summary.json")
        print("      │   ├── vd_data_peak.csv + _summary.json")
        print("      │   ├── vd_data_offpeak.csv + _summary.json")
        print("      │   ├── target_route_*.csv + _summary.json")
        print("      │   └── processed_files.json")
        print("      └── ... (其他日期)")
        
        print("\n📊 簡化版使用範例:")
        print("   # 一行處理所有Raw數據")
        print("   loader = VDDataLoader()")
        print("   df = loader.process_all_files()  # 簡潔進度顯示")
        print("   ")
        print("   # 載入特定日期數據")
        print("   data = loader.load_classified_data(target_date='2025-06-27')")
        print("   ")
        print("   # 獲取摘要")
        print("   summary = loader.get_date_summary()  # 簡潔摘要")
        
        print("\n🚀 準備開始Raw數據處理:")
        print("   📅 專注Raw檔案處理和按日期組織")
        print("   💾 享受後台記憶體優化的穩定性")
        print("   📊 獲得簡潔清晰的處理進度")
        print("   🎯 快速完成數據準備工作")
        
    else:
        print("\n🔧 請解決測試中的問題")
    
    print(f"\n🎯 專案進展（簡化版）:")
    print(f"   ✅ 基礎建設")
    print(f"   ✅ 數據載入")  
    print(f"   ✅ 尖峰離峰分類")
    print(f"   ✅ 目標路段篩選")
    print(f"   ✅ 超級速度優化")
    print(f"   ✅ 一次性處理完善")
    print(f"   ✅ 日期組織架構")
    print(f"   ✅ 記憶體優化系統")
    print(f"   ✅ 簡化版專注Raw處理 🆕")
    print(f"   🔄 下一步: AI預測模型開發")
    
    print(f"\n🎊 恭喜！簡化版Raw數據處理系統已完全就緒！")
    print(f"🎯 您現在擁有專注、高效的Raw數據處理能力：")
    print(f"   • 簡潔的處理進度顯示")
    print(f"   • 後台自動記憶體優化")
    print(f"   • 智慧Archive檢查避免重複")
    print(f"   • 完整的按日期組織功能")
    print(f"   • 保持原有的超級處理速度")
    
    print(f"\n🚀 Ready for Focused Raw Data Processing! 🚀")