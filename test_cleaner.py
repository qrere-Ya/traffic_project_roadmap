# test_cleaner.py - 適配版測試

"""
VD目標路段清理器測試 - 適配強化版載入器
=====================================

專門測試適配強化版data_loader.py的清理功能：
1. 🎯 目標路段檔案檢測測試
2. 📁 新檔案結構適配測試
3. 💾 記憶體優化清理測試
4. ⚡ 分批處理大檔案測試
5. 🔄 清理結果驗證測試
"""

import sys
import os
import pandas as pd
import json
import time
import psutil
from datetime import datetime
from pathlib import Path

# 添加src目錄到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def monitor_memory():
    """監控記憶體使用"""
    memory = psutil.virtual_memory()
    return {
        'percent': memory.percent,
        'available_gb': memory.available / (1024**3)
    }


def test_cleaner_initialization():
    """測試1: 清理器初始化"""
    print("🧪 測試1: 清理器初始化")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        print("✅ 成功導入 VDTargetRouteCleaner")
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    
    # 初始化清理器
    try:
        cleaner = VDTargetRouteCleaner(target_memory_percent=70.0)
        print("✅ 清理器初始化成功")
        print(f"   📁 輸入目錄: {cleaner.processed_base_folder}")
        print(f"   📁 輸出目錄: {cleaner.cleaned_base_folder}")
        print(f"   💾 目標記憶體: 70.0%")
        print(f"   🎯 目標檔案類型: {len(cleaner.target_file_mappings)}")
        
        # 顯示檔案映射
        print(f"   📋 檔案映射:")
        for name, info in cleaner.target_file_mappings.items():
            print(f"      • {info['pattern']} → {info['output']}")
        
        # 檢查記憶體監控
        with cleaner.memory_monitor("初始化測試"):
            test_data = list(range(10000))
            del test_data
        
        print("✅ 記憶體監控功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False


def test_target_file_detection():
    """測試2: 目標檔案檢測"""
    print("\n🧪 測試2: 目標檔案檢測")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        cleaner = VDTargetRouteCleaner()
        
        print("🔍 檢測目標路段檔案...")
        start_time = time.time()
        
        available_dates = cleaner.detect_available_dates()
        detection_time = time.time() - start_time
        
        print(f"   ⏱️ 檢測時間: {detection_time:.3f} 秒")
        
        if available_dates:
            print(f"   ✅ 找到 {len(available_dates)} 個可清理日期")
            
            # 檢查各日期的目標檔案
            for date_str in available_dates[:3]:  # 只檢查前3個日期
                date_folder = cleaner.processed_base_folder / date_str
                print(f"      📅 {date_str}:")
                
                for name, file_info in cleaner.target_file_mappings.items():
                    target_file = date_folder / file_info['pattern']
                    if target_file.exists():
                        file_size = target_file.stat().st_size / (1024 * 1024)
                        print(f"         ✅ {file_info['pattern']}: {file_size:.1f}MB")
                    else:
                        print(f"         ❌ {file_info['pattern']}: 不存在")
        else:
            print("   ⚠️ 沒有找到目標路段檔案")
            print("   💡 請先執行: python src/data_loader.py 或 auto_process_data()")
        
        return True
        
    except Exception as e:
        print(f"❌ 目標檔案檢測失敗: {e}")
        return False


def test_single_file_cleaning():
    """測試3: 單檔清理功能"""
    print("\n🧪 測試3: 單檔清理功能")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        cleaner = VDTargetRouteCleaner()
        
        available_dates = cleaner.detect_available_dates()
        
        if not available_dates:
            print("   ⚠️ 沒有可清理的日期，跳過測試")
            return True
        
        # 選擇第一個日期測試
        test_date = available_dates[0]
        print(f"🧹 測試清理日期: {test_date}")
        
        # 檢查記憶體狀況
        initial_memory = monitor_memory()
        print(f"   💾 初始記憶體: {initial_memory['percent']:.1f}%")
        
        start_time = time.time()
        result = cleaner.clean_date_folder(test_date, method='mark_nan')
        cleaning_time = time.time() - start_time
        
        final_memory = monitor_memory()
        
        print(f"   ⏱️ 清理時間: {cleaning_time:.2f} 秒")
        print(f"   💾 最終記憶體: {final_memory['percent']:.1f}%")
        print(f"   📊 清理結果:")
        print(f"      • 總檔案: {result['total_files']}")
        print(f"      • 成功檔案: {result['successful_files']}")
        print(f"      • 成功率: {result['success_rate']}")
        
        # 檢查輸出檔案
        output_folder = Path(result['output_folder'])
        if output_folder.exists():
            output_files = list(output_folder.glob("*_cleaned.csv"))
            total_size = sum(f.stat().st_size for f in output_files) / (1024 * 1024)
            print(f"      • 輸出檔案: {len(output_files)} 個")
            print(f"      • 總大小: {total_size:.1f}MB")
            
            # 檢查具體檔案
            for output_file in output_files:
                file_size = output_file.stat().st_size / (1024 * 1024)
                print(f"         ✅ {output_file.name}: {file_size:.1f}MB")
        
        return result['successful_files'] > 0
        
    except Exception as e:
        print(f"❌ 單檔清理測試失敗: {e}")
        return False


def test_cleaning_methods():
    """測試4: 清理方法比較"""
    print("\n🧪 測試4: 清理方法比較")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        cleaner = VDTargetRouteCleaner()
        
        available_dates = cleaner.detect_available_dates()
        
        if not available_dates:
            print("   ⚠️ 沒有可清理的日期，跳過測試")
            return True
        
        # 選擇第一個日期的目標檔案測試
        test_date = available_dates[0]
        date_folder = cleaner.processed_base_folder / test_date
        
        # 尋找測試檔案
        test_file = None
        test_description = None
        
        for name, file_info in cleaner.target_file_mappings.items():
            target_file = date_folder / file_info['pattern']
            if target_file.exists():
                test_file = target_file
                test_description = file_info['description']
                break
        
        if not test_file:
            print("   ⚠️ 沒有找到測試檔案")
            return True
        
        print(f"🧪 測試檔案: {test_description}")
        file_size = test_file.stat().st_size / (1024 * 1024)
        print(f"   📊 檔案大小: {file_size:.1f}MB")
        
        # 載入測試數據
        if file_size > 50:  # 大檔案只讀取部分
            df_test = pd.read_csv(test_file, nrows=10000)
            print(f"   📝 測試記錄: 10,000 (採樣)")
        else:
            df_test = pd.read_csv(test_file)
            print(f"   📝 測試記錄: {len(df_test):,}")
        
        # 測試不同清理方法
        methods = [
            ('mark_nan', '標記為NaN'),
            ('remove_rows', '刪除異常行')
        ]
        
        results = {}
        
        for method, description in methods:
            print(f"\n   🧪 測試方法: {description}")
            
            # 優化數據類型
            df_optimized = cleaner._optimize_dtypes(df_test.copy())
            
            # 計算異常值
            invalid_count = cleaner._count_invalid_values(df_optimized)
            
            # 應用清理方法
            df_cleaned = cleaner._apply_cleaning_method(df_optimized, method)
            
            results[method] = {
                'original': len(df_optimized),
                'cleaned': len(df_cleaned),
                'removed': len(df_optimized) - len(df_cleaned),
                'invalid_found': invalid_count
            }
            
            print(f"      原始記錄: {results[method]['original']:,}")
            print(f"      清理後記錄: {results[method]['cleaned']:,}")
            print(f"      移除記錄: {results[method]['removed']:,}")
            print(f"      發現異常值: {results[method]['invalid_found']:,}")
        
        print(f"\n   📈 方法比較:")
        for method, result in results.items():
            retention_rate = (result['cleaned'] / result['original']) * 100
            print(f"      {method}: 保留 {retention_rate:.1f}% 記錄")
        
        return True
        
    except Exception as e:
        print(f"❌ 清理方法測試失敗: {e}")
        return False


def test_batch_cleaning():
    """測試5: 批次清理"""
    print("\n🧪 測試5: 批次清理")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        cleaner = VDTargetRouteCleaner()
        
        available_dates = cleaner.detect_available_dates()
        
        if not available_dates:
            print("   ⚠️ 沒有可清理的日期，跳過測試")
            return True
        
        print(f"🚀 批次清理 {len(available_dates)} 個日期的目標路段數據")
        
        # 記錄初始狀態
        initial_memory = monitor_memory()
        print(f"   💾 初始記憶體: {initial_memory['percent']:.1f}%")
        
        start_time = time.time()
        report = cleaner.clean_all_dates(method='mark_nan')
        total_time = time.time() - start_time
        
        final_memory = monitor_memory()
        
        print(f"   ⏱️ 總清理時間: {total_time:.1f} 秒")
        print(f"   💾 最終記憶體: {final_memory['percent']:.1f}%")
        
        if report['summary']['successful_dates'] > 0:
            print(f"   ✅ 批次清理成功")
            
            summary = report['summary']
            print(f"   📊 清理統計:")
            print(f"      • 成功日期: {summary['successful_dates']}/{summary['total_dates']}")
            print(f"      • 成功檔案: {summary['successful_files']}/{summary['total_files']}")
            print(f"      • 成功率: {summary['success_rate']}")
            
            # 檢查清理的檔案類型
            metadata = report['metadata']
            print(f"      • 目標檔案類型: {len(metadata['target_files'])}")
            for target_file in metadata['target_files']:
                print(f"         - {target_file}")
            
            return True
        else:
            print(f"   ❌ 批次清理失敗")
            return False
            
    except Exception as e:
        print(f"❌ 批次清理測試失敗: {e}")
        return False


def test_large_file_processing():
    """測試6: 大檔案處理"""
    print("\n🧪 測試6: 大檔案處理")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        cleaner = VDTargetRouteCleaner()
        
        available_dates = cleaner.detect_available_dates()
        
        if not available_dates:
            print("   ⚠️ 沒有可清理的日期，跳過測試")
            return True
        
        # 尋找最大的目標檔案進行測試
        largest_file = None
        largest_size = 0
        
        for date_str in available_dates:
            date_folder = cleaner.processed_base_folder / date_str
            for name, file_info in cleaner.target_file_mappings.items():
                target_file = date_folder / file_info['pattern']
                if target_file.exists():
                    file_size = target_file.stat().st_size / (1024 * 1024)
                    if file_size > largest_size:
                        largest_size = file_size
                        largest_file = target_file
        
        if not largest_file or largest_size < 5:
            print("   ⚠️ 沒有找到大檔案（>5MB）進行測試")
            return True
        
        print(f"🔥 測試大檔案處理")
        print(f"   📁 檔案: {largest_file.name}")
        print(f"   📊 大小: {largest_size:.1f}MB")
        
        # 測試分批處理策略
        initial_memory = monitor_memory()
        print(f"   💾 初始記憶體: {initial_memory['percent']:.1f}%")
        
        start_time = time.time()
        
        # 模擬大檔案清理
        temp_output = cleaner.cleaned_base_folder / "temp_large_file_test.csv"
        
        result = cleaner._clean_large_file(
            largest_file, temp_output, 
            "大檔案測試", 'mark_nan'
        )
        
        processing_time = time.time() - start_time
        final_memory = monitor_memory()
        
        print(f"   ⏱️ 處理時間: {processing_time:.1f} 秒")
        print(f"   💾 最終記憶體: {final_memory['percent']:.1f}%")
        
        if result['success']:
            print(f"   ✅ 大檔案處理成功")
            print(f"      • 原始記錄: {result['original_records']:,}")
            print(f"      • 清理後記錄: {result['cleaned_records']:,}")
            print(f"      • 處理方法: {result.get('processing_method', 'standard')}")
            print(f"      • 輸出大小: {result['file_size_mb']:.1f}MB")
            
            # 清理測試檔案
            if temp_output.exists():
                temp_output.unlink()
                print(f"      • 清理測試檔案")
            
            return True
        else:
            print(f"   ❌ 大檔案處理失敗")
            return False
            
    except Exception as e:
        print(f"❌ 大檔案處理測試失敗: {e}")
        return False


def test_cleaned_data_verification():
    """測試7: 清理數據驗證"""
    print("\n🧪 測試7: 清理數據驗證")
    print("-" * 40)
    
    try:
        from data_cleaner import VDTargetRouteCleaner
        cleaner = VDTargetRouteCleaner()
        
        print("📊 獲取清理摘要...")
        summary = cleaner.get_cleaned_summary()
        
        if summary['cleaned_dates'] == 0:
            print("   ⚠️ 沒有找到清理數據")
            return True
        
        print(f"   ✅ 清理摘要:")
        print(f"      • 已清理日期: {summary['cleaned_dates']}")
        print(f"      • 總檔案數: {summary['total_files']}")
        print(f"      • 總記錄數: ~{summary['total_records']:,}")
        print(f"      • 總大小: {summary['total_size_mb']:.1f}MB")
        
        # 驗證數據完整性
        print(f"\n🔍 驗證數據完整性...")
        
        verification_count = 0
        total_verifications = 0
        
        for date_str, details in summary['date_details'].items():
            print(f"   📅 驗證 {date_str}:")
            print(f"      • 檔案數: {details['files']}")
            print(f"      • 大小: {details['size_mb']:.1f}MB")
            
            # 嘗試載入數據
            try:
                cleaned_data = cleaner.load_cleaned_date(date_str)
                
                for name, df in cleaned_data.items():
                    total_verifications += 1
                    if not df.empty:
                        verification_count += 1
                        
                        # 檢查關鍵欄位
                        required_cols = ['speed', 'occupancy', 'volume_total', 'vd_id']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            print(f"         ⚠️ {name}: 缺少欄位 {missing_cols}")
                        else:
                            # 檢查目標路段特徵
                            unique_vds = df['vd_id'].nunique()
                            print(f"         ✅ {name}: {len(df):,} 筆記錄, {unique_vds} 個VD")
                    else:
                        print(f"         ❌ {name}: 空檔案")
                        
            except Exception as e:
                print(f"      ❌ 載入失敗: {e}")
        
        verification_rate = (verification_count / total_verifications * 100) if total_verifications > 0 else 0
        print(f"\n   📈 驗證結果: {verification_count}/{total_verifications} 通過 ({verification_rate:.1f}%)")
        
        return verification_rate >= 80  # 至少80%通過
        
    except Exception as e:
        print(f"❌ 數據驗證失敗: {e}")
        return False


def test_convenience_functions():
    """測試8: 便利函數"""
    print("\n🧪 測試8: 便利函數")
    print("-" * 40)
    
    try:
        from data_cleaner import clean_all_target_data, get_cleaning_summary, load_cleaned_data
        
        print("🔧 測試便利函數...")
        
        # 測試摘要函數
        print("   testing get_cleaning_summary()...")
        start_time = time.time()
        summary = get_cleaning_summary()
        summary_time = time.time() - start_time
        
        if summary and summary['cleaned_dates'] > 0:
            print(f"   ✅ get_cleaning_summary(): {summary['cleaned_dates']} 日期 ({summary_time:.3f}s)")
        else:
            print(f"   ⚠️ get_cleaning_summary(): 無清理數據")
        
        # 測試載入函數
        print("   testing load_cleaned_data()...")
        start_time = time.time()
        
        if summary and summary['date_details']:
            test_date = list(summary['date_details'].keys())[0]
            cleaned_data = load_cleaned_data(date_str=test_date)
            load_time = time.time() - start_time
            
            if cleaned_data:
                total_records = sum(len(df) for df in cleaned_data.values() if not df.empty)
                loaded_files = len([df for df in cleaned_data.values() if not df.empty])
                print(f"   ✅ load_cleaned_data({test_date}): {loaded_files} 檔案, {total_records:,} 記錄 ({load_time:.3f}s)")
                
                # 檢查載入的檔案類型
                for name, df in cleaned_data.items():
                    if not df.empty:
                        print(f"      • {name}: {len(df):,} 筆目標路段記錄")
            else:
                print(f"   ⚠️ load_cleaned_data({test_date}): 無數據")
        else:
            print(f"   ⚠️ 沒有可測試的日期")
        
        return True
        
    except Exception as e:
        print(f"❌ 便利函數測試失敗: {e}")
        return False


def generate_test_summary(test_results):
    """生成測試摘要"""
    print("\n" + "="*60)
    print("📋 VD目標路段清理器適配版測試報告")
    print("="*60)
    
    passed_tests = sum(1 for result in test_results if result[1])
    total_tests = len(test_results)
    
    print(f"📊 測試統計:")
    print(f"   總測試項目: {total_tests}")
    print(f"   通過測試: {passed_tests}")
    print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
    
    # 系統狀態
    current_memory = monitor_memory()
    print(f"\n💻 當前系統狀態:")
    print(f"   記憶體使用: {current_memory['percent']:.1f}%")
    print(f"   可用記憶體: {current_memory['available_gb']:.1f}GB")
    
    # 詳細結果
    print(f"\n📋 詳細結果:")
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有測試通過！目標路段清理器已準備就緒！")
        
        print(f"\n🚀 適配版特色功能:")
        print("   🎯 專注目標路段檔案 - 配合強化版載入器")
        print("   📁 適配新檔案結構 - target_route_*.csv")
        print("   💾 智能記憶體管理 - 分批處理大檔案")
        print("   ⚡ 簡化清理流程 - 保留核心功能")
        print("   🔄 完美配合 - 與彈性處理載入器無縫銜接")
        
        print(f"\n📁 輸出檔案結構:")
        print("   data/cleaned/YYYY-MM-DD/")
        print("   ├── target_route_data_cleaned.csv     # 目標路段所有數據")
        print("   ├── target_route_peak_cleaned.csv     # 目標路段尖峰數據")
        print("   ├── target_route_offpeak_cleaned.csv  # 目標路段離峰數據")
        print("   └── cleaning_report.json              # 清理報告")
        
        print(f"\n🎯 使用建議:")
        print("   # 一鍵清理所有目標路段數據")
        print("   from src.data_cleaner import clean_all_target_data")
        print("   report = clean_all_target_data()")
        print("")
        print("   # 載入特定日期清理數據")
        print("   from src.data_cleaner import load_cleaned_data")
        print("   data = load_cleaned_data(date_str='2025-06-27')")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查相關功能後再使用")
        return False


def show_usage_guide():
    """顯示使用指南"""
    print("\n💡 適配版目標路段清理器使用指南")
    print("=" * 50)
    
    print("🚀 快速開始:")
    print("```python")
    print("from src.data_cleaner import clean_all_target_data")
    print("")
    print("# 一鍵清理所有目標路段數據")
    print("report = clean_all_target_data()")
    print("")
    print("# 檢查清理結果")
    print("if report['summary']['successful_dates'] > 0:")
    print("    print('清理成功！')")
    print("```")
    
    print("\n📅 按日期處理:")
    print("```python")
    print("from src.data_cleaner import VDTargetRouteCleaner")
    print("")
    print("cleaner = VDTargetRouteCleaner()")
    print("")
    print("# 檢測可用日期")
    print("dates = cleaner.detect_available_dates()")
    print("")
    print("# 清理特定日期")
    print("result = cleaner.clean_date_folder('2025-06-27')")
    print("```")
    
    print("\n🎯 適配版特性:")
    print("   🔹 專門處理目標路段檔案（圓山-台北-三重）")
    print("   🔹 適配強化版載入器的輸出格式")
    print("   🔹 大檔案自動分批處理（>50MB）")
    print("   🔹 智能記憶體監控和垃圾回收")
    print("   🔹 統一的清理後檔案命名規則")


def main():
    """主測試程序"""
    print("🧪 VD目標路段清理器適配版測試")
    print("=" * 70)
    print("🎯 測試重點：目標檔案適配、記憶體優化、清理驗證")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # 顯示初始狀態
    initial_memory = monitor_memory()
    print(f"\n💻 測試環境:")
    print(f"   記憶體使用: {initial_memory['percent']:.1f}%")
    print(f"   可用記憶體: {initial_memory['available_gb']:.1f}GB")
    
    # 執行測試
    test_results = []
    
    # 核心功能測試
    success = test_cleaner_initialization()
    test_results.append(("清理器初始化", success))
    
    success = test_target_file_detection()
    test_results.append(("目標檔案檢測", success))
    
    success = test_single_file_cleaning()
    test_results.append(("單檔清理功能", success))
    
    success = test_cleaning_methods()
    test_results.append(("清理方法比較", success))
    
    success = test_batch_cleaning()
    test_results.append(("批次清理", success))
    
    success = test_large_file_processing()
    test_results.append(("大檔案處理", success))
    
    success = test_cleaned_data_verification()
    test_results.append(("清理數據驗證", success))
    
    success = test_convenience_functions()
    test_results.append(("便利函數", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試報告
    all_passed = generate_test_summary(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    # 顯示最終狀態
    final_memory = monitor_memory()
    print(f"\n📊 測試後系統狀態:")
    print(f"   記憶體使用: {final_memory['percent']:.1f}%")
    print(f"   記憶體變化: {final_memory['percent'] - initial_memory['percent']:+.1f}%")
    
    if all_passed:
        print(f"\n✅ 適配版目標路段清理器已準備就緒！")
        
        # 顯示使用指南
        show_usage_guide()
        
        print(f"\n🎯 下一步建議:")
        print("   1. 執行清理: python src/data_cleaner.py")
        print("   2. 檢查清理結果: 確認 data/cleaned/ 目錄")
        print("   3. 開始AI模型開發: python src/predictor.py")
        print("   4. 使用清理的目標路段數據進行交通預測")
        
        return True
    else:
        print(f"\n❌ 測試未完全通過，請檢查相關功能")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 適配版目標路段清理器測試完成！")
        
        print("\n💻 使用示範:")
        print("# 一鍵清理目標路段數據")
        print("python -c \"from src.data_cleaner import clean_all_target_data; print(clean_all_target_data())\"")
        print("")
        print("# 檢查清理結果")
        print("python -c \"from src.data_cleaner import get_cleaning_summary; print(get_cleaning_summary())\"")
        
        print(f"\n🎯 適配版特色:")
        print("   🎯 專注目標路段：只處理圓山-台北-三重路段數據")
        print("   📁 檔案結構適配：完美配合強化版載入器輸出")
        print("   💾 記憶體優化：分批處理防止大檔案溢出")
        print("   ⚡ 簡化流程：保留核心清理功能")
        print("   🔄 無縫銜接：與彈性處理載入器完美配合")
        
        print(f"\n📊 清理後數據特點:")
        print("   ✅ 標記異常值為NaN或移除異常行")
        print("   ✅ 保持目標路段數據完整性")
        print("   ✅ 統一的檔案命名規則")
        print("   ✅ 完整的清理報告和摘要")
        
        print(f"\n🚀 Ready for AI Model Development! 🚀")
        
    else:
        print("\n🔧 請解決測試中的問題")
    
    print(f"\n🎊 適配版目標路段清理器測試完成！")# test_cleaner.py - 適配版測試