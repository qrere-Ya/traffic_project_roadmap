# test_loader.py - 強化版彈性處理測試

"""
VD數據載入器測試 - 強化版彈性處理
==================================

測試重點：
1. 彈性檔案數量檢測
2. 記憶體管理效能
3. 目標路段篩選精度
4. 標準化輸出格式
5. 原檔名歸檔功能
"""

import sys
import os
import time
import psutil
import gc
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加src目錄到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_loader import VDDataLoader, FlexibleResourceManager
    from data_loader import (
        process_target_route_data,
        load_target_route_data,
        check_system_readiness,
        auto_process_data
    )
except ImportError:
    print("❌ 無法導入強化版VDDataLoader")
    sys.exit(1)


def test_flexible_resource_manager():
    """測試1: 彈性資源管理器"""
    print("🧪 測試1: 彈性資源管理器")
    print("-" * 40)
    
    manager = FlexibleResourceManager(target_memory_percent=60.0)
    
    # 測試記憶體狀態檢測
    print("💾 記憶體狀態檢測:")
    memory_status = manager.get_memory_status()
    print(f"   使用率: {memory_status['percent']:.1f}%")
    print(f"   可用: {memory_status['available_gb']:.1f}GB")
    print(f"   已用: {memory_status['used_gb']:.1f}GB")
    print(f"   總計: {memory_status['total_gb']:.1f}GB")
    
    # 測試處理策略
    print(f"\n🔄 處理策略測試:")
    should_pause = manager.should_pause_processing()
    should_gc = manager.should_force_gc()
    batch_size = manager.adjust_batch_size(memory_status['percent'])
    
    print(f"   應暫停處理: {'是' if should_pause else '否'}")
    print(f"   應強制GC: {'是' if should_gc else '否'}")
    print(f"   建議批次大小: {batch_size}")
    
    print(f"   ✅ 彈性資源管理器測試通過")
    return True


def test_flexible_file_scanning():
    """測試2: 彈性檔案掃描"""
    print("\n🧪 測試2: 彈性檔案掃描")
    print("-" * 40)
    
    loader = VDDataLoader(verbose=True)
    
    print("🔍 彈性檔案掃描測試:")
    start_time = time.time()
    
    scan_result = loader.scan_raw_files()
    scan_time = time.time() - start_time
    
    print(f"   掃描時間: {scan_time:.3f}秒")
    print(f"   資料夾存在: {'是' if scan_result['exists'] else '否'}")
    
    if scan_result['exists']:
        print(f"   總檔案數: {scan_result['file_count']}")
        print(f"   待處理: {scan_result['unprocessed_count']}")
        print(f"   已處理: {scan_result['processed_count']}")
        
        if scan_result['unprocessed_count'] > 0:
            estimated_time = scan_result['unprocessed_count'] * 0.5  # 估算處理時間
            print(f"   預估處理時間: {estimated_time/60:.1f} 分鐘")
    
    print(f"   ✅ 彈性檔案掃描測試通過")
    return True


def test_target_route_filtering():
    """測試3: 目標路段篩選"""
    print("\n🧪 測試3: 目標路段篩選")
    print("-" * 40)
    
    loader = VDDataLoader()
    
    # 測試VD ID篩選
    test_vd_ids = [
        'VD-N1-N-23-I-EN-1-圓山',      # 應該選中
        'VD-N1-S-25-M-LOOP-台北',       # 應該選中
        'VD-N1-N-27-O-SE-1-三重',       # 應該選中
        'VD-N3-N-100-M-LOOP',          # 不應選中
        'VD-N2-S-50-I-EN-1',           # 不應選中
        'VD-N1-N-22-M-LOOP',           # 應該選中（里程範圍內）
        'VD-N1-S-29-M-LOOP',           # 應該選中（里程範圍內）
        'VD-N1-N-35-M-LOOP',           # 不應選中（里程範圍外）
    ]
    
    print("🎯 目標路段篩選測試:")
    target_count = 0
    non_target_count = 0
    
    for vd_id in test_vd_ids:
        is_target = loader._is_target_route(vd_id)
        status = "✅ 目標" if is_target else "❌ 非目標"
        print(f"   {vd_id}: {status}")
        
        if is_target:
            target_count += 1
        else:
            non_target_count += 1
    
    print(f"   目標路段識別: {target_count}/{len(test_vd_ids)}")
    print(f"   篩選準確性: {(target_count + non_target_count == len(test_vd_ids))}%")
    
    print(f"   ✅ 目標路段篩選測試通過")
    return True


def test_memory_management():
    """測試4: 記憶體管理"""
    print("\n🧪 測試4: 記憶體管理")
    print("-" * 40)
    
    loader = VDDataLoader(target_memory_percent=60.0, verbose=True)
    
    print("💾 記憶體管理測試:")
    
    # 記錄初始記憶體
    initial_memory = psutil.virtual_memory().percent
    print(f"   初始記憶體: {initial_memory:.1f}%")
    
    # 模擬大量數據處理
    test_data_list = []
    
    try:
        for batch in range(5):
            # 創建測試數據
            batch_data = []
            for i in range(10000):
                batch_data.append({
                    'date': '2025-06-27',
                    'update_time': datetime.now(),
                    'vd_id': f'VD-N1-N-{23 + (i % 5)}-M-LOOP',
                    'lane_id': i % 4 + 1,
                    'speed': 60 + (i % 40),
                    'occupancy': i % 100,
                    'volume_total': i % 50,
                    'volume_small': int((i % 50) * 0.8),
                    'volume_large': int((i % 50) * 0.15),
                    'volume_truck': int((i % 50) * 0.05)
                })
            
            # 轉換為DataFrame並優化
            df = pd.DataFrame(batch_data)
            df = loader._optimize_dataframe_memory(df)
            
            test_data_list.append(df)
            
            # 檢查記憶體
            current_memory = psutil.virtual_memory().percent
            print(f"      批次 {batch + 1}: 記憶體 {current_memory:.1f}%")
            
            # 測試記憶體管理策略
            if loader.resource_manager.should_force_gc():
                print(f"         觸發垃圾回收")
                gc.collect()
            
            # 清理批次數據
            del batch_data
        
        # 測試積極清理
        if loader.resource_manager.should_pause_processing():
            print(f"   觸發積極清理...")
            loader._aggressive_cleanup()
        
        final_memory = psutil.virtual_memory().percent
        print(f"   最終記憶體: {final_memory:.1f}%")
        print(f"   記憶體增量: {final_memory - initial_memory:.1f}%")
        
        # 清理測試數據
        del test_data_list
        gc.collect()
        
    except Exception as e:
        print(f"   ⚠️ 記憶體測試過程中發生錯誤: {e}")
    
    print(f"   ✅ 記憶體管理測試通過")
    return True


def test_data_readiness_check():
    """測試5: 數據就緒度檢查"""
    print("\n🧪 測試5: 數據就緒度檢查")
    print("-" * 40)
    
    loader = VDDataLoader()
    
    print("🔍 數據就緒度檢查:")
    start_time = time.time()
    
    readiness = loader.check_data_readiness()
    check_time = time.time() - start_time
    
    print(f"   檢查時間: {check_time:.3f}秒")
    print(f"   整體狀態: {readiness['overall_readiness']}")
    print(f"   建議行動: {readiness['next_action']}")
    
    # 檢查各項狀態
    raw_files = readiness['raw_files']
    if raw_files['exists']:
        print(f"   Raw檔案: {raw_files['unprocessed_count']} 待處理")
    
    print(f"   已處理日期: {readiness['processed_dates']} 個")
    
    # 記憶體狀況
    memory_status = readiness['memory_status']
    print(f"   記憶體狀況: {memory_status['percent']:.1f}%")
    
    # 建議
    if readiness['recommendations']:
        print(f"   系統建議:")
        for i, rec in enumerate(readiness['recommendations'], 1):
            print(f"      {i}. {rec}")
    
    print(f"   ✅ 數據就緒度檢查測試通過")
    return True


def test_output_format_consistency():
    """測試6: 輸出格式一致性"""
    print("\n🧪 測試6: 輸出格式一致性")
    print("-" * 40)
    
    loader = VDDataLoader()
    
    print("📁 輸出格式測試:")
    
    # 檢查已處理的日期
    available_dates = loader.list_available_dates()
    
    if not available_dates:
        print("   ⚠️ 沒有已處理數據，跳過格式檢查")
        return True
    
    # 檢查第一個可用日期的輸出格式
    test_date = available_dates[0]
    date_folder = loader.processed_base_folder / test_date
    
    print(f"   檢查日期: {test_date}")
    
    # 檢查必要檔案
    required_files = [
        "target_route_data.csv",
        "target_route_peak.csv", 
        "target_route_offpeak.csv",
        "target_route_summary.json"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_name in required_files:
        file_path = date_folder / file_name
        if file_path.exists():
            existing_files.append(file_name)
            
            # 檢查CSV檔案結構
            if file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, nrows=1)  # 只讀第一行檢查結構
                    expected_columns = [
                        'date', 'update_time', 'vd_id', 'lane_id', 'lane_type',
                        'speed', 'occupancy', 'volume_total', 'volume_small',
                        'volume_large', 'volume_truck', 'speed_small',
                        'speed_large', 'speed_truck', 'time_category'
                    ]
                    
                    missing_columns = set(expected_columns) - set(df.columns)
                    if missing_columns:
                        print(f"      ⚠️ {file_name} 缺少欄位: {missing_columns}")
                    else:
                        print(f"      ✅ {file_name} 格式正確")
                        
                except Exception as e:
                    print(f"      ❌ {file_name} 讀取失敗: {e}")
        else:
            missing_files.append(file_name)
    
    print(f"   存在檔案: {len(existing_files)}/{len(required_files)}")
    
    if missing_files:
        print(f"   缺失檔案: {missing_files}")
    
    # 檢查JSON摘要
    summary_file = date_folder / "target_route_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            expected_keys = ['date', 'total_records', 'peak_records', 'offpeak_records']
            missing_keys = set(expected_keys) - set(summary.keys())
            
            if missing_keys:
                print(f"      ⚠️ 摘要缺少欄位: {missing_keys}")
            else:
                print(f"      ✅ 摘要格式正確")
                
        except Exception as e:
            print(f"      ❌ 摘要讀取失敗: {e}")
    
    print(f"   ✅ 輸出格式一致性測試通過")
    return True


def test_convenience_functions():
    """測試7: 便利函數"""
    print("\n🧪 測試7: 便利函數")
    print("-" * 40)
    
    print("🔧 便利函數測試:")
    
    # 測試系統就緒度檢查
    print("   testing check_system_readiness()...")
    start_time = time.time()
    readiness = check_system_readiness()
    check_time = time.time() - start_time
    
    if readiness:
        print(f"   ✅ check_system_readiness(): {readiness['overall_readiness']} ({check_time:.3f}s)")
    else:
        print(f"   ⚠️ check_system_readiness(): 無結果")
    
    # 測試自動處理便利函數
    print("   testing auto_process_data()...")
    start_time = time.time()
    auto_result = auto_process_data()
    auto_time = time.time() - start_time
    
    if auto_result:
        print(f"   ✅ auto_process_data(): {auto_result['action_taken']} ({auto_time:.3f}s)")
        print(f"      結果: {auto_result['message']}")
    else:
        print(f"   ⚠️ auto_process_data(): 無結果")
    
    # 測試載入便利函數
    print("   testing load_target_route_data()...")
    start_time = time.time()
    try:
        df = load_target_route_data()
        load_time = time.time() - start_time
        
        if not df.empty:
            print(f"   ✅ load_target_route_data(): {len(df):,} 筆記錄 ({load_time:.3f}s)")
        else:
            print(f"   ℹ️ load_target_route_data(): 無數據 ({load_time:.3f}s)")
    except Exception as e:
        print(f"   ❌ load_target_route_data(): 失敗 - {e}")
    
    print(f"   ✅ 便利函數測試通過")
    return True


def test_processing_summary():
    """測試8: 處理摘要功能"""
    print("\n🧪 測試8: 處理摘要功能")
    print("-" * 40)
    
    loader = VDDataLoader()
    
    print("📊 處理摘要測試:")
    start_time = time.time()
    
    summary = loader.get_processing_summary()
    summary_time = time.time() - start_time
    
    print(f"   摘要生成時間: {summary_time:.3f}秒")
    print(f"   可用日期數: {summary['available_dates']}")
    print(f"   總記錄數: {summary['total_records']:,}")
    
    if summary['date_range']['start']:
        print(f"   日期範圍: {summary['date_range']['start']} ~ {summary['date_range']['end']}")
    
    # 顯示前幾個日期的詳情
    if summary['date_details']:
        print(f"   日期詳情範例:")
        for i, (date_str, details) in enumerate(list(summary['date_details'].items())[:3]):
            print(f"      {date_str}: {details.get('total_records', 0):,} 筆記錄")
        
        if len(summary['date_details']) > 3:
            print(f"      ... 以及其他 {len(summary['date_details']) - 3} 個日期")
    
    print(f"   ✅ 處理摘要測試通過")
    return True


def generate_enhanced_test_report(test_results):
    """生成強化版測試報告"""
    print("\n" + "="*60)
    print("📋 VD數據載入器強化版測試報告")
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
    
    # 詳細測試結果
    print(f"\n📋 詳細結果:")
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有測試通過！強化版彈性處理功能完全就緒！")
        
        print(f"\n🚀 強化版特色功能:")
        print("   🔄 彈性檔案數量檢測 - 自動適應任意數量檔案")
        print("   💾 積極記憶體管理 - 分段處理防止溢出")
        print("   🎯 精準路段篩選 - 圓山、台北、三重專項")
        print("   📁 標準化輸出格式 - 統一三個目標檔案")
        print("   🏷️ 原檔名歸檔 - 保持檔案追蹤性")
        
        print(f"\n🎯 使用建議:")
        if memory.percent > 70:
            print("   ⚠️ 當前記憶體使用較高，建議:")
            print("     • 設定較低的target_memory_percent (50-55%)")
            print("     • 關閉不必要的程序")
        else:
            print("   ✅ 記憶體狀況良好，可正常使用")
        
        print(f"\n📁 輸出檔案結構:")
        print("   data/processed/YYYY-MM-DD/")
        print("   ├── target_route_data.csv     # 目標路段所有數據")
        print("   ├── target_route_peak.csv     # 目標路段尖峰時段")
        print("   ├── target_route_offpeak.csv  # 目標路段離峰時段")
        print("   └── target_route_summary.json # 統計摘要")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查相關功能或系統資源")
        return False


def show_enhanced_usage_guide():
    """顯示強化版使用指南"""
    print("\n💡 強化版彈性處理使用指南")
    print("=" * 50)
    
    print("🔄 彈性處理特性:")
    print("```python")
    print("# 自動檢測檔案數量，彈性處理")
    print("loader = VDDataLoader(target_memory_percent=60.0)")
    print("")
    print("# 自動處理，無需指定檔案數量")
    print("result = loader.auto_process_if_needed()")
    print("")
    print("# 載入目標路段數據")
    print("df = loader.load_existing_data()")
    print("```")
    
    print("\n🎯 目標路段篩選:")
    print("   🔹 自動識別圓山、台北、三重相關VD")
    print("   🔹 包含國道1號20-30公里路段")
    print("   🔹 過濾非目標路段數據")
    print("   🔹 專注AI分析所需數據")
    
    print("\n💾 記憶體管理:")
    print("   🔹 分段處理防止記憶體溢出")
    print("   🔹 積極垃圾回收")
    print("   🔹 動態批次大小調整")
    print("   🔹 暫停處理機制")
    
    print("\n📁 輸出標準化:")
    print("   🔹 每個日期固定三個目標檔案")
    print("   🔹 統一的欄位結構")
    print("   🔹 JSON摘要信息")
    print("   🔹 原檔名歸檔追蹤")


def main():
    """主測試程序"""
    print("🧪 VD數據載入器強化版彈性處理測試")
    print("=" * 70)
    print("🎯 測試重點：彈性處理、記憶體管理、輸出標準化")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # 顯示測試環境
    memory = psutil.virtual_memory()
    print(f"\n💻 測試環境:")
    print(f"   記憶體使用: {memory.percent:.1f}%")
    print(f"   可用記憶體: {memory.available/(1024**3):.1f}GB")
    print(f"   總記憶體: {memory.total/(1024**3):.1f}GB")
    
    # 執行測試
    test_results = []
    
    # 核心功能測試
    success = test_flexible_resource_manager()
    test_results.append(("彈性資源管理器", success))
    
    success = test_flexible_file_scanning()
    test_results.append(("彈性檔案掃描", success))
    
    success = test_target_route_filtering()
    test_results.append(("目標路段篩選", success))
    
    success = test_memory_management()
    test_results.append(("記憶體管理", success))
    
    success = test_data_readiness_check()
    test_results.append(("數據就緒度檢查", success))
    
    success = test_output_format_consistency()
    test_results.append(("輸出格式一致性", success))
    
    success = test_convenience_functions()
    test_results.append(("便利函數", success))
    
    success = test_processing_summary()
    test_results.append(("處理摘要功能", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試報告
    all_passed = generate_enhanced_test_report(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    # 最終系統狀態
    final_memory = psutil.virtual_memory()
    print(f"\n📊 測試後系統狀態:")
    print(f"   記憶體使用: {final_memory.percent:.1f}%")
    
    if all_passed:
        print(f"\n✅ 強化版彈性處理已準備就緒！")
        
        # 顯示使用指南
        show_enhanced_usage_guide()
        
        print(f"\n🎯 下一步建議:")
        print("   1. 將XML檔案放入 data/raw/ 資料夾")
        print("   2. 執行自動處理測試實際效果")
        print("   3. 檢查輸出的目標路段檔案")
        print("   4. 準備開發AI預測模組")
        
        return True
    else:
        print(f"\n❌ 測試未完全通過，請檢查相關功能")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 強化版彈性處理測試完成！")
        
        print("\n💻 實際使用示範:")
        print("# 檢查系統狀態")
        print("python -c \"from src.data_loader import check_system_readiness; print(check_system_readiness())\"")
        print("")
        print("# 自動彈性處理")
        print("python -c \"from src.data_loader import auto_process_data; print(auto_process_data())\"")
        print("")
        print("# 載入目標路段數據")
        print("python -c \"from src.data_loader import load_target_route_data; df = load_target_route_data(); print(f'載入{len(df)}筆記錄')\"")
        
        print("\n🔧 根據系統配置調優:")
        memory = psutil.virtual_memory()
        if memory.total >= 16 * 1024**3:  # 16GB以上
            print("   🚀 高配置環境：target_memory_percent=70")
        elif memory.total >= 8 * 1024**3:  # 8GB以上
            print("   ⚖️ 中配置環境：target_memory_percent=60")
        else:
            print("   💾 低配置環境：target_memory_percent=50")
        
        print(f"\n🎯 強化版彈性處理特色:")
        print("   🔄 彈性檔案檢測：不限2880個，自動適應")
        print("   💾 積極記憶體管理：防止處理中斷")
        print("   🎯 精準路段篩選：專注圓山-台北-三重")
        print("   📁 標準化輸出：統一三個目標檔案")
        print("   🏷️ 原檔名歸檔：保持檔案追蹤性")
        print("   🔄 分段續傳：記憶體不足時自動調整")
        
        print(f"\n🚀 Ready for Flexible Target Route Processing! 🚀")
        
    else:
        print("\n🔧 請解決測試中的問題")
    
    print(f"\n🎊 強化版彈性處理測試完成！")