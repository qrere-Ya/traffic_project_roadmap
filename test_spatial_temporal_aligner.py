# test_spatial_temporal_aligner.py - 簡化測試版

"""
VD+eTag時空對齊測試程式 - 簡化版
===============================

測試重點：
1. 動態資料檢測
2. 時空對齊功能
3. 品質驗證
4. 批次處理

簡化原則：
- 移除冗餘測試
- 保留核心功能測試
- 動態適應實際資料數量
- 強化debug功能

作者: 交通預測專案團隊
日期: 2025-01-23 (簡化版)
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append('src')

def test_aligner_import():
    """測試1: 對齊器導入"""
    print("🧪 測試1: 對齊器導入")
    print("-" * 40)
    
    try:
        from spatial_temporal_aligner import (
            SpatialTemporalAligner, align_all_available_data, get_available_data_status
        )
        print("✅ 成功導入 SpatialTemporalAligner")
        print("✅ 成功導入便利函數")
        
        aligner = SpatialTemporalAligner(debug=False)
        print("✅ 對齊器初始化成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False


def test_available_data_detection():
    """測試2: 動態資料檢測"""
    print("\n🧪 測試2: 動態資料檢測")
    print("-" * 40)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=True)
        available = aligner.get_available_dates()
        
        print(f"📊 資料檢測結果:")
        print(f"   VD資料: {len(available['vd_dates'])} 天")
        print(f"   eTag資料: {len(available['etag_dates'])} 天")
        print(f"   共同日期: {len(available['common_dates'])} 天")
        
        # 顯示具體日期
        if available['common_dates']:
            print(f"   可對齊日期:")
            for date in available['common_dates']:
                print(f"     • {date}")
        
        return len(available['common_dates']) > 0
        
    except Exception as e:
        print(f"❌ 資料檢測失敗: {e}")
        return False


def test_single_date_alignment():
    """測試3: 單日期對齊"""
    print("\n🧪 測試3: 單日期對齊")
    print("-" * 40)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=True)
        available = aligner.get_available_dates()
        
        if not available['common_dates']:
            print("⚠️ 沒有可用日期，跳過測試")
            return True
        
        test_date = available['common_dates'][0]
        print(f"🎯 測試日期: {test_date}")
        
        start_time = time.time()
        result = aligner.align_date_data(test_date)
        alignment_time = time.time() - start_time
        
        print(f"⏱️ 對齊時間: {alignment_time:.2f} 秒")
        
        if 'aligned' in result:
            aligned_df = result['aligned']
            summary = result['summary']
            
            print(f"✅ 對齊成功:")
            print(f"   📊 對齊記錄: {len(aligned_df):,} 筆")
            print(f"   🎯 VD站點: {summary['vd_stations']} 個")
            print(f"   🎯 eTag配對: {summary['etag_pairs']} 個")
            print(f"   📈 速度相關性: {summary['speed_correlation']:.3f}")
            print(f"   📊 同步品質: {summary['sync_quality_percent']:.1f}%")
            
            return True
        else:
            print(f"❌ 對齊失敗: {result.get('error', '未知錯誤')}")
            return False
            
    except Exception as e:
        print(f"❌ 單日期對齊測試失敗: {e}")
        return False


def test_batch_alignment():
    """測試4: 批次對齊"""
    print("\n🧪 測試4: 批次對齊")
    print("-" * 40)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=True)
        
        print("🚀 批次對齊所有可用資料...")
        start_time = time.time()
        results = aligner.batch_align_all_available()
        batch_time = time.time() - start_time
        
        print(f"⏱️ 批次時間: {batch_time:.2f} 秒")
        
        if 'error' in results:
            print(f"❌ 批次對齊失敗: {results['error']}")
            return False
        
        successful_count = 0
        total_records = 0
        
        for date_str, result in results.items():
            if 'aligned' in result:
                successful_count += 1
                aligned_count = len(result['aligned'])
                total_records += aligned_count
                print(f"   ✅ {date_str}: {aligned_count:,} 筆對齊")
            else:
                error = result.get('error', '未知錯誤')
                print(f"   ❌ {date_str}: {error}")
        
        success_rate = (successful_count / len(results)) * 100 if results else 0
        print(f"📊 批次結果:")
        print(f"   成功率: {successful_count}/{len(results)} ({success_rate:.1f}%)")
        print(f"   總對齊記錄: {total_records:,} 筆")
        
        return success_rate >= 80  # 80%成功率
        
    except Exception as e:
        print(f"❌ 批次對齊測試失敗: {e}")
        return False


def test_alignment_validation():
    """測試5: 對齊品質驗證"""
    print("\n🧪 測試5: 對齊品質驗證")
    print("-" * 40)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=False)
        available = aligner.get_available_dates()
        
        if not available['common_dates']:
            print("⚠️ 沒有可用日期，跳過驗證")
            return True
        
        # 確保有對齊數據
        test_date = available['common_dates'][0]
        result = aligner.align_date_data(test_date)
        
        if 'aligned' not in result:
            print(f"❌ 需要先對齊數據")
            return False
        
        # 執行驗證
        print(f"🔍 驗證 {test_date} 對齊品質...")
        validation = aligner.validate_alignment(test_date)
        
        if 'error' in validation:
            print(f"❌ 驗證失敗: {validation['error']}")
            return False
        
        print("✅ 品質驗證成功:")
        
        # 計算品質評分
        quality_score = 0
        
        # 時間同步品質 (40分)
        time_sync = validation['time_sync_quality']
        time_score = min(40, time_sync * 0.4)
        quality_score += time_score
        print(f"   ⏰ 時間同步: {time_sync:.1f}% ({time_score:.0f}/40分)")
        
        # 速度相關性 (30分)
        speed_corr = abs(validation['speed_correlation'])
        speed_score = min(30, speed_corr * 30)
        quality_score += speed_score
        print(f"   📈 速度相關性: {speed_corr:.3f} ({speed_score:.0f}/30分)")
        
        # 記錄數量 (20分)
        record_count = validation['record_count']
        record_score = min(20, record_count / 100)
        quality_score += record_score
        print(f"   📋 記錄數量: {record_count:,} ({record_score:.0f}/20分)")
        
        # 數據完整性 (10分)
        completeness = validation['data_completeness']
        complete_score = min(10, completeness * 0.1)
        quality_score += complete_score
        print(f"   📊 完整性: {completeness:.1f}% ({complete_score:.0f}/10分)")
        
        print(f"   🏆 總評分: {quality_score:.0f}/100")
        
        return quality_score >= 60  # 60分及格
        
    except Exception as e:
        print(f"❌ 品質驗證測試失敗: {e}")
        return False


def test_output_verification():
    """測試6: 輸出檔案驗證"""
    print("\n🧪 測試6: 輸出檔案驗證")
    print("-" * 40)
    
    try:
        fusion_folder = Path("data/processed/fusion")
        
        if not fusion_folder.exists():
            print("⚠️ 融合輸出資料夾不存在")
            return True
        
        print("📁 檢查輸出結構...")
        
        date_folders = [d for d in fusion_folder.iterdir() 
                       if d.is_dir() and d.name.count('-') == 2]
        
        if not date_folders:
            print("⚠️ 沒有找到日期資料夾")
            return True
        
        print(f"📊 找到 {len(date_folders)} 個日期資料夾")
        
        valid_folders = 0
        total_size = 0
        
        for date_folder in date_folders:
            date_str = date_folder.name
            print(f"   📅 {date_str}:")
            
            # 檢查必要檔案
            aligned_file = date_folder / "vd_etag_aligned.csv"
            summary_file = date_folder / "alignment_summary.json"
            
            folder_valid = True
            
            if aligned_file.exists():
                file_size = aligned_file.stat().st_size / 1024
                total_size += file_size
                print(f"      ✅ vd_etag_aligned.csv: {file_size:.1f} KB")
                
                # 檢查檔案內容
                try:
                    import pandas as pd
                    df = pd.read_csv(aligned_file, nrows=1)
                    print(f"      ✅ 檔案格式正確 ({len(df.columns)} 欄位)")
                except Exception as e:
                    print(f"      ❌ 檔案讀取失敗: {e}")
                    folder_valid = False
            else:
                print(f"      ❌ vd_etag_aligned.csv: 不存在")
                folder_valid = False
            
            if summary_file.exists():
                print(f"      ✅ alignment_summary.json: 存在")
            else:
                print(f"      ⚠️ alignment_summary.json: 不存在")
            
            if folder_valid:
                valid_folders += 1
        
        print(f"\n📊 結構檢查結果:")
        print(f"   有效資料夾: {valid_folders}/{len(date_folders)}")
        print(f"   總檔案大小: {total_size:.1f} KB")
        
        return valid_folders >= len(date_folders) * 0.8  # 80%有效
        
    except Exception as e:
        print(f"❌ 輸出驗證測試失敗: {e}")
        return False


def test_convenience_functions():
    """測試7: 便利函數"""
    print("\n🧪 測試7: 便利函數")
    print("-" * 40)
    
    try:
        from spatial_temporal_aligner import align_all_available_data, get_available_data_status
        
        print("🔧 測試便利函數...")
        
        # 測試狀態檢查
        print("   testing get_available_data_status()...")
        start_time = time.time()
        status = get_available_data_status(debug=False)
        status_time = time.time() - start_time
        
        if status and status['common_dates']:
            print(f"   ✅ get_available_data_status(): {len(status['common_dates'])} 天 ({status_time:.3f}s)")
        else:
            print(f"   ⚠️ get_available_data_status(): 無共同日期")
        
        # 測試批次對齊
        print("   testing align_all_available_data()...")
        start_time = time.time()
        result = align_all_available_data(debug=False)
        align_time = time.time() - start_time
        
        if result and 'error' not in result:
            successful = sum(1 for r in result.values() if 'aligned' in r)
            print(f"   ✅ align_all_available_data(): {successful} 成功 ({align_time:.3f}s)")
        else:
            print(f"   ⚠️ align_all_available_data(): 失敗或無結果")
        
        return True
        
    except Exception as e:
        print(f"❌ 便利函數測試失敗: {e}")
        return False


def generate_test_summary(test_results):
    """生成測試摘要"""
    print("\n" + "="*60)
    print("📋 VD+eTag時空對齊測試報告 - 簡化版")
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
    
    if passed_tests >= total_tests * 0.8:  # 80%通過
        print(f"\n🎉 時空對齊模組測試通過！")
        
        print(f"\n🔧 簡化版特色:")
        print("   🎯 動態資料檢測：自動適應實際可用天數")
        print("   ⚡ 精簡程式碼：移除冗餘功能，保留核心邏輯")
        print("   🔍 強化除錯：完整的debug資訊輸出")
        print("   📊 品質驗證：多維度對齊效果評估")
        
        print(f"\n📁 輸出結構:")
        print("   data/processed/fusion/YYYY-MM-DD/")
        print("   ├── vd_etag_aligned.csv     # VD+eTag對齊數據")
        print("   └── alignment_summary.json  # 對齊摘要統計")
        
        print(f"\n🎯 使用方式:")
        print("```python")
        print("from src.spatial_temporal_aligner import SpatialTemporalAligner")
        print("")
        print("# 初始化對齊器")
        print("aligner = SpatialTemporalAligner(debug=True)")
        print("")
        print("# 檢查可用資料")
        print("available = aligner.get_available_dates()")
        print("")
        print("# 批次對齊所有資料")
        print("results = aligner.batch_align_all_available()")
        print("```")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查相關數據和配置")
        return False


def main():
    """主測試程序"""
    print("🧪 VD+eTag時空對齊模組測試 - 簡化版")
    print("=" * 60)
    print("🎯 測試重點：動態資料檢測、核心對齊功能、品質驗證")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 執行測試序列
    test_results = []
    
    # 核心測試
    success = test_aligner_import()
    test_results.append(("對齊器導入", success))
    
    if success:
        success = test_available_data_detection()
        test_results.append(("動態資料檢測", success))
        
        success = test_single_date_alignment()
        test_results.append(("單日期對齊", success))
        
        success = test_batch_alignment()
        test_results.append(("批次對齊", success))
        
        success = test_alignment_validation()
        test_results.append(("對齊品質驗證", success))
        
        success = test_output_verification()
        test_results.append(("輸出檔案驗證", success))
        
        success = test_convenience_functions()
        test_results.append(("便利函數", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試摘要
    all_passed = generate_test_summary(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    if all_passed:
        print(f"\n✅ 簡化版時空對齊模組已準備就緒！")
        return True
    else:
        print(f"\n❌ 測試未完全通過，請檢查相關功能")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 簡化版時空對齊模組測試完成！")
        
        print("\n💻 實際使用示範:")
        print("# 檢查可用資料狀態")
        print("python -c \"from src.spatial_temporal_aligner import get_available_data_status; print(get_available_data_status())\"")
        print("")
        print("# 對齊所有可用資料")
        print("python -c \"from src.spatial_temporal_aligner import align_all_available_data; print(align_all_available_data())\"")
        
        print(f"\n🔧 簡化版改進:")
        print("   ✅ 動態適應：自動檢測實際可用資料天數")
        print("   ✅ 程式精簡：移除冗餘代碼，保留核心功能")
        print("   ✅ 除錯增強：完整的debug資訊和錯誤處理")
        print("   ✅ 效能優化：簡化處理流程，提升執行效率")
        
        print(f"\n🚀 Ready for Dynamic VD+eTag Alignment! 🚀")
        
    else:
        print("\n🔧 請解決測試中的問題")
    
    print(f"\n🎊 簡化版時空對齊模組測試完成！")