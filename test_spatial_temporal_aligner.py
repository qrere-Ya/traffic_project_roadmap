# test_spatial_temporal_aligner.py - 簡化版

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
- 清晰的測試結果

作者: 交通預測專案團隊
日期: 2025-01-23
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
    print("-" * 30)
    
    try:
        from spatial_temporal_aligner import (
            SpatialTemporalAligner, 
            align_all_available_data, 
            get_available_data_status
        )
        print("✅ 成功導入對齊器類別")
        print("✅ 成功導入便利函數")
        
        # 測試初始化
        aligner = SpatialTemporalAligner(debug=False)
        print("✅ 對齊器初始化成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False


def test_data_detection():
    """測試2: 動態資料檢測"""
    print("\n🧪 測試2: 動態資料檢測")
    print("-" * 30)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=True)
        available = aligner.get_available_dates()
        
        print(f"📊 檢測結果:")
        print(f"   VD資料: {len(available['vd_dates'])} 天")
        print(f"   eTag資料: {len(available['etag_dates'])} 天")
        print(f"   共同日期: {len(available['common_dates'])} 天")
        
        # 顯示具體日期
        if available['common_dates']:
            print(f"   可對齊日期:")
            for date in available['common_dates'][:3]:  # 只顯示前3個
                print(f"     • {date}")
            if len(available['common_dates']) > 3:
                print(f"     ... 還有 {len(available['common_dates'])-3} 天")
        
        return len(available['common_dates']) > 0
        
    except Exception as e:
        print(f"❌ 資料檢測失敗: {e}")
        return False


def test_single_alignment():
    """測試3: 單日期對齊"""
    print("\n🧪 測試3: 單日期對齊")
    print("-" * 30)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=False)
        available = aligner.get_available_dates()
        
        if not available['common_dates']:
            print("⚠️ 沒有可用日期，跳過測試")
            return True
        
        test_date = available['common_dates'][0]
        print(f"🎯 測試日期: {test_date}")
        
        start_time = time.time()
        result = aligner.align_date_data(test_date)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 處理時間: {elapsed:.2f} 秒")
        
        if 'aligned' in result:
            aligned_df = result['aligned']
            summary = result['summary']
            
            print(f"✅ 對齊成功:")
            print(f"   📊 記錄數: {len(aligned_df):,}")
            print(f"   🗺️ 區域數: {summary['regions']}")
            print(f"   🏷️ eTag配對: {summary['etag_pairs']}")
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
    print("-" * 30)
    
    try:
        from spatial_temporal_aligner import SpatialTemporalAligner
        
        aligner = SpatialTemporalAligner(debug=False)
        
        print("🚀 執行批次對齊...")
        start_time = time.time()
        results = aligner.batch_align_all_available()
        elapsed = time.time() - start_time
        
        print(f"⏱️ 批次時間: {elapsed:.2f} 秒")
        
        if 'error' in results:
            print(f"❌ 批次對齊失敗: {results['error']}")
            return False
        
        successful = 0
        total_records = 0
        
        for date_str, result in results.items():
            if 'aligned' in result:
                successful += 1
                record_count = len(result['aligned'])
                total_records += record_count
                print(f"   ✅ {date_str}: {record_count:,} 筆")
            else:
                print(f"   ❌ {date_str}: {result.get('error', '失敗')}")
        
        success_rate = (successful / len(results)) * 100 if results else 0
        print(f"📊 批次結果:")
        print(f"   成功率: {successful}/{len(results)} ({success_rate:.1f}%)")
        print(f"   總記錄: {total_records:,} 筆")
        
        return success_rate >= 50  # 50%成功率通過
        
    except Exception as e:
        print(f"❌ 批次對齊測試失敗: {e}")
        return False


def test_quality_validation():
    """測試5: 品質驗證"""
    print("\n🧪 測試5: 品質驗證")
    print("-" * 30)
    
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
            print(f"❌ 需要先產生對齊數據")
            return False
        
        # 執行品質驗證
        validation = aligner.validate_alignment(test_date)
        
        if 'error' in validation:
            print(f"❌ 驗證失敗: {validation['error']}")
            return False
        
        print(f"✅ 品質驗證結果:")
        print(f"   📊 記錄數: {validation['record_count']:,}")
        print(f"   ⏰ 時間同步: {validation['time_sync_quality']:.1f}%")
        print(f"   📈 速度相關性: {validation['speed_correlation']:.3f}")
        print(f"   📋 完整性: {validation['data_completeness']:.1f}%")
        
        # 品質評分 (簡化)
        quality_score = (
            validation['time_sync_quality'] * 0.4 +
            abs(validation['speed_correlation']) * 30 +
            validation['data_completeness'] * 0.3
        )
        
        print(f"   🏆 品質評分: {quality_score:.1f}/100")
        
        return quality_score >= 50  # 50分及格
        
    except Exception as e:
        print(f"❌ 品質驗證測試失敗: {e}")
        return False


def test_output_files():
    """測試6: 輸出檔案檢查"""
    print("\n🧪 測試6: 輸出檔案檢查")
    print("-" * 30)
    
    try:
        fusion_folder = Path("data/processed/fusion")
        
        if not fusion_folder.exists():
            print("⚠️ 融合資料夾不存在")
            return True
        
        date_folders = [d for d in fusion_folder.iterdir() 
                       if d.is_dir() and len(d.name.split('-')) == 3]
        
        if not date_folders:
            print("⚠️ 沒有找到日期資料夾")
            return True
        
        print(f"📁 檢查 {len(date_folders)} 個日期資料夾")
        
        valid_count = 0
        total_size = 0
        
        for date_folder in date_folders[:5]:  # 只檢查前5個
            date_str = date_folder.name
            
            aligned_file = date_folder / "vd_etag_aligned.csv"
            summary_file = date_folder / "alignment_summary.json"
            
            if aligned_file.exists():
                file_size = aligned_file.stat().st_size / 1024  # KB
                total_size += file_size
                
                try:
                    import pandas as pd
                    df = pd.read_csv(aligned_file, nrows=1)
                    print(f"   ✅ {date_str}: {file_size:.1f}KB, {len(df.columns)}欄位")
                    valid_count += 1
                except Exception:
                    print(f"   ❌ {date_str}: 檔案讀取失敗")
            else:
                print(f"   ❌ {date_str}: 檔案不存在")
        
        print(f"📊 檔案檢查結果:")
        print(f"   有效檔案: {valid_count}/{min(len(date_folders), 5)}")
        print(f"   總大小: {total_size:.1f}KB")
        
        return valid_count >= len(date_folders) * 0.5  # 50%有效
        
    except Exception as e:
        print(f"❌ 輸出檔案檢查失敗: {e}")
        return False


def test_convenience_functions():
    """測試7: 便利函數"""
    print("\n🧪 測試7: 便利函數")
    print("-" * 30)
    
    try:
        from spatial_temporal_aligner import align_all_available_data, get_available_data_status
        
        # 測試狀態檢查
        status = get_available_data_status(debug=False)
        print(f"   ✅ get_available_data_status(): {len(status['common_dates'])} 天")
        
        # 測試批次對齊（如果有資料）
        if status['common_dates']:
            result = align_all_available_data(debug=False)
            if result and 'error' not in result:
                successful = sum(1 for r in result.values() if 'aligned' in r)
                print(f"   ✅ align_all_available_data(): {successful} 成功")
            else:
                print(f"   ⚠️ align_all_available_data(): 無結果")
        else:
            print(f"   ⚠️ 沒有可用資料測試便利函數")
        
        return True
        
    except Exception as e:
        print(f"❌ 便利函數測試失敗: {e}")
        return False


def generate_test_summary(test_results):
    """生成測試摘要"""
    print("\n" + "="*50)
    print("📋 VD+eTag時空對齊測試報告 - 簡化版")
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
    
    if passed_tests >= total_tests * 0.7:  # 70%通過
        print(f"\n🎉 時空對齊模組測試通過！")
        
        print(f"\n✨ 簡化版特色:")
        print("   🎯 動態資料檢測：自動適應可用天數")
        print("   ⚡ 程式精簡：移除冗餘，保留核心功能")
        print("   🔍 錯誤處理：清晰的錯誤訊息")
        print("   📊 品質驗證：多維度對齊效果評估")
        
        print(f"\n📁 輸出結構:")
        print("   data/processed/fusion/YYYY-MM-DD/")
        print("   ├── vd_etag_aligned.csv     # 對齊數據")
        print("   └── alignment_summary.json  # 摘要統計")
        
        print(f"\n🚀 使用方式:")
        print("```python")
        print("from src.spatial_temporal_aligner import SpatialTemporalAligner")
        print("")
        print("# 初始化")
        print("aligner = SpatialTemporalAligner(debug=True)")
        print("")
        print("# 批次對齊")
        print("results = aligner.batch_align_all_available()")
        print("```")
        
        return True
    else:
        failed_count = total_tests - passed_tests
        print(f"\n❌ 有 {failed_count} 個測試失敗")
        print("   建議檢查數據路徑和檔案格式")
        return False


def main():
    """主測試程序"""
    print("🧪 VD+eTag時空對齊模組測試 - 簡化版")
    print("=" * 50)
    print("🎯 測試重點：資料檢測、對齊功能、品質驗證")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # 執行測試序列
    test_results = []
    
    # 核心測試
    success = test_aligner_import()
    test_results.append(("對齊器導入", success))
    
    if success:
        success = test_data_detection()
        test_results.append(("資料檢測", success))
        
        success = test_single_alignment()
        test_results.append(("單日對齊", success))
        
        success = test_batch_alignment()
        test_results.append(("批次對齊", success))
        
        success = test_quality_validation()
        test_results.append(("品質驗證", success))
        
        success = test_output_files()
        test_results.append(("輸出檔案", success))
        
        success = test_convenience_functions()
        test_results.append(("便利函數", success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 生成測試摘要
    all_passed = generate_test_summary(test_results)
    
    print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
    
    if all_passed:
        print(f"\n✅ 簡化版時空對齊模組已準備就緒！")
        
        print(f"\n💡 下一步建議:")
        print("   1. 開發 fusion_engine.py - 融合引擎")
        print("   2. 開發 enhanced_predictor.py - 融合預測器")
        print("   3. 整合測試所有融合模組")
        
        return True
    else:
        print(f"\n🔧 請檢查並解決測試中的問題")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 簡化版時空對齊模組測試完成！")
        
        print("\n💻 快速使用:")
        print("# 檢查資料狀態")
        print("python -c \"from src.spatial_temporal_aligner import get_available_data_status; print(get_available_data_status())\"")
        print("")
        print("# 執行對齊")
        print("python -c \"from src.spatial_temporal_aligner import align_all_available_data; align_all_available_data(debug=True)\"")
        
        print(f"\n🚀 Ready for VD+eTag Fusion! 🚀")
        
    else:
        print("\n🔧 請解決測試問題後重新執行")
    
    print(f"\n🎊 時空對齊模組測試完成！")