# test_cleaner.py - 日期資料夾組織版

"""
VD批次數據清理器測試程式 - 日期資料夾組織版
===============================================

新增測試功能：
1. 測試按日期組織的清理：data/cleaned/2025-06-27/
2. 測試多日期資料夾批次清理
3. 測試日期資料夾檢測功能
4. 測試指定日期清理數據載入

- 自動檢測 data/processed/YYYY-MM-DD/ 中的檔案
- 批次清理並保存到 data/cleaned/YYYY-MM-DD/
- 生成完整的日期組織清理報告
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# 添加src目錄到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_date_folder_detection():
    """測試1: 日期資料夾檢測功能"""
    print("🧪 測試1: 日期資料夾檢測功能")
    print("-" * 70)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        print("✅ 成功導入 VDBatchDataCleaner 日期組織版")
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        print("💡 請確認 data_cleaner.py 在 src/ 目錄中")
        return False
    
    # 初始化批次清理器
    print("\n2️⃣ 初始化日期組織批次清理器...")
    try:
        cleaner = VDBatchDataCleaner()
        print("✅ 日期組織批次清理器初始化成功")
        print(f"   📁 輸入基礎目錄: {cleaner.processed_base_folder}")
        print(f"   📁 輸出基礎目錄: {cleaner.cleaned_base_folder}")
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False
    
    # 檢測可用日期資料夾
    print("\n3️⃣ 檢測可用的日期資料夾...")
    available_date_folders = cleaner.detect_available_date_folders()
    
    if not available_date_folders:
        print("❌ 找不到可清理的日期資料夾！")
        print("💡 請先執行以下步驟：")
        print("   1. python test_loader.py  (生成日期組織的分類檔案)")
        print("   2. 確認 data/processed/YYYY-MM-DD/ 目錄中有分類檔案")
        print("   3. 預期結構:")
        print("      📂 data/processed/")
        print("         ├── 2025-06-27/")
        print("         │   ├── vd_data_all.csv")
        print("         │   ├── vd_data_peak.csv")
        print("         │   └── ...")
        print("         └── 2025-06-26/")
        return False
    
    print(f"✅ 找到 {len(available_date_folders)} 個可清理的日期資料夾")
    
    # 顯示各日期資料夾詳情
    total_files = 0
    total_size_mb = 0
    
    for date_str, date_info in sorted(available_date_folders.items()):
        file_count = date_info['file_count']
        available_files = date_info['available_files']
        
        print(f"\n   📅 {date_str}:")
        print(f"      檔案數: {file_count}")
        
        date_size = 0
        for name, file_info in available_files.items():
            if file_info.get('status') == 'ready':
                size_mb = file_info['file_size_mb']
                records = file_info['estimated_records']
                description = file_info['description']
                
                print(f"         ✅ {description}: {size_mb:.1f}MB (~{records:,} 記錄)")
                date_size += size_mb
            else:
                print(f"         ❌ {file_info['description']}: {file_info.get('error', '檔案問題')}")
        
        print(f"      總大小: {date_size:.1f}MB")
        total_files += file_count
        total_size_mb += date_size
    
    print(f"\n📊 總計: {len(available_date_folders)} 個日期, {total_files} 個檔案, {total_size_mb:.1f}MB")
    
    return True


def test_single_date_folder_cleaning():
    """測試2: 單一日期資料夾清理"""
    print("\n🧪 測試2: 單一日期資料夾清理")
    print("-" * 50)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        
        cleaner = VDBatchDataCleaner()
        
        # 檢測可用日期資料夾
        available_date_folders = cleaner.detect_available_date_folders()
        
        if not available_date_folders:
            print("⚠️ 沒有可清理的日期資料夾")
            return True
        
        # 選擇第一個日期資料夾進行測試
        test_date = list(available_date_folders.keys())[0]
        date_info = available_date_folders[test_date]
        
        print(f"🧹 測試清理日期資料夾: {test_date}")
        print(f"   可清理檔案數: {date_info['file_count']}")
        
        try:
            # 執行單一日期資料夾清理
            date_result = cleaner.clean_single_date_folder(test_date, date_info, method='mark_nan')
            
            if date_result['successful_cleanings'] > 0:
                print(f"✅ 單一日期資料夾清理成功")
                print(f"   清理日期: {date_result['date']}")
                print(f"   成功清理: {date_result['successful_cleanings']} 個檔案")
                print(f"   失敗清理: {date_result['failed_cleanings']} 個檔案")
                print(f"   成功率: {date_result['success_rate']}")
                print(f"   輸出資料夾: {date_result['cleaned_folder']}")
                
                # 檢查輸出檔案
                cleaned_folder = Path(date_result['cleaned_folder'])
                if cleaned_folder.exists():
                    output_files = list(cleaned_folder.glob("*.csv"))
                    print(f"   生成CSV檔案: {len(output_files)} 個")
                    
                    for csv_file in output_files:
                        file_size = csv_file.stat().st_size / 1024 / 1024
                        print(f"      • {csv_file.name}: {file_size:.1f}MB")
                
                return True
            else:
                print(f"❌ 單一日期資料夾清理失敗")
                return False
                
        except Exception as e:
            print(f"❌ 單一日期資料夾清理測試失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False


def test_cleaning_methods():
    """測試3: 不同清理方法測試"""
    print("\n🧪 測試3: 不同清理方法測試")
    print("-" * 50)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        
        cleaner = VDBatchDataCleaner()
        
        # 檢測可用日期資料夾
        available_date_folders = cleaner.detect_available_date_folders()
        
        if not available_date_folders:
            print("⚠️ 沒有可清理的日期資料夾")
            return True
        
        # 選擇最小的日期資料夾進行方法測試
        smallest_date = min(available_date_folders.items(), 
                           key=lambda x: sum(f.get('file_size_mb', 0) for f in x[1]['available_files'].values()))
        
        test_date = smallest_date[0]
        date_info = smallest_date[1]
        
        print(f"🧪 使用 {test_date} 進行清理方法測試")
        
        # 測試不同清理方法
        cleaning_methods = [
            ('mark_nan', '標記為NaN'),
            ('remove_rows', '刪除異常行')
        ]
        
        method_results = {}
        
        # 獲取測試檔案（選擇最小的檔案）
        available_files = date_info['available_files']
        test_file_info = None
        test_file_name = None
        
        for name, file_info in available_files.items():
            if file_info.get('status') == 'ready':
                test_file_info = file_info
                test_file_name = name
                break
        
        if not test_file_info:
            print("❌ 沒有可用的測試檔案")
            return False
        
        print(f"   測試檔案: {test_file_info['description']}")
        
        for method, description in cleaning_methods:
            print(f"\n   🧪 測試 {method} - {description}")
            
            try:
                # 載入原始數據進行測試
                input_csv = test_file_info['input_path']
                df_test = pd.read_csv(input_csv)
                
                # 優化數據類型
                df_test = cleaner._optimize_data_types(df_test)
                
                # 識別無效數據
                invalid_stats = cleaner._identify_invalid_data_quick(df_test)
                
                # 應用清理方法
                df_cleaned_test = cleaner._clean_invalid_values(df_test, method)
                
                # 計算效果
                original_count = len(df_test)
                cleaned_count = len(df_cleaned_test)
                removed_records = original_count - cleaned_count
                
                method_results[method] = {
                    'description': description,
                    'original_records': original_count,
                    'cleaned_records': cleaned_count,
                    'records_removed': removed_records,
                    'removal_percentage': round(removed_records / original_count * 100, 2),
                    'invalid_values_found': invalid_stats['total_invalid'],
                    'invalid_percentage': invalid_stats['invalid_percentage']
                }
                
                print(f"      ✅ 測試成功")
                print(f"         原始記錄: {original_count:,}")
                print(f"         清理後記錄: {cleaned_count:,}")
                print(f"         移除記錄: {removed_records:,} ({method_results[method]['removal_percentage']:.1f}%)")
                print(f"         發現異常值: {invalid_stats['total_invalid']:,} ({invalid_stats['invalid_percentage']:.2f}%)")
                
            except Exception as e:
                print(f"      ❌ 測試失敗: {e}")
                method_results[method] = {
                    'description': description,
                    'success': False,
                    'error': str(e)
                }
        
        # 顯示方法比較
        print(f"\n📊 清理方法比較:")
        for method, result in method_results.items():
            if result.get('success', True):
                print(f"   {method}: 保留 {result['cleaned_records']:,}/{result['original_records']:,} 記錄 ({100-result['removal_percentage']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 清理方法測試失敗: {e}")
        return False


def test_batch_date_cleaning():
    """測試4: 批次日期清理"""
    print("\n🧪 測試4: 批次日期清理")
    print("-" * 50)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        
        cleaner = VDBatchDataCleaner()
        
        # 檢測可用日期資料夾
        available_date_folders = cleaner.detect_available_date_folders()
        
        if not available_date_folders:
            print("⚠️ 沒有可清理的日期資料夾")
            return True
        
        print(f"🚀 開始批次清理 {len(available_date_folders)} 個日期資料夾...")
        
        # 選擇推薦的清理方法
        recommended_method = 'mark_nan'
        print(f"   使用推薦方法: {recommended_method}")
        
        try:
            start_time = datetime.now()
            
            # 執行批次清理
            batch_report = cleaner.clean_all_date_folders(method=recommended_method)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if batch_report['清理統計']['成功清理日期數'] > 0:
                print(f"✅ 批次清理成功完成")
                
                stats = batch_report['清理統計']
                print(f"   ⏱️ 耗時: {duration:.1f} 秒")
                print(f"   📅 成功日期: {stats['成功清理日期數']}/{stats['總日期資料夾數']}")
                print(f"   📄 成功檔案: {stats['總成功檔案數']}")
                print(f"   ❌ 失敗檔案: {stats['總失敗檔案數']}")
                print(f"   📈 日期成功率: {stats['日期成功率']}")
                
                # 顯示各日期清理結果
                print(f"\n📅 各日期清理結果:")
                for date_result in batch_report['各日期清理結果']:
                    if 'successful_cleanings' in date_result:
                        date_str = date_result['date']
                        success_count = date_result['successful_cleanings']
                        total_count = date_result['total_files']
                        print(f"      {date_str}: {success_count}/{total_count} 檔案清理成功")
                
                return True
            else:
                print(f"❌ 批次清理失敗")
                return False
                
        except Exception as e:
            print(f"❌ 批次清理過程失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 批次清理測試失敗: {e}")
        return False


def test_cleaned_data_verification():
    """測試5: 清理數據驗證"""
    print("\n🧪 測試5: 清理數據驗證")
    print("-" * 50)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        
        cleaner = VDBatchDataCleaner()
        
        # 獲取清理後檔案摘要
        print("📊 獲取清理後檔案摘要...")
        summary = cleaner.get_cleaned_files_summary()
        
        if summary['清理檔案統計']['清理日期數'] == 0:
            print("⚠️ 沒有找到清理後檔案")
            return True
        
        stats = summary['清理檔案統計']
        print(f"✅ 清理後統計:")
        print(f"   清理日期數: {stats['清理日期數']}")
        print(f"   總檔案數: {stats['存在檔案數']}")
        print(f"   總記錄數: {stats['總記錄數']:,}")
        print(f"   總檔案大小: {stats['檔案大小_MB']:.1f} MB")
        
        # 檢查各日期的清理結果
        print(f"\n📅 各日期清理結果:")
        for date_str, date_details in summary['各日期詳情'].items():
            print(f"   {date_str}:")
            print(f"      檔案數: {date_details['檔案數']}")
            print(f"      記錄數: {date_details['總記錄數']:,}")
            print(f"      檔案大小: {date_details['總檔案大小_MB']:.1f}MB")
        
        # 驗證檔案完整性
        print(f"\n🔍 驗證檔案完整性...")
        
        available_cleaned_dates = cleaner.list_available_cleaned_dates()
        verification_passed = 0
        total_verifications = 0
        
        for date_str in available_cleaned_dates:
            print(f"   📅 驗證 {date_str}:")
            
            # 檢查該日期的清理檔案
            cleaned_date_folder = cleaner.cleaned_base_folder / date_str
            
            for name, mapping in cleaner.file_mappings.items():
                output_csv = cleaned_date_folder / mapping['output_csv']
                description = mapping['description']
                total_verifications += 1
                
                if output_csv.exists():
                    try:
                        df_verify = pd.read_csv(output_csv, nrows=10)
                        print(f"      ✅ {description}: 可正常讀取")
                        verification_passed += 1
                    except Exception as e:
                        print(f"      ❌ {description}: 讀取失敗 - {e}")
                else:
                    print(f"      ⚠️ {description}: 檔案不存在")
        
        print(f"\n📈 檔案完整性: {verification_passed}/{total_verifications} 通過驗證 ({verification_passed/total_verifications*100:.1f}%)")
        
        return verification_passed >= total_verifications * 0.8  # 至少80%通過
        
    except Exception as e:
        print(f"❌ 清理數據驗證失敗: {e}")
        return False


def test_date_specific_loading():
    """測試6: 指定日期清理數據載入"""
    print("\n🧪 測試6: 指定日期清理數據載入")
    print("-" * 50)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        
        cleaner = VDBatchDataCleaner()
        
        # 獲取可用的清理日期
        available_cleaned_dates = cleaner.list_available_cleaned_dates()
        
        if not available_cleaned_dates:
            print("⚠️ 沒有可用的清理日期數據")
            return True
        
        print(f"📅 可用清理日期: {available_cleaned_dates}")
        
        # 測試載入特定日期的清理數據
        test_date = available_cleaned_dates[0]
        print(f"\n🎯 測試載入 {test_date} 清理數據...")
        
        try:
            start_time = datetime.now()
            cleaned_data = cleaner.load_cleaned_data_by_date(test_date)
            load_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   ⏱️ 載入時間: {load_time:.3f} 秒")
            
            # 檢查載入結果
            file_descriptions = {
                'all': '全部VD資料',
                'peak': '尖峰時段數據',
                'offpeak': '離峰時段數據',
                'target_route': '目標路段數據',
                'target_route_peak': '目標路段尖峰',
                'target_route_offpeak': '目標路段離峰'
            }
            
            total_records = 0
            loaded_files = 0
            
            for name, description in file_descriptions.items():
                df = cleaned_data.get(name, pd.DataFrame())
                if not df.empty:
                    print(f"   ✅ {description}: {len(df):,} 筆記錄")
                    total_records += len(df)
                    loaded_files += 1
                else:
                    print(f"   ⚠️ {description}: 無數據")
            
            print(f"\n   📊 {test_date} 載入統計:")
            print(f"      成功載入檔案: {loaded_files}/{len(file_descriptions)}")
            print(f"      總記錄數: {total_records:,}")
            print(f"      載入速度: {total_records/load_time:,.0f} 記錄/秒")
            
            # 驗證數據品質
            if 'all' in cleaned_data and not cleaned_data['all'].empty:
                df_all = cleaned_data['all']
                
                print(f"\n   🔍 數據品質檢查:")
                
                # 檢查關鍵欄位
                key_columns = ['speed', 'occupancy', 'volume_total']
                for col in key_columns:
                    if col in df_all.columns:
                        valid_count = df_all[col].notna().sum()
                        completeness = valid_count / len(df_all) * 100
                        print(f"      {col}: {completeness:.1f}% 完整度")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 指定日期載入失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 指定日期載入測試失敗: {e}")
        return False


def test_convenience_functions():
    """測試7: 便利函數"""
    print("\n🧪 測試7: 便利函數")
    print("-" * 50)
    
    try:
        from data_cleaner import (
            clean_all_vd_data_by_date,
            get_cleaned_data_summary_by_date,
            load_cleaned_data_by_date
        )
        
        print("🔧 測試便利函數導入...")
        print("   ✅ 成功導入所有日期組織便利函數")
        
        # 測試摘要便利函數
        print("\n📊 測試摘要便利函數...")
        summary = get_cleaned_data_summary_by_date()
        
        if summary and summary['清理檔案統計']['清理日期數'] > 0:
            dates_count = summary['清理檔案統計']['清理日期數']
            files_count = summary['清理檔案統計']['存在檔案數']
            print(f"   ✅ get_cleaned_data_summary_by_date(): {dates_count} 個日期, {files_count} 個檔案")
        else:
            print(f"   ⚠️ get_cleaned_data_summary_by_date(): 無結果")
        
        # 測試載入便利函數
        print("\n📂 測試載入便利函數...")
        
        # 檢查是否有可用的清理日期
        cleaned_base = Path("data/cleaned")
        available_dates = []
        
        if cleaned_base.exists():
            available_dates = [d.name for d in cleaned_base.iterdir() 
                             if d.is_dir() and d.name.count('-') == 2]
        
        if available_dates:
            test_date = available_dates[0]
            try:
                date_data = load_cleaned_data_by_date(target_date=test_date)
                
                if date_data:
                    total_records = sum(len(df) for df in date_data.values() if not df.empty)
                    loaded_count = len([df for df in date_data.values() if not df.empty])
                    print(f"   ✅ load_cleaned_data_by_date({test_date}): {loaded_count} 檔案, {total_records:,} 筆記錄")
                else:
                    print(f"   ⚠️ load_cleaned_data_by_date({test_date}): 無結果")
            except Exception as e:
                print(f"   ❌ load_cleaned_data_by_date({test_date}): 載入失敗 - {e}")
        else:
            print(f"   ⚠️ 沒有可用的清理日期資料夾測試")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 便利函數測試失敗: {e}")
        return False


def test_output_structure_verification():
    """測試8: 輸出結構驗證"""
    print("\n🧪 測試8: 輸出結構驗證")
    print("-" * 50)
    
    try:
        from data_cleaner import VDBatchDataCleaner
        
        cleaner = VDBatchDataCleaner()
        
        # 檢查清理基礎資料夾
        if not cleaner.cleaned_base_folder.exists():
            print("⚠️ 清理基礎資料夾不存在")
            return True
        
        print(f"📂 檢查輸出結構: {cleaner.cleaned_base_folder}")
        
        # 檢查日期資料夾結構
        date_folders = [d for d in cleaner.cleaned_base_folder.iterdir() 
                       if d.is_dir() and d.name.count('-') == 2]
        
        if not date_folders:
            print("⚠️ 沒有找到日期資料夾")
            return True
        
        print(f"📅 找到 {len(date_folders)} 個日期資料夾")
        
        # 檢查每個日期資料夾的結構
        expected_files = {
            'vd_data_all_cleaned.csv': '全部VD資料',
            'vd_data_peak_cleaned.csv': '尖峰時段數據',
            'vd_data_offpeak_cleaned.csv': '離峰時段數據',
            'target_route_data_cleaned.csv': '目標路段數據',
            'target_route_peak_cleaned.csv': '目標路段尖峰',
            'target_route_offpeak_cleaned.csv': '目標路段離峰'
        }
        
        total_structure_score = 0
        max_structure_score = len(date_folders) * len(expected_files)
        
        for date_folder in sorted(date_folders):
            date_str = date_folder.name
            print(f"\n   📅 檢查 {date_str}:")
            
            date_score = 0
            date_total_size = 0
            
            for filename, description in expected_files.items():
                csv_file = date_folder / filename
                json_file = date_folder / filename.replace('.csv', '_summary.json')
                report_file = date_folder / "date_cleaning_report.json"
                
                csv_exists = csv_file.exists()
                json_exists = json_file.exists()
                
                csv_status = "✅" if csv_exists else "❌"
                json_status = "✅" if json_exists else "❌"
                
                print(f"      {description}:")
                print(f"        CSV {csv_status} {filename}")
                print(f"        JSON {json_status} {json_file.name}")
                
                if csv_exists:
                    date_score += 1
                    total_structure_score += 1
                    
                    # 檢查檔案大小
                    file_size = csv_file.stat().st_size / 1024 / 1024
                    date_total_size += file_size
                    print(f"        📊 大小: {file_size:.1f}MB")
                    
                    # 快速檢查檔案內容
                    try:
                        df = pd.read_csv(csv_file, nrows=5)
                        print(f"        📄 欄位數: {len(df.columns)}")
                        
                        # 檢查必要欄位
                        required_columns = ['date', 'update_time', 'vd_id', 'speed', 'occupancy']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if missing_columns:
                            print(f"        ⚠️ 缺少欄位: {missing_columns}")
                        else:
                            print(f"        ✅ 核心欄位完整")
                            
                    except Exception as e:
                        print(f"        ❌ 讀取錯誤: {e}")
            
            # 檢查日期清理報告
            if report_file.exists():
                print(f"      ✅ 日期清理報告: {report_file.name}")
            else:
                print(f"      ⚠️ 缺少日期清理報告")
            
            print(f"      📈 {date_str} 完整性: {date_score}/{len(expected_files)} ({date_score/len(expected_files)*100:.1f}%)")
            print(f"      💾 {date_str} 總大小: {date_total_size:.1f}MB")
        
        # 檢查批次清理報告
        batch_report_file = cleaner.cleaned_base_folder / "batch_date_cleaning_report.json"
        if batch_report_file.exists():
            print(f"\n✅ 批次清理報告: {batch_report_file.name}")
            
            # 讀取報告內容
            try:
                with open(batch_report_file, 'r', encoding='utf-8') as f:
                    batch_report = json.load(f)
                
                stats = batch_report.get('清理統計', {})
                print(f"   📊 報告統計:")
                print(f"      成功清理日期數: {stats.get('成功清理日期數', 0)}")
                print(f"      總成功檔案數: {stats.get('總成功檔案數', 0)}")
                print(f"      日期成功率: {stats.get('日期成功率', '0%')}")
                
            except Exception as e:
                print(f"   ❌ 讀取批次報告失敗: {e}")
        else:
            print(f"\n⚠️ 缺少批次清理報告")
        
        print(f"\n📈 總體結構完整性: {total_structure_score}/{max_structure_score} ({total_structure_score/max_structure_score*100:.1f}%)")
        
        # 顯示預期的完整結構
        print(f"\n📁 預期的完整輸出結構:")
        print(f"   📂 data/cleaned/")
        for date_folder in sorted(date_folders):
            print(f"      ├── {date_folder.name}/")
            print(f"      │   ├── vd_data_all_cleaned.csv + _summary.json")
            print(f"      │   ├── vd_data_peak_cleaned.csv + _summary.json")
            print(f"      │   ├── vd_data_offpeak_cleaned.csv + _summary.json")
            print(f"      │   ├── target_route_*_cleaned.csv + _summary.json")
            print(f"      │   └── date_cleaning_report.json")
        print(f"      └── batch_date_cleaning_report.json")
        
        return total_structure_score >= max_structure_score * 0.8  # 至少80%檔案存在
        
    except Exception as e:
        print(f"❌ 輸出結構驗證失敗: {e}")
        return False


def generate_test_summary():
    """生成測試摘要報告"""
    
    print("\n📋 生成測試摘要報告")
    print("=" * 50)
    
    try:
        # 檢查生成的報告檔案
        report_files = [
            "data/cleaned/batch_date_cleaning_report.json",
        ]
        
        existing_reports = []
        for report_file in report_files:
            if os.path.exists(report_file):
                file_size = os.path.getsize(report_file) / 1024  # KB
                existing_reports.append({
                    'file': report_file,
                    'size_kb': round(file_size, 1)
                })
        
        print(f"📁 可用清理報告: {len(existing_reports)} 個")
        for report in existing_reports:
            print(f"   • {report['file']} ({report['size_kb']} KB)")
        
        # 檢查清理數據檔案
        cleaned_base_folder = Path("data/cleaned")
        if cleaned_base_folder.exists():
            date_folders = [d for d in cleaned_base_folder.iterdir() 
                           if d.is_dir() and d.name.count('-') == 2]
            
            total_files = 0
            total_size_mb = 0
            
            for date_folder in date_folders:
                csv_files = list(date_folder.glob("*.csv"))
                folder_size = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
                total_files += len(csv_files)
                total_size_mb += folder_size
            
            print(f"\n📊 清理數據統計:")
            print(f"   日期數: {len(date_folders)}")
            print(f"   檔案數: {total_files}")
            print(f"   總大小: {total_size_mb:.1f} MB")
            
            print(f"\n📅 各日期清理結果:")
            for date_folder in sorted(date_folders):
                csv_files = list(date_folder.glob("*.csv"))
                folder_size = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
                print(f"   {date_folder.name}: {len(csv_files)} 檔案 ({folder_size:.1f}MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 摘要報告生成失敗: {e}")
        return False


def demonstrate_date_organized_usage():
    """示範日期組織使用方法"""
    print("\n💡 日期組織清理使用方法示範")
    print("=" * 60)
    
    print("🔧 基本日期組織清理:")
    print("```python")
    print("from src.data_cleaner import VDBatchDataCleaner")
    print("")
    print("# 初始化日期組織批次清理器")
    print("cleaner = VDBatchDataCleaner()")
    print("")
    print("# 檢測可用日期資料夾")
    print("available_dates = cleaner.detect_available_date_folders()")
    print("")
    print("# 批次清理所有日期資料夾")
    print("report = cleaner.clean_all_date_folders()")
    print("")
    print("# 檢查清理結果")
    print("summary = cleaner.get_cleaned_files_summary()")
    print("```")
    
    print("\n⚡ 一鍵日期組織清理:")
    print("```python")
    print("from src.data_cleaner import clean_all_vd_data_by_date")
    print("")
    print("# 一鍵清理所有日期資料夾")
    print("report = clean_all_vd_data_by_date()")
    print("```")
    
    print("\n📅 指定日期載入:")
    print("```python")
    print("from src.data_cleaner import load_cleaned_data_by_date")
    print("")
    print("# 載入特定日期的清理數據")
    print("date_data = load_cleaned_data_by_date(target_date='2025-06-27')")
    print("```")
    
    print("\n📁 清理後檔案結構:")
    print("   📂 data/cleaned/")
    print("      ├── 2025-06-27/")
    print("      │   ├── vd_data_all_cleaned.csv + .json")
    print("      │   ├── vd_data_peak_cleaned.csv + .json")
    print("      │   ├── vd_data_offpeak_cleaned.csv + .json")
    print("      │   ├── target_route_data_cleaned.csv + .json")
    print("      │   ├── target_route_peak_cleaned.csv + .json")
    print("      │   ├── target_route_offpeak_cleaned.csv + .json")
    print("      │   └── date_cleaning_report.json")
    print("      ├── 2025-06-26/")
    print("      │   └── ... (同樣結構)")
    print("      └── batch_date_cleaning_report.json")
    
    print("\n🎯 AI訓練推薦檔案（按日期）:")
    print("   🚀 主要訓練數據:")
    print("      • data/cleaned/2025-06-27/target_route_peak_cleaned.csv")
    print("      • data/cleaned/2025-06-27/target_route_offpeak_cleaned.csv")
    print("      • data/cleaned/2025-06-26/target_route_peak_cleaned.csv")
    print("      • data/cleaned/2025-06-26/target_route_offpeak_cleaned.csv")
    print("   📊 時間序列分析:")
    print("      • 跨日期比較分析")
    print("      • 時間趨勢預測")
    print("      • 週期性模式識別")


def main():
    """主測試程序"""
    print("🧪 VD批次數據清理器日期組織版完整測試")
    print("=" * 80)
    print("這將測試日期組織清理的所有功能，包括:")
    print("• 自動檢測 data/processed/YYYY-MM-DD/ 中的檔案")
    print("• 批次清理並保存到 data/cleaned/YYYY-MM-DD/")
    print("• 多日期資料夾批次處理")
    print("• 指定日期清理數據載入")
    print("• 完整性驗證和品質評估")
    print("• 生成詳細的日期組織清理報告")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 主要測試流程
    test_results = []
    
    # 測試1: 日期資料夾檢測
    success = test_date_folder_detection()
    test_results.append(("日期資料夾檢測", success))
    
    if success:
        # 測試2: 單一日期資料夾清理
        success = test_single_date_folder_cleaning()
        test_results.append(("單一日期資料夾清理", success))
        
        # 測試3: 清理方法測試
        success = test_cleaning_methods()
        test_results.append(("清理方法測試", success))
        
        # 測試4: 批次日期清理
        success = test_batch_date_cleaning()
        test_results.append(("批次日期清理", success))
        
        # 測試5: 清理數據驗證
        success = test_cleaned_data_verification()
        test_results.append(("清理數據驗證", success))
        
        # 測試6: 指定日期載入
        success = test_date_specific_loading()
        test_results.append(("指定日期載入", success))
        
        # 測試7: 便利函數
        success = test_convenience_functions()
        test_results.append(("便利函數測試", success))
        
        # 測試8: 輸出結構驗證
        success = test_output_structure_verification()
        test_results.append(("輸出結構驗證", success))
        
        # 生成測試摘要
        generate_test_summary()
        
        # 使用方法示範
        demonstrate_date_organized_usage()
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # 測試結果統計
    passed_tests = sum(1 for _, success in test_results if success)
    
    print(f"\n🏁 日期組織清理測試完成")
    print("=" * 80)
    print("📋 測試結果:")
    
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   • {test_name}: {status}")
    
    print(f"\n📊 測試統計:")
    print(f"   • 總測試項目: {len(test_results)}")
    print(f"   • 通過測試: {passed_tests}")
    print(f"   • 成功率: {passed_tests/len(test_results)*100:.1f}%")
    print(f"   • 執行時間: {total_duration:.1f} 秒")
    
    if passed_tests == len(test_results):
        print(f"\n🎉 所有測試完成！日期組織清理功能完全就緒！")
        
        print(f"\n🎯 成功完成:")
        print("   ✅ 日期資料夾檢測")
        print("   ✅ 多日期批次清理")
        print("   ✅ 指定日期載入")
        print("   ✅ 清理結果驗證")
        print("   ✅ 輸出結構完整性")
        
        print(f"\n📁 輸出位置:")
        print("   • 清理檔案: data/cleaned/YYYY-MM-DD/")
        print("   • 批次報告: data/cleaned/batch_date_cleaning_report.json")
        print("   • 各日期報告: data/cleaned/YYYY-MM-DD/date_cleaning_report.json")
        
        print(f"\n🚀 下一步行動:")
        print("   1. 檢查 data/cleaned/YYYY-MM-DD/ 中的清理檔案")
        print("   2. 使用按日期組織的清理數據進行時間序列分析")
        print("   3. 重點使用各日期的 target_route_*_cleaned.csv 進行AI訓練")
        print("   4. 執行按日期組織的探索性數據分析")
        
        print(f"\n📅 日期組織優勢:")
        print("   🕰️ 時間序列分析：便於跨日期比較")
        print("   🎯 精準查詢：快速載入特定日期清理數據")
        print("   📊 趨勢分析：識別日期間的變化模式")
        print("   🤖 AI訓練：支援時間序列模型開發")
        
        return True
    else:
        print(f"\n❌ 有 {len(test_results) - passed_tests} 個測試失敗")
        print("   建議檢查相關功能後再使用")
        
        print(f"\n🔧 故障排除:")
        print("   1. 確認是否已執行 test_loader.py 生成日期組織的處理檔案")
        print("   2. 檢查 data/processed/YYYY-MM-DD/ 目錄是否包含所有分類檔案")
        print("   3. 確認 src/data_cleaner.py 檔案是否正確")
        
        return False


if __name__ == "__main__":
    main()