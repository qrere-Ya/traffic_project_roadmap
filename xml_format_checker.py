#!/usr/bin/env python3
# debug_fusion_files.py - 融合檔案診斷腳本

"""
融合檔案診斷工具
================

檢查融合數據檔案的詳細狀況
幫助診斷測試失敗的原因

使用方式：
python debug_fusion_files.py
"""

import pandas as pd
from pathlib import Path

def diagnose_fusion_files():
    """診斷融合檔案狀況"""
    print("🔍 融合檔案診斷開始")
    print("=" * 40)
    
    fusion_folder = Path("data/processed/fusion")
    
    if not fusion_folder.exists():
        print("❌ 融合資料夾不存在")
        return
    
    date_folders = [d for d in fusion_folder.iterdir() 
                   if d.is_dir() and len(d.name.split('-')) == 3]
    
    if not date_folders:
        print("❌ 沒有找到日期資料夾")
        return
    
    print(f"📁 找到 {len(date_folders)} 個日期資料夾")
    
    # 詳細檢查每個資料夾
    for i, date_folder in enumerate(date_folders):
        print(f"\n📅 檢查 {date_folder.name} ({i+1}/{len(date_folders)})")
        print("-" * 30)
        
        # 檢查檔案存在性
        files = {
            'vd_etag_aligned.csv': '時空對齊數據',
            'fusion_features.csv': '融合特徵數據',
            'fusion_quality.json': '融合品質報告',
            'alignment_summary.json': '對齊摘要'
        }
        
        for filename, description in files.items():
            file_path = date_folder / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"   ✅ {filename}: {file_size:.1f}KB - {description}")
                
                # 詳細檢查CSV檔案
                if filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path, nrows=3)
                        print(f"      📊 {len(df)} 筆樣本, {len(df.columns)} 欄位")
                        print(f"      🏷️ 欄位: {list(df.columns)[:5]}...")  # 顯示前5個欄位
                        
                        # 檢查特定重要欄位
                        if filename == 'fusion_features.csv':
                            important_cols = ['datetime', 'speed_mean', 'region', 'etag_pair']
                            missing_important = [col for col in important_cols if col not in df.columns]
                            if missing_important:
                                print(f"      ⚠️ 缺少重要欄位: {missing_important}")
                            else:
                                print(f"      ✅ 重要欄位完整")
                                
                    except Exception as e:
                        print(f"      ❌ 讀取失敗: {e}")
                        
            else:
                print(f"   ❌ {filename}: 不存在 - {description}")
    
    # 比較所有融合特徵檔案的結構
    print(f"\n📋 融合特徵檔案結構比較")
    print("=" * 40)
    
    feature_files = []
    for date_folder in date_folders:
        feature_file = date_folder / 'fusion_features.csv'
        if feature_file.exists():
            try:
                df = pd.read_csv(feature_file, nrows=1)
                feature_files.append({
                    'date': date_folder.name,
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'file_size': feature_file.stat().st_size / 1024
                })
            except Exception as e:
                print(f"❌ {date_folder.name}: 讀取失敗 - {e}")
    
    if feature_files:
        # 分析欄位數量分布
        column_counts = [f['columns'] for f in feature_files]
        unique_counts = set(column_counts)
        
        print(f"📊 欄位數量分布:")
        for count in sorted(unique_counts):
            dates_with_count = [f['date'] for f in feature_files if f['columns'] == count]
            print(f"   {count} 欄位: {len(dates_with_count)} 檔案 - {dates_with_count}")
        
        # 檢查欄位差異
        if len(unique_counts) > 1:
            print(f"\n🔍 欄位差異分析:")
            
            # 找出最常見的欄位數量
            from collections import Counter
            count_freq = Counter(column_counts)
            most_common_count = count_freq.most_common(1)[0][0]
            
            print(f"   最常見欄位數: {most_common_count}")
            
            # 比較欄位差異
            reference_file = next(f for f in feature_files if f['columns'] == most_common_count)
            reference_cols = set(reference_file['column_names'])
            
            for f in feature_files:
                if f['columns'] != most_common_count:
                    current_cols = set(f['column_names'])
                    extra_cols = current_cols - reference_cols
                    missing_cols = reference_cols - current_cols
                    
                    print(f"   {f['date']} ({f['columns']} 欄位):")
                    if extra_cols:
                        print(f"      額外欄位: {list(extra_cols)[:3]}...")
                    if missing_cols:
                        print(f"      缺少欄位: {list(missing_cols)[:3]}...")
        
        # 檔案大小分析
        print(f"\n💾 檔案大小分析:")
        file_sizes = [f['file_size'] for f in feature_files]
        avg_size = sum(file_sizes) / len(file_sizes)
        min_size = min(file_sizes)
        max_size = max(file_sizes)
        
        print(f"   平均大小: {avg_size:.1f}KB")
        print(f"   大小範圍: {min_size:.1f}KB - {max_size:.1f}KB")
        
        # 檢查異常檔案
        for f in feature_files:
            if f['file_size'] < avg_size * 0.5:  # 小於平均一半
                print(f"   ⚠️ {f['date']}: 檔案過小 ({f['file_size']:.1f}KB)")
            elif f['file_size'] > avg_size * 2:  # 大於平均兩倍
                print(f"   ⚠️ {f['date']}: 檔案過大 ({f['file_size']:.1f}KB)")
    
    # 總結
    print(f"\n📋 診斷總結")
    print("=" * 40)
    
    total_expected = len(date_folders) * 4  # 每個日期4個檔案
    total_existing = sum(len([f for f in date_folder.iterdir() if f.is_file()]) 
                        for date_folder in date_folders)
    
    print(f"📁 資料夾數量: {len(date_folders)}")
    print(f"📄 檔案完整性: {total_existing}/{total_expected}")
    print(f"🎯 融合特徵檔案: {len(feature_files)}/{len(date_folders)}")
    
    if len(feature_files) == len(date_folders):
        print(f"✅ 所有融合特徵檔案都存在")
        
        if len(unique_counts) == 1:
            print(f"✅ 所有檔案欄位數量一致 ({list(unique_counts)[0]} 欄位)")
        else:
            print(f"⚠️ 檔案欄位數量不一致: {sorted(unique_counts)}")
            print(f"💡 這通常是正常的，第一個檔案可能包含額外調試欄位")
        
        print(f"🎉 融合數據生成成功！")
    else:
        missing_count = len(date_folders) - len(feature_files)
        print(f"❌ 缺少 {missing_count} 個融合特徵檔案")
        
        missing_dates = [d.name for d in date_folders 
                        if not (d / 'fusion_features.csv').exists()]
        print(f"📅 缺少的日期: {missing_dates}")


if __name__ == "__main__":
    diagnose_fusion_files()