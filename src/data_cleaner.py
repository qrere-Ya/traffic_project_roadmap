# src/data_cleaner.py - 適配版

"""
VD數據清理器 - 適配強化版載入器
===============================

專門適配強化版data_loader.py的輸出格式：
1. 🎯 專注目標路段檔案清理
2. 📁 適配新的檔案結構 (target_route_*.csv)
3. 💾 保持記憶體優化特性
4. ⚡ 簡化清理流程
5. 🔄 完美配合彈性處理
"""

import pandas as pd
import numpy as np
import os
import json
import gc
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')


class VDTargetRouteCleaner:
    """VD目標路段清理器 - 適配強化版載入器"""
    
    def __init__(self, base_folder: str = "data", target_memory_percent: float = 70.0):
        """
        初始化目標路段清理器
        
        Args:
            base_folder: 基礎資料夾路徑
            target_memory_percent: 目標記憶體使用率
        """
        self.base_folder = Path(base_folder)
        self.processed_base_folder = self.base_folder / "processed"
        self.cleaned_base_folder = self.base_folder / "cleaned"
        self.target_memory_percent = target_memory_percent
        
        # 確保資料夾存在
        self.cleaned_base_folder.mkdir(parents=True, exist_ok=True)
        
        # 異常值定義
        self.invalid_markers = [-99, -1, 999, 9999, float('inf'), -float('inf')]
        
        # 合理數值範圍
        self.valid_ranges = {
            'speed': (0, 150),
            'occupancy': (0, 100),
            'volume_total': (0, 100),
            'volume_small': (0, 100),
            'volume_large': (0, 50),
            'volume_truck': (0, 50),
            'speed_small': (0, 150),
            'speed_large': (0, 150),
            'speed_truck': (0, 150)
        }
        
        # 🆕 適配強化版載入器的檔案結構
        self.target_file_mappings = {
            'target_route_data': {
                'pattern': 'target_route_data.csv',
                'output': 'target_route_data_cleaned.csv',
                'description': "目標路段所有數據"
            },
            'target_route_peak': {
                'pattern': 'target_route_peak.csv',
                'output': 'target_route_peak_cleaned.csv',
                'description': "目標路段尖峰數據"
            },
            'target_route_offpeak': {
                'pattern': 'target_route_offpeak.csv',
                'output': 'target_route_offpeak_cleaned.csv',
                'description': "目標路段離峰數據"
            }
        }
        
        print("🧹 VD目標路段清理器適配版初始化")
        print(f"   📁 輸入目錄: {self.processed_base_folder}")
        print(f"   📁 輸出目錄: {self.cleaned_base_folder}")
        print(f"   💾 目標記憶體: {target_memory_percent}%")
        print(f"   🎯 目標檔案: {len(self.target_file_mappings)} 種")
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "清理操作"):
        """記憶體監控上下文"""
        start_memory = psutil.virtual_memory().percent
        
        try:
            yield
        finally:
            end_memory = psutil.virtual_memory().percent
            
            # 智能垃圾回收
            if end_memory > self.target_memory_percent:
                gc.collect()
                final_memory = psutil.virtual_memory().percent
                if abs(final_memory - start_memory) > 5:  # 記憶體變化超過5%才顯示
                    print(f"   🧹 {operation_name}: {start_memory:.1f}% → {final_memory:.1f}%")
    
    def detect_available_dates(self) -> List[str]:
        """檢測可用的日期資料夾（適配強化版）"""
        print("🔍 檢測可用日期資料夾...")
        
        available_dates = []
        
        if not self.processed_base_folder.exists():
            print(f"   ❌ 處理資料夾不存在: {self.processed_base_folder}")
            return available_dates
        
        # 掃描日期資料夾
        for item in self.processed_base_folder.iterdir():
            if item.is_dir() and item.name.count('-') == 2:  # YYYY-MM-DD 格式
                date_str = item.name
                
                # 檢查是否有目標路段檔案（適配強化版輸出）
                has_target_files = False
                target_files_found = []
                
                for name, file_info in self.target_file_mappings.items():
                    target_file = item / file_info['pattern']
                    if target_file.exists():
                        has_target_files = True
                        target_files_found.append(file_info['pattern'])
                
                if has_target_files:
                    available_dates.append(date_str)
                    print(f"   ✅ {date_str}: {len(target_files_found)} 個目標檔案")
                else:
                    print(f"   ⚠️ {date_str}: 無目標路段檔案")
        
        print(f"📊 找到 {len(available_dates)} 個可清理日期")
        return sorted(available_dates)
    
    def clean_date_folder(self, date_str: str, method: str = 'mark_nan') -> Dict[str, Any]:
        """清理單一日期資料夾（適配強化版）"""
        print(f"📅 清理 {date_str}...")
        
        with self.memory_monitor(f"日期 {date_str} 清理"):
            date_input_folder = self.processed_base_folder / date_str
            date_output_folder = self.cleaned_base_folder / date_str
            date_output_folder.mkdir(parents=True, exist_ok=True)
            
            cleaning_results = []
            total_files = 0
            successful_files = 0
            
            # 處理目標路段檔案
            for name, file_info in self.target_file_mappings.items():
                input_file = date_input_folder / file_info['pattern']
                
                if input_file.exists():
                    total_files += 1
                    output_file = date_output_folder / file_info['output']
                    
                    result = self._clean_single_file(
                        input_file, output_file, 
                        file_info['description'], method
                    )
                    
                    cleaning_results.append(result)
                    if result['success']:
                        successful_files += 1
                        print(f"      ✅ {file_info['description']}")
                    else:
                        print(f"      ❌ {file_info['description']}: {result.get('error', '未知錯誤')}")
            
            # 複製摘要檔案（如果存在）
            summary_source = date_input_folder / "target_route_summary.json"
            if summary_source.exists():
                summary_dest = date_output_folder / "target_route_summary.json"
                import shutil
                shutil.copy2(summary_source, summary_dest)
                print(f"      📋 複製摘要檔案")
            
            # 生成日期清理報告
            date_report = {
                'date': date_str,
                'total_files': total_files,
                'successful_files': successful_files,
                'success_rate': f"{(successful_files/total_files*100):.1f}%" if total_files > 0 else "0%",
                'cleaning_results': cleaning_results,
                'output_folder': str(date_output_folder)
            }
            
            # 保存報告
            report_path = date_output_folder / "cleaning_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(date_report, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"   ✅ {date_str}: {successful_files}/{total_files} 檔案成功")
            return date_report
    
    def _clean_single_file(self, input_path: Path, output_path: Path, 
                          description: str, method: str) -> Dict[str, Any]:
        """清理單一檔案（記憶體優化版）"""
        try:
            with self.memory_monitor(f"清理 {description}"):
                # 檢查檔案大小，決定處理策略
                file_size_mb = input_path.stat().st_size / (1024 * 1024)
                
                if file_size_mb > 50:  # 大檔案分批處理（調低閾值）
                    return self._clean_large_file(input_path, output_path, description, method)
                else:
                    return self._clean_small_file(input_path, output_path, description, method)
                    
        except Exception as e:
            return {
                'file': description,
                'success': False,
                'error': str(e),
                'input_path': str(input_path),
                'output_path': str(output_path)
            }
    
    def _clean_small_file(self, input_path: Path, output_path: Path, 
                         description: str, method: str) -> Dict[str, Any]:
        """清理小檔案"""
        # 載入數據
        df = pd.read_csv(input_path, low_memory=True)
        original_count = len(df)
        
        # 優化數據類型
        df = self._optimize_dtypes(df)
        
        # 識別並清理異常值
        invalid_count = self._count_invalid_values(df)
        df_cleaned = self._apply_cleaning_method(df, method)
        cleaned_count = len(df_cleaned)
        
        # 保存清理後數據
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return {
            'file': description,
            'success': True,
            'original_records': original_count,
            'cleaned_records': cleaned_count,
            'removed_records': original_count - cleaned_count,
            'invalid_values': invalid_count,
            'input_path': str(input_path),
            'output_path': str(output_path),
            'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 1)
        }
    
    def _clean_large_file(self, input_path: Path, output_path: Path, 
                         description: str, method: str) -> Dict[str, Any]:
        """清理大檔案（分批處理）"""
        chunk_size = 30000  # 每批處理3萬記錄（降低批次大小）
        total_original = 0
        total_cleaned = 0
        total_invalid = 0
        
        # 分批讀取和處理
        first_chunk = True
        
        for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=True):
            total_original += len(chunk)
            
            # 優化數據類型
            chunk = self._optimize_dtypes(chunk)
            
            # 清理異常值
            total_invalid += self._count_invalid_values(chunk)
            chunk_cleaned = self._apply_cleaning_method(chunk, method)
            total_cleaned += len(chunk_cleaned)
            
            # 保存（第一批包含標頭）
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            chunk_cleaned.to_csv(output_path, mode=mode, header=header, 
                               index=False, encoding='utf-8-sig')
            
            first_chunk = False
            
            # 清理記憶體
            del chunk, chunk_cleaned
            gc.collect()
        
        return {
            'file': description,
            'success': True,
            'original_records': total_original,
            'cleaned_records': total_cleaned,
            'removed_records': total_original - total_cleaned,
            'invalid_values': total_invalid,
            'input_path': str(input_path),
            'output_path': str(output_path),
            'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 1),
            'processing_method': 'chunked'
        }
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """快速優化數據類型"""
        if df.empty:
            return df
        
        # 時間類型
        if 'update_time' in df.columns:
            df['update_time'] = pd.to_datetime(df['update_time'], errors='coerce')
        
        # 數值類型
        numeric_cols = [col for col in self.valid_ranges.keys() if col in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        
        # 類別類型
        if 'vd_id' in df.columns:
            df['vd_id'] = df['vd_id'].astype('category')
        
        if 'time_category' in df.columns:
            df['time_category'] = df['time_category'].astype('category')
        
        return df
    
    def _count_invalid_values(self, df: pd.DataFrame) -> int:
        """計算異常值數量"""
        invalid_count = 0
        
        numeric_cols = [col for col in self.valid_ranges.keys() if col in df.columns]
        
        for col in numeric_cols:
            # 異常標記
            invalid_count += df[col].isin(self.invalid_markers).sum()
            
            # 超出範圍
            min_val, max_val = self.valid_ranges[col]
            invalid_count += ((df[col] < min_val) | (df[col] > max_val)).sum()
            
            # NaN值
            invalid_count += df[col].isna().sum()
        
        return int(invalid_count)
    
    def _apply_cleaning_method(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """應用清理方法"""
        df_cleaned = df.copy()
        numeric_cols = [col for col in self.valid_ranges.keys() if col in df_cleaned.columns]
        
        if method == 'mark_nan':
            for col in numeric_cols:
                # 替換異常標記
                df_cleaned.loc[df_cleaned[col].isin(self.invalid_markers), col] = np.nan
                
                # 替換超出範圍的值
                min_val, max_val = self.valid_ranges[col]
                out_of_range = (df_cleaned[col] < min_val) | (df_cleaned[col] > max_val)
                df_cleaned.loc[out_of_range, col] = np.nan
                
        elif method == 'remove_rows':
            for col in numeric_cols:
                # 移除異常行
                df_cleaned = df_cleaned[~df_cleaned[col].isin(self.invalid_markers)]
                
                # 移除超出範圍的行
                min_val, max_val = self.valid_ranges[col]
                df_cleaned = df_cleaned[
                    (df_cleaned[col] >= min_val) & (df_cleaned[col] <= max_val)
                ]
                
                # 移除NaN行
                df_cleaned = df_cleaned.dropna(subset=[col])
        
        return df_cleaned
    
    def clean_all_dates(self, method: str = 'mark_nan') -> Dict[str, Any]:
        """批次清理所有日期（記憶體優化版）"""
        print("🚀 開始批次清理所有目標路段數據")
        print("=" * 50)
        
        start_time = datetime.now()
        
        with self.memory_monitor("批次清理"):
            available_dates = self.detect_available_dates()
            
            if not available_dates:
                return {"success": False, "error": "無可清理日期"}
            
            print(f"🧹 清理 {len(available_dates)} 個日期，方法: {method}")
            
            # 批次清理
            date_results = []
            successful_dates = 0
            total_files = 0
            successful_files = 0
            
            for i, date_str in enumerate(available_dates, 1):
                try:
                    print(f"   📅 [{i}/{len(available_dates)}] {date_str}")
                    date_result = self.clean_date_folder(date_str, method)
                    date_results.append(date_result)
                    
                    if date_result['successful_files'] > 0:
                        successful_dates += 1
                    
                    total_files += date_result['total_files']
                    successful_files += date_result['successful_files']
                    
                    # 定期清理記憶體
                    if i % 3 == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"   ❌ {date_str}: 清理失敗 - {e}")
                    date_results.append({
                        'date': date_str,
                        'success': False,
                        'error': str(e)
                    })
            
            # 生成總報告
            duration = (datetime.now() - start_time).total_seconds()
            
            batch_report = {
                'metadata': {
                    'start_time': start_time.isoformat(),
                    'duration_seconds': round(duration, 2),
                    'method': method,
                    'target_files': list(self.target_file_mappings.keys())
                },
                'summary': {
                    'total_dates': len(available_dates),
                    'successful_dates': successful_dates,
                    'total_files': total_files,
                    'successful_files': successful_files,
                    'success_rate': f"{(successful_dates/len(available_dates)*100):.1f}%"
                },
                'date_results': date_results,
                'output_folder': str(self.cleaned_base_folder)
            }
            
            # 保存批次報告
            report_path = self.cleaned_base_folder / "batch_cleaning_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(batch_report, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n🏁 批次清理完成")
            print(f"   ⏱️ 耗時: {duration:.1f} 秒")
            print(f"   📅 成功日期: {successful_dates}/{len(available_dates)}")
            print(f"   📄 成功檔案: {successful_files}/{total_files}")
            print(f"   📁 報告: {report_path}")
            
            return batch_report
    
    def get_cleaned_summary(self) -> Dict[str, Any]:
        """獲取清理摘要"""
        print("📊 檢查清理結果...")
        
        summary = {
            'cleaned_dates': 0,
            'total_files': 0,
            'total_records': 0,
            'total_size_mb': 0,
            'date_details': {}
        }
        
        if not self.cleaned_base_folder.exists():
            return summary
        
        # 掃描清理後日期
        for date_folder in self.cleaned_base_folder.iterdir():
            if date_folder.is_dir() and date_folder.name.count('-') == 2:
                date_str = date_folder.name
                
                # 統計該日期的檔案
                csv_files = list(date_folder.glob("*_cleaned.csv"))
                if csv_files:
                    summary['cleaned_dates'] += 1
                    summary['total_files'] += len(csv_files)
                    
                    date_size = 0
                    date_records = 0
                    
                    for csv_file in csv_files:
                        try:
                            file_size = csv_file.stat().st_size / (1024 * 1024)
                            date_size += file_size
                            
                            # 估算記錄數
                            estimated_records = int(file_size * 1000)
                            date_records += estimated_records
                            
                        except:
                            continue
                    
                    summary['total_size_mb'] += date_size
                    summary['total_records'] += date_records
                    
                    summary['date_details'][date_str] = {
                        'files': len(csv_files),
                        'size_mb': round(date_size, 1),
                        'estimated_records': date_records
                    }
                    
                    print(f"   ✅ {date_str}: {len(csv_files)} 檔案 ({date_size:.1f}MB)")
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 1)
        
        print(f"\n📈 清理摘要:")
        print(f"   📅 已清理日期: {summary['cleaned_dates']}")
        print(f"   📄 總檔案數: {summary['total_files']}")
        print(f"   📊 總記錄數: ~{summary['total_records']:,}")
        print(f"   💾 總大小: {summary['total_size_mb']:.1f}MB")
        
        return summary
    
    def load_cleaned_date(self, date_str: str) -> Dict[str, pd.DataFrame]:
        """載入特定日期的清理數據"""
        print(f"📅 載入 {date_str} 清理數據...")
        
        date_folder = self.cleaned_base_folder / date_str
        if not date_folder.exists():
            print(f"   ❌ 日期資料夾不存在: {date_str}")
            return {}
        
        cleaned_data = {}
        
        # 載入目標檔案
        for name, file_info in self.target_file_mappings.items():
            csv_file = date_folder / file_info['output']
            
            if csv_file.exists():
                try:
                    with self.memory_monitor(f"載入 {file_info['description']}"):
                        df = pd.read_csv(csv_file, low_memory=True)
                        df = self._optimize_dtypes(df)
                        cleaned_data[name] = df
                        print(f"   ✅ {file_info['description']}: {len(df):,} 筆")
                        
                except Exception as e:
                    print(f"   ❌ {file_info['description']}: 載入失敗 - {e}")
                    cleaned_data[name] = pd.DataFrame()
            else:
                print(f"   ⚠️ {file_info['description']}: 檔案不存在")
                cleaned_data[name] = pd.DataFrame()
        
        return cleaned_data


# ============================================================
# 便利函數
# ============================================================

def clean_all_target_data(base_folder: str = "data", method: str = 'mark_nan') -> Dict[str, Any]:
    """一鍵清理所有目標路段數據"""
    cleaner = VDTargetRouteCleaner(base_folder)
    return cleaner.clean_all_dates(method)


def get_cleaning_summary(base_folder: str = "data") -> Dict[str, Any]:
    """獲取清理摘要"""
    cleaner = VDTargetRouteCleaner(base_folder)
    return cleaner.get_cleaned_summary()


def load_cleaned_data(base_folder: str = "data", date_str: str = None) -> Dict[str, pd.DataFrame]:
    """載入清理數據"""
    cleaner = VDTargetRouteCleaner(base_folder)
    
    if date_str:
        return cleaner.load_cleaned_date(date_str)
    else:
        # 載入最新日期
        available_dates = cleaner.detect_available_dates()
        if available_dates:
            return cleaner.load_cleaned_date(available_dates[-1])
        return {}


# 保持向後相容性
clean_all_data = clean_all_target_data


if __name__ == "__main__":
    print("🧹 VD目標路段清理器 - 適配強化版載入器")
    print("=" * 60)
    print("🎯 專門處理目標路段檔案:")
    print("   • target_route_data.csv → target_route_data_cleaned.csv")
    print("   • target_route_peak.csv → target_route_peak_cleaned.csv")
    print("   • target_route_offpeak.csv → target_route_offpeak_cleaned.csv")
    print("=" * 60)
    
    cleaner = VDTargetRouteCleaner()
    
    # 檢測可用日期
    available_dates = cleaner.detect_available_dates()
    
    if available_dates:
        print(f"\n📅 發現 {len(available_dates)} 個可清理日期")
        for date_str in available_dates:
            print(f"   • {date_str}")
        
        response = input(f"\n開始批次清理目標路段數據？(y/N): ")
        
        if response.lower() in ['y', 'yes']:
            # 執行清理
            report = cleaner.clean_all_dates()
            
            if report['summary']['successful_dates'] > 0:
                print(f"\n🎉 清理完成！")
                
                # 顯示摘要
                summary = cleaner.get_cleaned_summary()
                print(f"\n📊 清理結果:")
                print(f"   📅 已清理日期: {summary['cleaned_dates']}")
                print(f"   📄 總檔案數: {summary['total_files']}")
                print(f"   📊 總記錄數: ~{summary['total_records']:,}")
                print(f"   💾 總大小: {summary['total_size_mb']:.1f}MB")
                
                print(f"\n📁 清理後結構:")
                print(f"   📂 data/cleaned/")
                for date_str in summary['date_details']:
                    details = summary['date_details'][date_str]
                    print(f"      ├── {date_str}/ ({details['files']} 檔案)")
                    print(f"      │   ├── target_route_data_cleaned.csv")
                    print(f"      │   ├── target_route_peak_cleaned.csv")
                    print(f"      │   └── target_route_offpeak_cleaned.csv")
                
                print(f"\n🎯 建議下一步:")
                print("   1. 檢查清理後數據: python test_cleaner.py")
                print("   2. 開始AI模型開發: python src/predictor.py")
                print("   3. 使用清理數據進行交通預測分析")
            else:
                print("❌ 清理失敗")
    else:
        print("\n💡 請先執行數據載入:")
        print("   python -c \"from src.data_loader import auto_process_data; auto_process_data()\"")
        print("   或")
        print("   python test_loader.py")