# src/data_cleaner.py - 日期資料夾組織版

"""
VD數據清理器 - 日期資料夾組織版
==============================

新增功能：
1. 支援按日期組織的處理資料夾：data/processed/2025-06-27/
2. 輸出到按日期組織的清理資料夾：data/cleaned/2025-06-27/
3. 自動偵測多日期資料夾並批次清理
4. 保持原有的批次清理所有分類檔案功能

配合新版 data_loader.py，支援批次清理所有分類檔案：
- 自動偵測 data/processed/YYYY-MM-DD/ 中的檔案
- 批次清理並保存到 data/cleaned/YYYY-MM-DD/
- 保持原有檔案結構和命名
- 生成完整的清理報告
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

class VDBatchDataCleaner:
    """
    VD批次數據清理器 - 日期資料夾組織版
    
    支援按日期組織的批次清理所有分類檔案
    """
    
    def __init__(self, base_folder: str = "data"):
        """
        初始化批次清理器
        
        Args:
            base_folder: 基礎資料夾路徑
        """
        self.base_folder = Path(base_folder)
        self.processed_base_folder = self.base_folder / "processed"
        self.cleaned_base_folder = self.base_folder / "cleaned"
        
        # 確保 cleaned 基礎資料夾存在
        self.cleaned_base_folder.mkdir(parents=True, exist_ok=True)
        
        # 定義異常值標記
        self.invalid_markers = [-99, -1, 999, 9999]
        
        # 定義合理的數值範圍
        self.valid_ranges = {
            'speed': (0, 150),        # 速度：0-150 km/h
            'occupancy': (0, 100),    # 佔有率：0-100%
            'volume_total': (0, 100), # 流量：0-100輛
            'volume_small': (0, 100),
            'volume_large': (0, 50),
            'volume_truck': (0, 50),
            'speed_small': (0, 150),
            'speed_large': (0, 150),
            'speed_truck': (0, 150)
        }
        
        # 定義要清理的檔案映射（相對於日期資料夾）
        self.file_mappings = {
            'all': {
                'input_csv': "vd_data_all.csv",
                'input_json': "vd_data_all_summary.json",
                'output_csv': "vd_data_all_cleaned.csv",
                'output_json': "vd_data_all_cleaned_summary.json",
                'description': "全部VD資料"
            },
            'peak': {
                'input_csv': "vd_data_peak.csv",
                'input_json': "vd_data_peak_summary.json",
                'output_csv': "vd_data_peak_cleaned.csv",
                'output_json': "vd_data_peak_cleaned_summary.json",
                'description': "所有尖峰時段數據"
            },
            'offpeak': {
                'input_csv': "vd_data_offpeak.csv",
                'input_json': "vd_data_offpeak_summary.json",
                'output_csv': "vd_data_offpeak_cleaned.csv",
                'output_json': "vd_data_offpeak_cleaned_summary.json",
                'description': "所有離峰時段數據"
            },
            'target_route': {
                'input_csv': "target_route_data.csv",
                'input_json': "target_route_data_summary.json",
                'output_csv': "target_route_data_cleaned.csv",
                'output_json': "target_route_data_cleaned_summary.json",
                'description': "目標路段數據"
            },
            'target_route_peak': {
                'input_csv': "target_route_peak.csv",
                'input_json': "target_route_peak_summary.json",
                'output_csv': "target_route_peak_cleaned.csv",
                'output_json': "target_route_peak_cleaned_summary.json",
                'description': "目標路段尖峰"
            },
            'target_route_offpeak': {
                'input_csv': "target_route_offpeak.csv",
                'input_json': "target_route_offpeak_summary.json",
                'output_csv': "target_route_offpeak_cleaned.csv",
                'output_json': "target_route_offpeak_cleaned_summary.json",
                'description': "目標路段離峰"
            }
        }
        
        print("🧹 VD批次數據清理器日期組織版初始化完成")
        print(f"   📁 輸入基礎資料夾: {self.processed_base_folder}")
        print(f"   📁 輸出基礎資料夾: {self.cleaned_base_folder}")
        print(f"   🗂️ 日期組織: YYYY-MM-DD/")
        print(f"   📊 預計清理檔案: {len(self.file_mappings)} 種")
    
    def detect_available_date_folders(self) -> Dict[str, Dict[str, Any]]:
        """檢測可用的日期資料夾"""
        print("🔍 檢測可用的日期資料夾...")
        
        available_date_folders = {}
        
        if not self.processed_base_folder.exists():
            print(f"   ❌ 處理資料夾不存在: {self.processed_base_folder}")
            return available_date_folders
        
        # 掃描日期資料夾
        for date_folder in self.processed_base_folder.iterdir():
            if date_folder.is_dir() and date_folder.name.count('-') == 2:  # YYYY-MM-DD 格式
                date_str = date_folder.name
                
                # 檢查該日期資料夾中的檔案
                available_files = self._detect_files_in_date_folder(date_folder, date_str)
                
                if available_files:
                    available_date_folders[date_str] = {
                        'folder_path': date_folder,
                        'available_files': available_files,
                        'file_count': len(available_files)
                    }
                    
                    print(f"   ✅ {date_str}: 找到 {len(available_files)} 個可清理檔案")
                else:
                    print(f"   ⚠️ {date_str}: 沒有可清理檔案")
        
        print(f"\n📊 檢測結果: 找到 {len(available_date_folders)} 個日期資料夾")
        return available_date_folders
    
    def _detect_files_in_date_folder(self, date_folder: Path, date_str: str) -> Dict[str, Dict[str, Any]]:
        """檢測特定日期資料夾中的檔案"""
        available_files = {}
        
        for name, mapping in self.file_mappings.items():
            input_csv = date_folder / mapping['input_csv']
            description = mapping['description']
            
            if input_csv.exists():
                try:
                    # 快速檢查檔案
                    df = pd.read_csv(input_csv, nrows=5)
                    file_size = input_csv.stat().st_size / 1024 / 1024  # MB
                    
                    # 估算總記錄數
                    estimated_records = int(file_size * 1000)
                    
                    available_files[name] = {
                        'input_path': input_csv,
                        'description': description,
                        'file_size_mb': round(file_size, 1),
                        'estimated_records': estimated_records,
                        'columns': len(df.columns),
                        'status': 'ready'
                    }
                    
                except Exception as e:
                    available_files[name] = {
                        'input_path': input_csv,
                        'description': description,
                        'status': 'error',
                        'error': str(e)
                    }
        
        return available_files
    
    def clean_single_date_folder(self, date_str: str, date_info: Dict[str, Any], method: str = 'mark_nan') -> Dict[str, Any]:
        """清理單一日期資料夾"""
        
        print(f"📅 清理 {date_str} 資料夾...")
        
        date_folder = date_info['folder_path']
        available_files = date_info['available_files']
        
        # 建立對應的清理日期資料夾
        cleaned_date_folder = self.cleaned_base_folder / date_str
        cleaned_date_folder.mkdir(parents=True, exist_ok=True)
        
        cleaning_results = []
        successful_cleanings = 0
        failed_cleanings = 0
        
        # 清理該日期資料夾中的所有檔案
        for name, file_info in available_files.items():
            if file_info.get('status') == 'ready':
                mapping = self.file_mappings[name]
                
                # 建立完整的檔案路徑
                full_mapping = {
                    'input_csv': date_folder / mapping['input_csv'],
                    'input_json': date_folder / mapping['input_json'],
                    'output_csv': cleaned_date_folder / mapping['output_csv'],
                    'output_json': cleaned_date_folder / mapping['output_json'],
                    'description': f"{date_str} {mapping['description']}"
                }
                
                result = self.clean_single_file(name, full_mapping, method)
                cleaning_results.append(result)
                
                if result['success']:
                    successful_cleanings += 1
                else:
                    failed_cleanings += 1
            else:
                failed_cleanings += 1
                cleaning_results.append({
                    'name': name,
                    'description': file_info['description'],
                    'success': False,
                    'error': file_info.get('error', '檔案不可用')
                })
        
        # 生成該日期的清理報告
        date_report = {
            'date': date_str,
            'cleaned_folder': str(cleaned_date_folder),
            'total_files': len(available_files),
            'successful_cleanings': successful_cleanings,
            'failed_cleanings': failed_cleanings,
            'success_rate': f"{(successful_cleanings / len(available_files) * 100):.1f}%",
            'cleaning_results': cleaning_results
        }
        
        # 保存該日期的清理報告
        date_report_path = cleaned_date_folder / "date_cleaning_report.json"
        with open(date_report_path, 'w', encoding='utf-8') as f:
            json.dump(date_report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"   ✅ {date_str}: 成功清理 {successful_cleanings}/{len(available_files)} 個檔案")
        
        return date_report
    
    def clean_single_file(self, name: str, mapping: Dict[str, Any], method: str = 'mark_nan') -> Dict[str, Any]:
        """清理單一檔案"""
        
        input_csv = mapping['input_csv']
        output_csv = mapping['output_csv']
        output_json = mapping['output_json']
        description = mapping['description']
        
        try:
            # 載入數據
            df_original = pd.read_csv(input_csv)
            
            # 數據類型最佳化
            df_original = self._optimize_data_types(df_original)
            
            # 識別無效數據
            invalid_stats = self._identify_invalid_data_quick(df_original)
            
            # 清理數據
            df_cleaned = self._clean_invalid_values(df_original, method)
            
            # 保存清理後數據
            df_cleaned.to_csv(output_csv, index=False, encoding='utf-8-sig')
            
            # 生成摘要
            summary = self._generate_cleaned_summary(df_cleaned, name, description, invalid_stats)
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            # 計算清理效果
            cleaning_result = {
                'name': name,
                'description': description,
                'success': True,
                'original_records': len(df_original),
                'cleaned_records': len(df_cleaned),
                'records_removed': len(df_original) - len(df_cleaned),
                'invalid_values_found': invalid_stats['total_invalid'],
                'invalid_percentage': invalid_stats['invalid_percentage'],
                'output_files': {
                    'csv': str(output_csv),
                    'json': str(output_json)
                }
            }
            
            return cleaning_result
            
        except Exception as e:
            return {
                'name': name,
                'description': description,
                'success': False,
                'error': str(e)
            }
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """快速最佳化數據類型"""
        if df.empty:
            return df
        
        # 時間類型
        time_columns = ['date', 'update_time']
        for col in time_columns:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 數值類型
        numeric_columns = [col for col in self.valid_ranges.keys() if col in df.columns]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _identify_invalid_data_quick(self, df: pd.DataFrame) -> Dict[str, Any]:
        """快速識別無效數據"""
        
        numeric_columns = [col for col in self.valid_ranges.keys() if col in df.columns]
        
        total_invalid = 0
        invalid_by_column = {}
        
        for col in numeric_columns:
            # 異常標記
            invalid_markers_count = df[col].isin(self.invalid_markers).sum()
            
            # 超出範圍
            min_val, max_val = self.valid_ranges[col]
            out_of_range_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
            
            col_invalid = invalid_markers_count + out_of_range_count
            total_invalid += col_invalid
            
            if col_invalid > 0:
                invalid_by_column[col] = {
                    'invalid_markers': int(invalid_markers_count),
                    'out_of_range': int(out_of_range_count),
                    'total': int(col_invalid)
                }
        
        return {
            'total_invalid': int(total_invalid),
            'invalid_percentage': round(total_invalid / (len(df) * len(numeric_columns)) * 100, 2),
            'columns_with_issues': len(invalid_by_column),
            'by_column': invalid_by_column
        }
    
    def _clean_invalid_values(self, df: pd.DataFrame, method: str = 'mark_nan') -> pd.DataFrame:
        """快速清理無效值"""
        
        df_cleaned = df.copy()
        numeric_columns = [col for col in self.valid_ranges.keys() if col in df_cleaned.columns]
        
        if method == 'mark_nan':
            for col in numeric_columns:
                # 替換異常標記
                mask = df_cleaned[col].isin(self.invalid_markers)
                df_cleaned.loc[mask, col] = np.nan
                
                # 替換超出範圍的值
                if col in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[col]
                    out_of_range_mask = (df_cleaned[col] < min_val) | (df_cleaned[col] > max_val)
                    df_cleaned.loc[out_of_range_mask, col] = np.nan
        
        elif method == 'remove_rows':
            for col in numeric_columns:
                # 刪除異常標記的行
                df_cleaned = df_cleaned[~df_cleaned[col].isin(self.invalid_markers)]
                
                # 刪除超出範圍的行
                if col in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[col]
                    df_cleaned = df_cleaned[
                        (df_cleaned[col] >= min_val) & (df_cleaned[col] <= max_val)
                    ]
        
        return df_cleaned
    
    def _generate_cleaned_summary(self, df: pd.DataFrame, category: str, description: str, 
                                 invalid_stats: Dict[str, Any]) -> Dict[str, Any]:
        """生成清理後摘要"""
        
        if df.empty:
            return {"category": category, "description": description, "error": "清理後無數據"}
        
        numeric_columns = [col for col in self.valid_ranges.keys() if col in df.columns]
        
        # 計算數據完整度
        completeness = {}
        for col in numeric_columns:
            if col in df.columns:
                total_values = len(df)
                missing_values = df[col].isnull().sum()
                completeness[col] = {
                    "完整度": f"{((total_values - missing_values) / total_values * 100):.1f}%",
                    "缺失值": int(missing_values),
                    "有效值": int(total_values - missing_values)
                }
        
        summary = {
            "category": category,
            "description": description,
            "清理時間": datetime.now().isoformat(),
            "基本資訊": {
                "記錄數": len(df),
                "欄位數": len(df.columns),
                "VD設備數": int(df['vd_id'].nunique()) if 'vd_id' in df.columns else 0
            },
            "清理效果": {
                "原始無效值": invalid_stats['total_invalid'],
                "無效值比例": f"{invalid_stats['invalid_percentage']:.2f}%",
                "問題欄位數": invalid_stats['columns_with_issues']
            },
            "數據完整度": completeness,
            "交通統計": {}
        }
        
        # 交通統計
        if 'speed' in df.columns:
            summary["交通統計"]["平均速度"] = round(df['speed'].mean(), 1)
            summary["交通統計"]["最高速度"] = int(df['speed'].max())
        
        if 'occupancy' in df.columns:
            summary["交通統計"]["平均佔有率"] = round(df['occupancy'].mean(), 1)
            summary["交通統計"]["最高佔有率"] = int(df['occupancy'].max())
        
        if 'volume_total' in df.columns:
            summary["交通統計"]["平均流量"] = round(df['volume_total'].mean(), 1)
            summary["交通統計"]["最高流量"] = int(df['volume_total'].max())
        
        # 時間範圍
        if 'date' in df.columns:
            summary["時間範圍"] = {
                "開始": str(df['date'].min())[:10],
                "結束": str(df['date'].max())[:10],
                "天數": df['date'].nunique()
            }
        
        return summary
    
    def clean_all_date_folders(self, method: str = 'mark_nan') -> Dict[str, Any]:
        """批次清理所有日期資料夾"""
        
        print("🚀 開始批次清理所有日期資料夾")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # 檢測可用日期資料夾
        available_date_folders = self.detect_available_date_folders()
        
        if not available_date_folders:
            print("❌ 沒有找到可清理的日期資料夾")
            return {"success": False, "error": "無可清理日期資料夾"}
        
        print(f"\n🧹 開始清理 {len(available_date_folders)} 個日期資料夾...")
        print(f"   清理方法: {method}")
        
        # 批次清理各日期資料夾
        date_cleaning_results = []
        successful_dates = 0
        failed_dates = 0
        total_successful_files = 0
        total_failed_files = 0
        
        for date_str, date_info in sorted(available_date_folders.items()):
            try:
                date_result = self.clean_single_date_folder(date_str, date_info, method)
                date_cleaning_results.append(date_result)
                
                if date_result['successful_cleanings'] > 0:
                    successful_dates += 1
                    total_successful_files += date_result['successful_cleanings']
                else:
                    failed_dates += 1
                
                total_failed_files += date_result['failed_cleanings']
                
            except Exception as e:
                print(f"   ❌ {date_str}: 清理失敗 - {e}")
                failed_dates += 1
                date_cleaning_results.append({
                    'date': date_str,
                    'success': False,
                    'error': str(e)
                })
        
        # 生成批次清理報告
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        batch_report = {
            '批次清理元數據': {
                '清理時間': start_time.isoformat(),
                '完成時間': end_time.isoformat(),
                '耗時秒數': round(duration, 2),
                '清理方法': method
            },
            '清理統計': {
                '總日期資料夾數': len(available_date_folders),
                '成功清理日期數': successful_dates,
                '失敗清理日期數': failed_dates,
                '總成功檔案數': total_successful_files,
                '總失敗檔案數': total_failed_files,
                '日期成功率': f"{(successful_dates / len(available_date_folders) * 100):.1f}%"
            },
            '各日期清理結果': date_cleaning_results,
            '輸出基礎目錄': str(self.cleaned_base_folder)
        }
        
        # 保存批次報告
        report_path = self.cleaned_base_folder / "batch_date_cleaning_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, ensure_ascii=False, indent=2, default=str)
        
        # 顯示結果
        print(f"\n🏁 批次清理完成")
        print("=" * 60)
        print(f"📊 清理統計:")
        print(f"   ⏱️ 耗時: {duration:.1f} 秒")
        print(f"   📅 成功日期: {successful_dates}/{len(available_date_folders)}")
        print(f"   📄 成功檔案: {total_successful_files}")
        print(f"   ❌ 失敗檔案: {total_failed_files}")
        print(f"   📈 日期成功率: {(successful_dates / len(available_date_folders) * 100):.1f}%")
        
        print(f"\n📁 清理後結構:")
        for date_str in sorted(available_date_folders.keys()):
            cleaned_date_folder = self.cleaned_base_folder / date_str
            if cleaned_date_folder.exists():
                cleaned_files = list(cleaned_date_folder.glob("*.csv"))
                print(f"   📂 data/cleaned/{date_str}/ ({len(cleaned_files)} 個CSV檔案)")
        
        print(f"\n📄 報告檔案: {report_path}")
        
        return batch_report
    
    def get_cleaned_files_summary(self) -> Dict[str, Any]:
        """獲取清理後檔案摘要"""
        
        print("📊 檢查清理後檔案狀況...")
        
        summary = {
            '清理檔案統計': {
                '清理日期數': 0,
                '存在檔案數': 0,
                '總記錄數': 0,
                '檔案大小_MB': 0
            },
            '各日期詳情': {}
        }
        
        if not self.cleaned_base_folder.exists():
            print(f"   ⚠️ 清理資料夾不存在: {self.cleaned_base_folder}")
            return summary
        
        # 掃描清理後的日期資料夾
        for cleaned_date_folder in self.cleaned_base_folder.iterdir():
            if cleaned_date_folder.is_dir() and cleaned_date_folder.name.count('-') == 2:
                date_str = cleaned_date_folder.name
                
                # 統計該日期資料夾中的檔案
                date_summary = self._get_date_folder_summary(cleaned_date_folder, date_str)
                
                if date_summary['檔案數'] > 0:
                    summary['各日期詳情'][date_str] = date_summary
                    summary['清理檔案統計']['清理日期數'] += 1
                    summary['清理檔案統計']['存在檔案數'] += date_summary['檔案數']
                    summary['清理檔案統計']['總記錄數'] += date_summary['總記錄數']
                    summary['清理檔案統計']['檔案大小_MB'] += date_summary['總檔案大小_MB']
                    
                    print(f"   ✅ {date_str}: {date_summary['檔案數']} 檔案 ({date_summary['總檔案大小_MB']:.1f}MB)")
        
        # 四捨五入統計
        summary['清理檔案統計']['檔案大小_MB'] = round(summary['清理檔案統計']['檔案大小_MB'], 1)
        
        print(f"\n📈 總計:")
        print(f"   📅 清理日期數: {summary['清理檔案統計']['清理日期數']}")
        print(f"   📄 總檔案數: {summary['清理檔案統計']['存在檔案數']}")
        print(f"   📊 總記錄數: {summary['清理檔案統計']['總記錄數']:,}")
        print(f"   💾 總檔案大小: {summary['清理檔案統計']['檔案大小_MB']:.1f} MB")
        
        return summary
    
    def _get_date_folder_summary(self, cleaned_date_folder: Path, date_str: str) -> Dict[str, Any]:
        """獲取特定日期資料夾的摘要"""
        
        date_summary = {
            '日期': date_str,
            '檔案數': 0,
            '總記錄數': 0,
            '總檔案大小_MB': 0,
            '檔案詳情': {}
        }
        
        # 檢查該日期資料夾中的清理檔案
        for name, mapping in self.file_mappings.items():
            output_csv = cleaned_date_folder / mapping['output_csv']
            output_json = cleaned_date_folder / mapping['output_json']
            description = mapping['description']
            
            if output_csv.exists():
                try:
                    df = pd.read_csv(output_csv, low_memory=False)
                    file_size_mb = output_csv.stat().st_size / 1024 / 1024
                    
                    date_summary['檔案數'] += 1
                    date_summary['總記錄數'] += len(df)
                    date_summary['總檔案大小_MB'] += file_size_mb
                    
                    # 讀取JSON摘要
                    json_summary = {}
                    if output_json.exists():
                        with open(output_json, 'r', encoding='utf-8') as f:
                            json_summary = json.load(f)
                    
                    date_summary['檔案詳情'][name] = {
                        '描述': description,
                        '記錄數': len(df),
                        '檔案大小_MB': round(file_size_mb, 1),
                        'CSV路徑': str(output_csv),
                        'JSON路徑': str(output_json),
                        '摘要': json_summary.get('交通統計', {})
                    }
                    
                except Exception as e:
                    print(f"      ❌ {date_str} {description}: 讀取失敗 - {e}")
        
        # 四捨五入
        date_summary['總檔案大小_MB'] = round(date_summary['總檔案大小_MB'], 1)
        
        return date_summary
    
    def list_available_cleaned_dates(self) -> List[str]:
        """列出可用的清理日期資料夾"""
        available_dates = []
        
        if self.cleaned_base_folder.exists():
            for cleaned_date_folder in self.cleaned_base_folder.iterdir():
                if cleaned_date_folder.is_dir() and cleaned_date_folder.name.count('-') == 2:
                    available_dates.append(cleaned_date_folder.name)
        
        return sorted(available_dates)
    
    def load_cleaned_data_by_date(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """載入特定日期的清理數據"""
        print(f"📅 載入 {target_date} 清理數據...")
        
        cleaned_date_folder = self.cleaned_base_folder / target_date
        
        if not cleaned_date_folder.exists():
            print(f"   ❌ 清理日期資料夾不存在: {target_date}")
            return {}
        
        cleaned_data = {}
        
        for name, mapping in self.file_mappings.items():
            output_csv = cleaned_date_folder / mapping['output_csv']
            description = mapping['description']
            
            if output_csv.exists():
                try:
                    df = pd.read_csv(output_csv)
                    df = self._optimize_data_types(df)
                    cleaned_data[name] = df
                    print(f"   ✅ {description}: {len(df):,} 筆記錄")
                except Exception as e:
                    print(f"   ❌ {description}: 載入失敗 - {e}")
                    cleaned_data[name] = pd.DataFrame()
            else:
                cleaned_data[name] = pd.DataFrame()
        
        return cleaned_data


# 便利函數
def clean_all_vd_data_by_date(base_folder: str = "data", method: str = 'mark_nan') -> Dict[str, Any]:
    """
    一鍵清理所有VD數據分類檔案（按日期組織）
    
    Args:
        base_folder: 基礎資料夾路徑
        method: 清理方法
    
    Returns:
        批次清理報告
    """
    cleaner = VDBatchDataCleaner(base_folder)
    return cleaner.clean_all_date_folders(method)


def get_cleaned_data_summary_by_date(base_folder: str = "data") -> Dict[str, Any]:
    """
    獲取清理後數據摘要（按日期組織）
    
    Args:
        base_folder: 基礎資料夾路徑
    
    Returns:
        清理檔案摘要
    """
    cleaner = VDBatchDataCleaner(base_folder)
    return cleaner.get_cleaned_files_summary()


def load_cleaned_data_by_date(base_folder: str = "data", target_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    載入特定日期的清理數據
    
    Args:
        base_folder: 基礎資料夾路徑
        target_date: 目標日期 (YYYY-MM-DD)
    
    Returns:
        清理數據字典
    """
    cleaner = VDBatchDataCleaner(base_folder)
    return cleaner.load_cleaned_data_by_date(target_date)


if __name__ == "__main__":
    print("🧹 VD批次數據清理器 - 日期組織版")
    print("=" * 50)
    
    cleaner = VDBatchDataCleaner()
    
    # 檢測可用日期資料夾
    available_date_folders = cleaner.detect_available_date_folders()
    
    if available_date_folders:
        print(f"\n發現 {len(available_date_folders)} 個日期資料夾可清理")
        
        # 顯示各日期詳情
        for date_str, date_info in sorted(available_date_folders.items()):
            print(f"   📅 {date_str}: {date_info['file_count']} 個檔案")
        
        response = input(f"\n開始批次清理所有日期資料夾？(y/N): ")
        
        if response.lower() in ['y', 'yes']:
            # 執行批次清理
            report = cleaner.clean_all_date_folders()
            
            if report['清理統計']['成功清理日期數'] > 0:
                print(f"\n🎉 批次清理完成！")
                print(f"✅ 成功清理 {report['清理統計']['成功清理日期數']} 個日期資料夾")
                print(f"📁 清理後檔案保存在: {cleaner.cleaned_base_folder}")
                
                # 顯示清理後摘要
                summary = cleaner.get_cleaned_files_summary()
                print(f"\n📊 清理後總計:")
                print(f"   📅 清理日期數: {summary['清理檔案統計']['清理日期數']}")
                print(f"   📄 總檔案數: {summary['清理檔案統計']['存在檔案數']}")
                print(f"   📊 總記錄數: {summary['清理檔案統計']['總記錄數']:,}")
                print(f"   💾 總檔案大小: {summary['清理檔案統計']['檔案大小_MB']:.1f} MB")
                
                print(f"\n📁 輸出結構:")
                for date_str in cleaner.list_available_cleaned_dates():
                    print(f"   📂 data/cleaned/{date_str}/")
                    print(f"      ├── vd_data_all_cleaned.csv + .json")
                    print(f"      ├── vd_data_peak_cleaned.csv + .json")
                    print(f"      ├── vd_data_offpeak_cleaned.csv + .json")
                    print(f"      ├── target_route_*_cleaned.csv + .json")
                    print(f"      └── date_cleaning_report.json")
            else:
                print("❌ 批次清理失敗")
    else:
        print("\n💡 請先執行 data_loader.py 生成按日期組織的分類檔案")
        print("預期結構:")
        print("   📂 data/processed/")
        print("      ├── 2025-06-27/")
        print("      │   ├── vd_data_all.csv")
        print("      │   ├── vd_data_peak.csv")
        print("      │   └── ...")
        print("      └── 2025-06-26/")
        print("          ├── vd_data_all.csv")
        print("          └── ...")