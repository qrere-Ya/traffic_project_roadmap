# src/data_loader.py - 簡化版（靜默記憶體優化）

"""
VD數據載入器 - 簡化版
=====================================

核心特色：
1. 靜默記憶體優化：後台自動處理，不顯示詳細信息
2. 專注Raw數據處理：主要輸出處理進度
3. 智慧Archive檢查：快速檢查，避免重複處理
4. 保留所有功能，簡化輸出

處理流程：
1. 自動檢測系統記憶體並調整策略（後台）
2. 快速檢查Archive狀態（後台）
3. 處理Raw數據並顯示主要進度
4. 按日期組織輸出
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, time
import os
import json
import shutil
import threading
import time as time_module
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

class VDDataLoader:
    """VD數據載入器 - 簡化版（靜默記憶體優化）"""
    
    def __init__(self, base_folder: str = "data", max_workers: int = None, verbose: bool = False):
        """
        初始化載入器
        
        Args:
            base_folder: 基礎資料夾路徑
            max_workers: 最大線程數
            verbose: 是否顯示詳細記憶體優化信息
        """
        self.base_folder = Path(base_folder)
        self.raw_folder = self.base_folder / "raw"
        self.processed_base_folder = self.base_folder / "processed"
        self.archive_folder = self.base_folder / "archive"
        self.verbose = verbose
        
        # 確保基礎資料夾存在
        for folder in [self.raw_folder, self.processed_base_folder, self.archive_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # 靜默記憶體優化設定
        self._init_memory_optimization_silent()
        
        # 超級優化線程數設定
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(cpu_count * 3, 20)
        else:
            self.max_workers = max_workers
        
        # XML命名空間
        self.namespace = {
            'traffic': 'http://traffic.transportdata.tw/standard/traffic/schema/'
        }
        
        # 日期資料夾映射
        self.date_folders = {}
        
        # 國道1號圓山-三重路段VD設備清單
        self.target_route_vd_ids = [
            'VD-N1-N-23.0-M-LOOP', 'VD-N1-N-23.5-M-LOOP', 
            'VD-N1-S-23.0-M-LOOP', 'VD-N1-S-23.5-M-LOOP',
            'VD-N1-N-25.0-M-LOOP', 'VD-N1-N-25.5-M-LOOP',
            'VD-N1-S-25.0-M-LOOP', 'VD-N1-S-25.5-M-LOOP',
            'VD-N1-N-27.0-M-LOOP', 'VD-N1-N-27.5-M-LOOP',
            'VD-N1-S-27.0-M-LOOP', 'VD-N1-S-27.5-M-LOOP',
            'VD-N1-N-86.120-M-LOOP', 'VD-N1-N-88.050-M-LOOP',
        ]
        
        # 內部批次大小設定（記憶體優化）
        self.internal_batch_size = self._calculate_optimal_batch_size()
        
        # 線程鎖
        self.file_lock = threading.Lock()
        
        # 簡化初始化輸出
        print(f"🏗️ VD數據載入器初始化完成")
        print(f"   📁 資料夾: {self.base_folder}")
        print(f"   🧵 處理線程: {self.max_workers}")
        print(f"   🎯 目標路段設備: {len(self.target_route_vd_ids)}個")
        if self.verbose:
            print(f"   💾 記憶體模式: {self.total_memory_gb:.1f}GB環境")
    
    def _init_memory_optimization_silent(self):
        """靜默初始化記憶體優化設定"""
        try:
            # 檢測系統記憶體
            memory_info = psutil.virtual_memory()
            self.total_memory_gb = memory_info.total / (1024**3)
            self.available_memory_gb = memory_info.available / (1024**3)
            
            # 設定記憶體使用策略（靜默）
            if self.total_memory_gb >= 16:
                self.max_memory_usage_percent = 80
                self.chunk_processing = True
                self.force_gc_frequency = 3
            elif self.total_memory_gb >= 8:
                self.max_memory_usage_percent = 70
                self.chunk_processing = True
                self.force_gc_frequency = 2
            else:
                self.max_memory_usage_percent = 60
                self.chunk_processing = True
                self.force_gc_frequency = 1
                
            self.operation_count = 0
            
        except Exception as e:
            # 靜默錯誤處理
            self.total_memory_gb = 8
            self.max_memory_usage_percent = 70
            self.chunk_processing = True
            self.force_gc_frequency = 2
            self.operation_count = 0
    
    def _calculate_optimal_batch_size(self) -> int:
        """靜默計算最佳批次大小"""
        if self.total_memory_gb >= 32:
            return 500
        elif self.total_memory_gb >= 16:
            return 300
        elif self.total_memory_gb >= 8:
            return 200
        else:
            return 100
    
    def _monitor_memory_and_gc(self, force: bool = False):
        """靜默記憶體監控和垃圾回收"""
        self.operation_count += 1
        
        try:
            if force or self.operation_count % self.force_gc_frequency == 0:
                memory_info = psutil.virtual_memory()
                memory_usage_percent = memory_info.percent
                
                if memory_usage_percent > self.max_memory_usage_percent or force:
                    gc.collect()
                    
                    # 只在verbose模式下顯示
                    if self.verbose and force:
                        memory_info_after = psutil.virtual_memory()
                        print(f"      🧹 記憶體清理: {memory_info_after.percent:.1f}%")
                
        except Exception:
            pass  # 靜默處理錯誤
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """靜默優化DataFrame記憶體使用"""
        if df.empty:
            return df
        
        try:
            # 時間類型最佳化
            time_columns = ['date', 'update_time']
            for col in time_columns:
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # 數值類型最佳化
            int_columns = [
                'lane_id', 'lane_type', 'speed', 'occupancy',
                'volume_total', 'volume_small', 'volume_large', 'volume_truck',
                'speed_small', 'speed_large', 'speed_truck'
            ]
            
            for col in int_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if df[col].notna().any():
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        if col_min >= 0 and col_max <= 255:
                            df[col] = df[col].astype('uint8')
                        elif col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype('int8')
                        elif col_min >= 0 and col_max <= 65535:
                            df[col] = df[col].astype('uint16')
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype('int16')
                        else:
                            df[col] = df[col].astype('int32')
            
            # 類別類型最佳化
            if 'vd_id' in df.columns:
                df['vd_id'] = df['vd_id'].astype('category')
            
            if 'time_category' in df.columns:
                df['time_category'] = df['time_category'].astype('category')
            
            return df
            
        except Exception:
            return df  # 靜默處理錯誤
    
    def check_archive_status_silent(self) -> Dict[str, Any]:
        """靜默Archive狀態檢查"""
        archive_status = {
            "archive_exists": self.archive_folder.exists(),
            "archived_dates": [],
            "archived_date_count": 0
        }
        
        if not self.archive_folder.exists():
            return archive_status
        
        try:
            date_folders = []
            for item in self.archive_folder.iterdir():
                if item.is_dir() and item.name.count('-') == 2:
                    date_folders.append(item.name)
            
            archive_status["archived_dates"] = sorted(date_folders)
            archive_status["archived_date_count"] = len(date_folders)
                
        except Exception:
            pass  # 靜默處理錯誤
        
        return archive_status
    
    def check_raw_folder(self) -> Dict[str, Any]:
        """檢查raw資料夾狀態"""
        print("🔍 檢查Raw資料夾...")
        
        if not self.raw_folder.exists():
            print(f"   ❌ raw資料夾不存在: {self.raw_folder}")
            return {"exists": False, "xml_files": 0, "vd_files": 0, "unprocessed": 0}
        
        # 掃描XML檔案
        xml_files = list(self.raw_folder.rglob("*.xml")) + list(self.raw_folder.rglob("*.txt"))
        vd_files = [f for f in xml_files if self._is_vd_file(f)]
        
        # 靜默檢查Archive狀態
        archive_status = self.check_archive_status_silent()
        archived_dates = set(archive_status["archived_dates"])
        
        # 快速檢查檔案狀態
        unprocessed_files = []
        processed_files = []
        
        for file_path in vd_files:
            try:
                file_date = self._extract_file_date_quick(file_path)
                if file_date and file_date in archived_dates:
                    processed_files.append(file_path)
                else:
                    unprocessed_files.append(file_path)
            except:
                unprocessed_files.append(file_path)
        
        result = {
            "exists": True,
            "xml_files": len(xml_files),
            "vd_files": len(vd_files),
            "unprocessed": len(unprocessed_files),
            "processed": len(processed_files),
            "archived_dates": len(archived_dates)
        }
        
        print(f"   📊 檔案狀態:")
        print(f"      • VD檔案總數: {result['vd_files']}")
        print(f"      • 已歸檔: {result['processed']} (涵蓋 {len(archived_dates)} 個日期)")
        print(f"      • 待處理: {result['unprocessed']}")
        
        if result['unprocessed'] > 0:
            estimated_minutes = result['unprocessed'] * 0.005
            print(f"      • 預估處理時間: {estimated_minutes:.1f} 分鐘")
        
        return result
    
    def _extract_file_date_quick(self, file_path: Path) -> str:
        """快速提取檔案日期"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 20:
                        break
                    if '<UpdateTime>' in line:
                        import re
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                        if date_match:
                            return date_match.group(1)
            return None
        except:
            return None
    
    def _is_vd_file(self, file_path: Path) -> bool:
        """檢查是否為VD檔案"""
        name_lower = file_path.name.lower()
        keywords = ['vd', '靜態資訊', 'traffic', 'detector']
        return any(keyword in name_lower for keyword in keywords)
    
    def get_or_create_date_folder(self, xml_timestamp: datetime) -> Path:
        """獲取或建立日期資料夾"""
        date_str = xml_timestamp.strftime('%Y-%m-%d')
        
        if date_str not in self.date_folders:
            date_folder = self.processed_base_folder / date_str
            date_folder.mkdir(parents=True, exist_ok=True)
            self.date_folders[date_str] = date_folder
        
        return self.date_folders[date_str]
    
    def get_date_file_paths(self, date_folder: Path) -> Dict[str, Path]:
        """獲取日期資料夾中的檔案路徑"""
        return {
            'main_csv': date_folder / "vd_data_all.csv",
            'main_json': date_folder / "vd_data_all_summary.json",
            'peak_csv': date_folder / "vd_data_peak.csv",
            'offpeak_csv': date_folder / "vd_data_offpeak.csv",
            'peak_json': date_folder / "vd_data_peak_summary.json",
            'offpeak_json': date_folder / "vd_data_offpeak_summary.json",
            'target_route_csv': date_folder / "target_route_data.csv",
            'target_route_json': date_folder / "target_route_data_summary.json",
            'target_route_peak_csv': date_folder / "target_route_peak.csv",
            'target_route_offpeak_csv': date_folder / "target_route_offpeak.csv",
            'target_route_peak_json': date_folder / "target_route_peak_summary.json",
            'target_route_offpeak_json': date_folder / "target_route_offpeak_summary.json",
            'processed_files_log': date_folder / "processed_files.json"
        }
    
    def classify_peak_hours_vectorized(self, datetime_series: pd.Series) -> pd.Series:
        """向量化尖峰離峰分類"""
        hours = datetime_series.dt.hour
        weekdays = datetime_series.dt.weekday
        
        categories = ['平日尖峰', '平日離峰', '假日尖峰', '假日離峰']
        result = pd.Series('平日離峰', index=datetime_series.index)
        
        # 假日判斷
        is_weekend = weekdays >= 5
        
        # 假日時間分類
        weekend_peak_mask = is_weekend & (
            ((hours >= 9) & (hours < 12)) |
            ((hours >= 15) & (hours < 19))
        )
        
        weekend_offpeak_mask = is_weekend & (
            ((hours >= 6) & (hours < 9)) |
            (hours >= 19) | (hours < 6)
        )
        
        result.loc[weekend_peak_mask] = '假日尖峰'
        result.loc[weekend_offpeak_mask] = '假日離峰'
        
        # 平日時間分類
        weekday_peak_mask = ~is_weekend & (
            ((hours >= 7) & (hours < 9)) |
            ((hours >= 17) & (hours < 20))
        )
        
        weekday_offpeak_mask = ~is_weekend & (
            ((hours >= 9) & (hours < 17)) |
            (hours >= 20) | (hours < 7)
        )
        
        result.loc[weekday_peak_mask] = '平日尖峰'
        result.loc[weekday_offpeak_mask] = '平日離峰'
        
        result = result.astype(pd.CategoricalDtype(categories=categories))
        return result
    
    def is_target_route_vectorized(self, vd_id_series: pd.Series) -> pd.Series:
        """向量化路段判斷"""
        return vd_id_series.isin(self.target_route_vd_ids)
    
    def quick_load_existing_data(self, target_date: str = None) -> pd.DataFrame:
        """快速載入已處理數據"""
        if target_date:
            date_folder = self.processed_base_folder / target_date
            if date_folder.exists():
                file_paths = self.get_date_file_paths(date_folder)
                main_csv = file_paths['main_csv']
                
                if main_csv.exists():
                    try:
                        print(f"⚡ 載入 {target_date} 數據...")
                        
                        if self.chunk_processing:
                            df = pd.read_csv(main_csv, engine='c', low_memory=True, chunksize=10000)
                            df = pd.concat(df, ignore_index=True)
                        else:
                            df = pd.read_csv(main_csv, engine='c', low_memory=False)
                        
                        df = self._optimize_dataframe_memory(df)
                        self._monitor_memory_and_gc()
                        
                        print(f"   ✅ 載入成功: {len(df):,} 筆記錄")
                        return df
                    except Exception as e:
                        print(f"   ❌ 載入失敗: {e}")
        else:
            print("⚡ 載入所有日期數據...")
            all_data = []
            
            for date_folder in self.processed_base_folder.iterdir():
                if date_folder.is_dir():
                    file_paths = self.get_date_file_paths(date_folder)
                    main_csv = file_paths['main_csv']
                    
                    if main_csv.exists():
                        try:
                            if self.chunk_processing:
                                df_chunks = pd.read_csv(main_csv, engine='c', low_memory=True, chunksize=10000)
                                df = pd.concat(df_chunks, ignore_index=True)
                            else:
                                df = pd.read_csv(main_csv, engine='c', low_memory=False)
                            
                            df = self._optimize_dataframe_memory(df)
                            all_data.append(df)
                            print(f"   ✅ {date_folder.name}: {len(df):,} 筆記錄")
                            
                            self._monitor_memory_and_gc()
                            
                        except Exception as e:
                            print(f"   ❌ {date_folder.name}: 載入失敗 - {e}")
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = self._optimize_dataframe_memory(combined_df)
                
                del all_data
                self._monitor_memory_and_gc(force=True)
                
                print(f"   🎯 總計載入: {len(combined_df):,} 筆記錄")
                return combined_df
        
        print("   ℹ️ 沒有找到已處理數據")
        return pd.DataFrame()
    
    def process_all_files(self) -> pd.DataFrame:
        """一次性處理所有XML檔案 - 簡化版"""
        print("🚀 開始處理Raw數據")
        print("=" * 60)
        
        start_time = time_module.time()
        
        # 檢查raw資料夾
        folder_status = self.check_raw_folder()
        
        if not folder_status["exists"]:
            print("❌ raw資料夾不存在，請先放入XML檔案")
            return pd.DataFrame()
        
        if folder_status["unprocessed"] == 0:
            print("✅ 所有檔案都已處理，載入現有數據")
            main_df = self.quick_load_existing_data()
            return main_df
        
        # 找到所有未處理檔案
        xml_files = list(self.raw_folder.rglob("*.xml")) + list(self.raw_folder.rglob("*.txt"))
        vd_files = [f for f in xml_files if self._is_vd_file(f)]
        
        # 靜默檢查Archive狀態
        archive_status = self.check_archive_status_silent()
        archived_dates = set(archive_status["archived_dates"])
        
        unprocessed_files = []
        for file_path in vd_files:
            file_date = self._extract_file_date_quick(file_path)
            if not file_date or file_date not in archived_dates:
                unprocessed_files.append(file_path)
        
        total_files = len(unprocessed_files)
        
        print(f"\n📋 處理計劃:")
        print(f"   • 待處理檔案: {total_files:,}")
        print(f"   • 處理線程: {self.max_workers}")
        print(f"   • 記憶體管理: 自動優化（後台）")
        
        # 開始處理
        print(f"\n🚀 開始處理...")
        
        date_organized_data = {}
        processed_count = 0
        failed_count = 0
        
        # 分批次處理
        batch_count = (total_files + self.internal_batch_size - 1) // self.internal_batch_size
        
        for batch_idx in range(batch_count):
            start_idx = batch_idx * self.internal_batch_size
            end_idx = min(start_idx + self.internal_batch_size, total_files)
            batch_files = unprocessed_files[start_idx:end_idx]
            
            print(f"   📦 批次 {batch_idx + 1}/{batch_count} ({len(batch_files)} 檔案)")
            
            # 並行處理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file_ultra_fast, file_path): file_path
                    for file_path in batch_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    processed_count += 1
                    
                    try:
                        result = future.result()
                        if result and result.get("success") and "data" in result:
                            xml_timestamp = result["xml_timestamp"]
                            date_str = xml_timestamp.strftime('%Y-%m-%d')
                            
                            if date_str not in date_organized_data:
                                date_organized_data[date_str] = []
                            date_organized_data[date_str].extend(result["data"])
                            
                            # 立即歸檔檔案
                            try:
                                self._archive_file_optimized(file_path, xml_timestamp)
                            except Exception:
                                pass  # 靜默處理歸檔錯誤
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1
                    
                    # 簡化進度顯示
                    if processed_count % 200 == 0 or processed_count == total_files:
                        progress = (processed_count / total_files) * 100
                        elapsed = time_module.time() - start_time
                        speed = processed_count / elapsed if elapsed > 0 else 0
                        total_records = sum(len(data_list) for data_list in date_organized_data.values())
                        
                        print(f"      進度: {processed_count}/{total_files} ({progress:.1f}%) "
                              f"| 速度: {speed:.1f} 檔案/秒 | 記錄: {total_records:,}")
                        
                        # 靜默記憶體清理
                        self._monitor_memory_and_gc()
            
            # 批次結束後靜默記憶體清理
            self._monitor_memory_and_gc(force=True)
        
        # 按日期保存數據並分類
        print(f"\n📊 按日期保存數據並分類...")
        
        all_combined_data = []
        date_summary = {}
        
        for date_str, data_list in date_organized_data.items():
            if data_list:
                print(f"   📅 {date_str}: {len(data_list):,} 筆記錄")
                
                date_folder = self.processed_base_folder / date_str
                date_folder.mkdir(parents=True, exist_ok=True)
                
                df = pd.DataFrame(data_list)
                df = self._optimize_dataframe_memory(df)
                
                file_paths = self.get_date_file_paths(date_folder)
                self._save_date_data_and_classify_silent(df, file_paths, date_str)
                
                all_combined_data.extend(data_list)
                date_summary[date_str] = len(data_list)
                
                del df
                self._monitor_memory_and_gc()
        
        # 清理臨時數據
        del date_organized_data
        self._monitor_memory_and_gc(force=True)
        
        # 生成總結報告
        total_time = time_module.time() - start_time
        
        if all_combined_data:
            total_df = pd.DataFrame(all_combined_data)
            total_df = self._optimize_dataframe_memory(total_df)
            
            print(f"\n🏁 處理完成！")
            print("=" * 60)
            print(f"📊 處理結果:")
            print(f"   ⏱️ 總時間: {total_time/60:.2f} 分鐘")
            print(f"   ✅ 成功處理: {processed_count - failed_count:,} 檔案")
            print(f"   📊 總記錄數: {len(total_df):,}")
            print(f"   📅 處理日期數: {len(date_summary)}")
            
            print(f"\n📅 各日期統計:")
            for date_str, count in sorted(date_summary.items()):
                print(f"      {date_str}: {count:,} 筆記錄")
            
            # 最終記憶體清理
            del all_combined_data
            self._monitor_memory_and_gc(force=True)
            
            return total_df
        else:
            print("❌ 無有效數據")
            return pd.DataFrame()
    
    def _save_date_data_and_classify_silent(self, df: pd.DataFrame, file_paths: Dict[str, Path], date_str: str):
        """靜默保存特定日期的數據並進行分類"""
        if df.empty:
            return
        
        # 確保時間欄位正確
        if not pd.api.types.is_datetime64_any_dtype(df['update_time']):
            df['update_time'] = pd.to_datetime(df['update_time'], errors='coerce')
        
        # 向量化分類
        df['time_category'] = self.classify_peak_hours_vectorized(df['update_time'])
        df['is_target_route'] = self.is_target_route_vectorized(df['vd_id'])
        
        # 1. 保存主檔案
        df_main = self._optimize_dataframe_memory(df.copy())
        df_main.to_csv(file_paths['main_csv'], index=False, encoding='utf-8-sig')
        summary = self._get_ultra_fast_summary(df_main, f"all_{date_str}", f"{date_str} 全部VD資料")
        with open(file_paths['main_json'], 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        del df_main
        self._monitor_memory_and_gc()
        
        # 2. 分批處理分類數據
        peak_mask = df['time_category'].str.contains('尖峰', na=False)
        target_route_mask = df['is_target_route']
        
        # 分批保存，避免同時載入所有分類
        classification_jobs = [
            ('peak', peak_mask, file_paths['peak_csv'], file_paths['peak_json'], f"{date_str} 尖峰時段"),
            ('offpeak', ~peak_mask, file_paths['offpeak_csv'], file_paths['offpeak_json'], f"{date_str} 離峰時段"),
            ('target_route', target_route_mask, file_paths['target_route_csv'], file_paths['target_route_json'], f"{date_str} 目標路段"),
            ('target_peak', target_route_mask & peak_mask, file_paths['target_route_peak_csv'], file_paths['target_route_peak_json'], f"{date_str} 目標路段尖峰"),
            ('target_offpeak', target_route_mask & ~peak_mask, file_paths['target_route_offpeak_csv'], file_paths['target_route_offpeak_json'], f"{date_str} 目標路段離峰")
        ]
        
        for job_name, mask, csv_path, json_path, description in classification_jobs:
            if mask.any():
                subset_df = df[mask].copy()
                subset_df = self._optimize_dataframe_memory(subset_df)
                
                subset_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                summary = self._get_ultra_fast_summary(subset_df, f"{csv_path.stem}_{date_str}", description)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
                
                del subset_df
                self._monitor_memory_and_gc()
    
    def _process_single_file_ultra_fast(self, file_path: Path) -> Dict[str, Any]:
        """超快單檔處理"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 快速提取時間
            update_time = self._extract_update_time_ultra_fast(root)
            date_str = update_time.strftime('%Y-%m-%d')
            
            # 快速提取數據
            vd_data_list = []
            vd_lives = root.findall('.//traffic:VDLive', self.namespace)
            
            for vd_live in vd_lives:
                vd_id_element = vd_live.find('traffic:VDID', self.namespace)
                if vd_id_element is None:
                    continue
                
                vd_id = vd_id_element.text
                lanes = vd_live.findall('.//traffic:Lane', self.namespace)
                
                for lane in lanes:
                    try:
                        lane_data = self._extract_lane_data_ultra_fast(lane, vd_id, date_str, update_time)
                        vd_data_list.append(lane_data)
                    except:
                        continue
            
            # 清理XML物件
            del tree, root
            
            if vd_data_list:
                return {
                    "success": True,
                    "file_name": file_path.name,
                    "record_count": len(vd_data_list),
                    "xml_timestamp": update_time,
                    "data": vd_data_list
                }
            else:
                return {"success": False, "file_name": file_path.name, "error": "無數據"}
                
        except Exception as e:
            return {"success": False, "file_name": file_path.name, "error": str(e)}
    
    def _extract_update_time_ultra_fast(self, root) -> datetime:
        """超快提取更新時間"""
        try:
            update_time_element = root.find('traffic:UpdateTime', self.namespace)
            if update_time_element is not None:
                time_str = update_time_element.text.replace('+08:00', '')
                return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except:
            pass
        return datetime.now()
    
    def _extract_lane_data_ultra_fast(self, lane_element, vd_id: str, date_str: str, 
                                    update_time: datetime) -> Dict[str, Any]:
        """超快提取車道數據"""
        def safe_get_int(element, tag_name, default=0):
            elem = element.find(f'traffic:{tag_name}', self.namespace)
            if elem is not None and elem.text is not None:
                try:
                    return int(elem.text)
                except:
                    return default
            return default
        
        # 基本資訊
        lane_id = safe_get_int(lane_element, 'LaneID')
        lane_type = safe_get_int(lane_element, 'LaneType')
        speed = safe_get_int(lane_element, 'Speed')
        occupancy = safe_get_int(lane_element, 'Occupancy')
        
        # 車種數據
        vehicles = lane_element.findall('traffic:Vehicles/traffic:Vehicle', self.namespace)
        volume_small = volume_large = volume_truck = 0
        speed_small = speed_large = speed_truck = 0
        
        for vehicle in vehicles:
            vehicle_type_elem = vehicle.find('traffic:VehicleType', self.namespace)
            if vehicle_type_elem is None:
                continue
            
            vehicle_type = vehicle_type_elem.text
            volume = safe_get_int(vehicle, 'Volume')
            v_speed = safe_get_int(vehicle, 'Speed')
            
            if vehicle_type == 'S':
                volume_small = volume
                speed_small = v_speed
            elif vehicle_type == 'L':
                volume_large = volume
                speed_large = v_speed
            elif vehicle_type == 'T':
                volume_truck = volume
                speed_truck = v_speed
        
        return {
            'date': date_str,
            'update_time': update_time,
            'vd_id': vd_id,
            'lane_id': lane_id,
            'lane_type': lane_type,
            'speed': speed,
            'occupancy': occupancy,
            'volume_total': volume_small + volume_large + volume_truck,
            'volume_small': volume_small,
            'volume_large': volume_large,
            'volume_truck': volume_truck,
            'speed_small': speed_small,
            'speed_large': speed_large,
            'speed_truck': speed_truck
        }
    
    def _get_ultra_fast_summary(self, df: pd.DataFrame, category: str, description: str) -> Dict[str, Any]:
        """超快生成數據摘要"""
        if df.empty:
            return {"category": category, "description": description, "error": "無數據"}
        
        summary = {
            "category": category,
            "description": description,
            "總記錄數": len(df),
            "VD設備數": df['vd_id'].nunique(),
            "交通統計": {
                "平均速度": round(df['speed'].mean(), 1),
                "平均佔有率": round(df['occupancy'].mean(), 1),
                "平均流量": round(df['volume_total'].mean(), 1),
                "最高速度": int(df['speed'].max()),
                "最高佔有率": int(df['occupancy'].max()),
                "最高流量": int(df['volume_total'].max())
            }
        }
        
        if 'date' in df.columns:
            summary["時間範圍"] = {
                "開始": str(df['date'].min())[:10],
                "結束": str(df['date'].max())[:10],
                "天數": df['date'].nunique()
            }
        
        if 'time_category' in df.columns:
            summary["時間分類統計"] = df['time_category'].value_counts().to_dict()
        
        if 'is_target_route' in df.columns:
            summary["路段統計"] = {
                "目標路段記錄": df['is_target_route'].sum(),
                "非目標路段記錄": (~df['is_target_route']).sum()
            }
        
        return summary
    
    def _archive_file_optimized(self, file_path: Path, xml_timestamp: datetime) -> str:
        """優化版歸檔檔案"""
        try:
            archive_date_folder = self.archive_folder / xml_timestamp.strftime("%Y-%m-%d")
            archive_date_folder.mkdir(exist_ok=True)
            
            new_filename = f"{xml_timestamp.strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
            archive_path = archive_date_folder / new_filename
            
            shutil.move(str(file_path), str(archive_path))
            return str(archive_path)
        except Exception:
            return ""  # 靜默處理錯誤
    
    def load_classified_data(self, target_date: str = None) -> Dict[str, pd.DataFrame]:
        """載入已分類的數據"""
        print("📂 載入已分類數據...")
        
        classified_data = {}
        
        if target_date:
            # 載入特定日期的數據
            date_folder = self.processed_base_folder / target_date
            if date_folder.exists():
                print(f"   📅 載入 {target_date} 數據")
                file_paths = self.get_date_file_paths(date_folder)
                classified_data = self._load_date_classified_data_silent(file_paths, target_date)
            else:
                print(f"   ⚠️ 日期資料夾不存在: {target_date}")
        else:
            # 載入所有日期的數據並合併
            print("   📅 載入所有日期數據並合併")
            all_date_data = {}
            
            for date_folder in self.processed_base_folder.iterdir():
                if date_folder.is_dir():
                    date_str = date_folder.name
                    file_paths = self.get_date_file_paths(date_folder)
                    date_data = self._load_date_classified_data_silent(file_paths, date_str)
                    
                    # 合併到總數據
                    for key, df in date_data.items():
                        if not df.empty:
                            if key not in all_date_data:
                                all_date_data[key] = []
                            all_date_data[key].append(df)
                    
                    # 靜默記憶體清理
                    self._monitor_memory_and_gc()
            
            # 合併各日期的數據
            for key, df_list in all_date_data.items():
                if df_list:
                    combined_df = pd.concat(df_list, ignore_index=True)
                    combined_df = self._optimize_dataframe_memory(combined_df)
                    classified_data[key] = combined_df
                    print(f"   ✅ {key}: {len(combined_df):,} 筆記錄")
                    
                    # 清理臨時數據
                    del df_list
                    self._monitor_memory_and_gc()
        
        return classified_data
    
    def _load_date_classified_data_silent(self, file_paths: Dict[str, Path], date_str: str) -> Dict[str, pd.DataFrame]:
        """靜默載入特定日期的分類數據"""
        file_mappings = {
            'all': file_paths['main_csv'],
            'peak': file_paths['peak_csv'],
            'offpeak': file_paths['offpeak_csv'],
            'target_route': file_paths['target_route_csv'],
            'target_route_peak': file_paths['target_route_peak_csv'],
            'target_route_offpeak': file_paths['target_route_offpeak_csv']
        }
        
        classified_data = {}
        
        for name, file_path in file_mappings.items():
            if file_path.exists():
                try:
                    # 記憶體優化載入
                    if self.chunk_processing and file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB以上使用chunk
                        df_chunks = pd.read_csv(file_path, engine='c', low_memory=True, chunksize=10000)
                        df = pd.concat(df_chunks, ignore_index=True)
                    else:
                        df = pd.read_csv(file_path, engine='c', low_memory=True)
                    
                    df = self._optimize_dataframe_memory(df)
                    classified_data[name] = df
                    print(f"      ✅ {date_str} {name}: {len(df):,} 筆記錄")
                    
                    # 靜默記憶體監控
                    self._monitor_memory_and_gc()
                    
                except Exception:
                    classified_data[name] = pd.DataFrame()
            else:
                classified_data[name] = pd.DataFrame()
        
        return classified_data
    
    def list_available_dates(self) -> List[str]:
        """列出可用的日期資料夾"""
        available_dates = []
        
        if self.processed_base_folder.exists():
            for date_folder in self.processed_base_folder.iterdir():
                if date_folder.is_dir() and date_folder.name.count('-') == 2:
                    available_dates.append(date_folder.name)
        
        return sorted(available_dates)
    
    def get_date_summary(self) -> Dict[str, Any]:
        """獲取各日期數據摘要"""
        print("📊 生成日期摘要...")
        
        date_summary = {
            "總覽": {
                "可用日期數": 0,
                "總記錄數": 0,
                "日期範圍": {"最早": None, "最晚": None}
            },
            "各日期詳情": {}
        }
        
        available_dates = self.list_available_dates()
        date_summary["總覽"]["可用日期數"] = len(available_dates)
        
        if available_dates:
            date_summary["總覽"]["日期範圍"]["最早"] = available_dates[0]
            date_summary["總覽"]["日期範圍"]["最晚"] = available_dates[-1]
        
        total_records = 0
        
        for date_str in available_dates:
            date_folder = self.processed_base_folder / date_str
            file_paths = self.get_date_file_paths(date_folder)
            
            main_csv = file_paths['main_csv']
            if main_csv.exists():
                try:
                    # 只讀取第一行來檢查結構
                    df_sample = pd.read_csv(main_csv, nrows=1)
                    
                    # 從檔案大小估算記錄數
                    file_size = main_csv.stat().st_size
                    estimated_records = int(file_size / 1024 * 10)
                    
                    date_summary["各日期詳情"][date_str] = {
                        "主檔案存在": True,
                        "預估記錄數": estimated_records,
                        "檔案大小MB": round(file_size / 1024 / 1024, 1),
                        "欄位數": len(df_sample.columns)
                    }
                    
                    total_records += estimated_records
                    del df_sample
                    
                except Exception as e:
                    date_summary["各日期詳情"][date_str] = {
                        "主檔案存在": True,
                        "錯誤": str(e)
                    }
            else:
                date_summary["各日期詳情"][date_str] = {
                    "主檔案存在": False
                }
        
        date_summary["總覽"]["總記錄數"] = total_records
        
        return date_summary


# 便利函數
def process_all_files_one_shot(folder_path: str = "data") -> pd.DataFrame:
    """一次性處理所有檔案 - 簡化版"""
    loader = VDDataLoader(base_folder=folder_path)
    return loader.process_all_files()


def load_classified_data_quick(folder_path: str = "data", target_date: str = None) -> Dict[str, pd.DataFrame]:
    """快速載入已分類數據 - 簡化版"""
    loader = VDDataLoader(base_folder=folder_path)
    return loader.load_classified_data(target_date=target_date)


def get_date_summary_quick(folder_path: str = "data") -> Dict[str, Any]:
    """快速獲取日期摘要 - 簡化版"""
    loader = VDDataLoader(base_folder=folder_path)
    return loader.get_date_summary()


if __name__ == "__main__":
    print("🚀 VD數據載入器 - 簡化版（靜默記憶體優化）")
    print("=" * 70)
    print("特色：專注Raw處理 + 後台記憶體優化")
    print("=" * 70)
    
    loader = VDDataLoader()
    
    # 顯示可用日期
    available_dates = loader.list_available_dates()
    if available_dates:
        print(f"\n📅 已處理日期:")
        for date_str in available_dates:
            print(f"   • {date_str}")
    
    # 檢查raw資料夾
    status = loader.check_raw_folder()
    
    if status["unprocessed"] > 0:
        print(f"\n🎯 發現 {status['unprocessed']} 個待處理檔案")
        print("簡化版特色：")
        print("   • 後台自動記憶體優化，不顯示詳細信息")
        print("   • 專注Raw數據處理進度")
        print("   • 智慧Archive檢查，避免重複處理")
        print("   • 按日期組織輸出")
        
        response = input(f"\n開始處理Raw數據？(y/N): ")
        if response.lower() in ['y', 'yes']:
            df = loader.process_all_files()
            
            if not df.empty:
                print(f"\n🎉 Raw數據處理完成！")
                
                # 顯示日期摘要
                date_summary = loader.get_date_summary()
                print(f"\n📊 處理結果:")
                print(f"   📅 處理日期數: {date_summary['總覽']['可用日期數']}")
                print(f"   📊 總記錄數: {date_summary['總覽']['總記錄數']:,}")
                print(f"   📆 日期範圍: {date_summary['總覽']['日期範圍']['最早']} ~ {date_summary['總覽']['日期範圍']['最晚']}")
                
                print(f"\n📁 輸出結構:")
                for date_str in loader.list_available_dates():
                    print(f"   📂 data/processed/{date_str}/")
                    print(f"      ├── vd_data_all.csv + _summary.json")
                    print(f"      ├── vd_data_peak.csv + _summary.json")
                    print(f"      ├── vd_data_offpeak.csv + _summary.json")
                    print(f"      ├── target_route_*.csv + _summary.json")
                    print(f"      └── processed_files.json")
    else:
        if available_dates:
            print(f"\n✅ 已有 {len(available_dates)} 個日期的處理數據")
            
            # 顯示日期摘要
            date_summary = loader.get_date_summary()
            print(f"\n📊 現有數據摘要:")
            for date_str, details in date_summary["各日期詳情"].items():
                if details.get("主檔案存在"):
                    print(f"   📅 {date_str}: {details.get('預估記錄數', 0):,} 筆記錄 ({details.get('檔案大小MB', 0):.1f}MB)")
        else:
            print("💡 請將XML檔案放入 data/raw/ 資料夾")
    
    print(f"\n💡 簡化版使用方法:")
    print(f"   # 處理所有Raw檔案（後台記憶體優化）")
    print(f"   loader = VDDataLoader()")
    print(f"   df = loader.process_all_files()")
    print(f"   ")
    print(f"   # 載入特定日期數據")
    print(f"   date_data = loader.load_classified_data(target_date='2025-06-27')")
    print(f"   ")
    print(f"   # 獲取日期摘要")
    print(f"   summary = loader.get_date_summary()")
    
    print(f"\n🎯 簡化版優勢:")
    print(f"   🎯 專注Raw處理：主要顯示處理進度")
    print(f"   💾 後台記憶體優化：自動管理，不干擾用戶")
    print(f"   📂 智慧Archive檢查：快速檢查，避免重複")
    print(f"   📊 簡潔輸出：只顯示重要信息")
    print(f"   ⚡ 保持高速：維持3-5分鐘處理千萬筆記錄")
    print(f"   🔄 完整功能：保留所有原功能")