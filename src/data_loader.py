# src/data_loader.py - 強化版彈性處理

"""
VD數據載入器 - 強化版彈性處理
===============================

核心特性：
1. 🔄 彈性數據量檢測 - 自動檢測raw資料夾檔案數量
2. 💾 積極記憶體管理 - 分段處理，即時釋放記憶體
3. 🎯 精準路段篩選 - 圓山、台北、三重路段專項處理
4. 📁 標準化輸出格式 - 統一三個目標檔案
5. 🏷️ 原檔名歸檔 - 保持原始檔案名稱
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import shutil
import threading
import time as time_module
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Callable, Optional
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')


class FlexibleResourceManager:
    """彈性資源管理器"""
    
    def __init__(self, target_memory_percent: float = 60.0):
        self.target_memory_percent = target_memory_percent
        self.safe_memory_percent = 55.0  # 安全記憶體閾值
        self.critical_memory_percent = 80.0  # 臨界記憶體閾值
        self.min_batch_size = 10
        self.max_batch_size = 100
        self.current_batch_size = 50
        
    def get_memory_status(self) -> Dict[str, float]:
        """獲取記憶體狀態"""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
    
    def should_pause_processing(self) -> bool:
        """是否應該暫停處理"""
        memory_percent = psutil.virtual_memory().percent
        return memory_percent > self.critical_memory_percent
    
    def should_force_gc(self) -> bool:
        """是否強制垃圾回收"""
        memory_percent = psutil.virtual_memory().percent
        return memory_percent > self.target_memory_percent
    
    def adjust_batch_size(self, current_memory: float) -> int:
        """調整批次大小"""
        if current_memory > 75:
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.7))
        elif current_memory < 45:
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.3))
        
        return self.current_batch_size


class VDDataLoader:
    """VD數據載入器 - 強化版彈性處理"""
    
    def __init__(self, base_folder: str = "data", max_workers: int = None, 
                 target_memory_percent: float = 60.0, verbose: bool = True):
        """初始化載入器"""
        self.base_folder = Path(base_folder)
        self.raw_folder = self.base_folder / "raw"
        self.processed_base_folder = self.base_folder / "processed"
        self.archive_folder = self.base_folder / "archive"
        self.verbose = verbose
        
        # 創建必要資料夾
        for folder in [self.raw_folder, self.processed_base_folder, self.archive_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # 資源管理器
        self.resource_manager = FlexibleResourceManager(target_memory_percent)
        
        # 線程數設定
        self.max_workers = max_workers or max(2, min(os.cpu_count() or 4, 6))
        
        # XML命名空間
        self.namespace = {
            'traffic': 'http://traffic.transportdata.tw/standard/traffic/schema/'
        }
        
        # 目標路段關鍵字
        self.target_keywords = ['圓山', '台北', '三重', 'N1-N-2', 'N1-S-2']
        
        # 線程鎖
        self.file_lock = threading.Lock()
        self.progress_callback = None
        
        print(f"🚀 VD數據載入器強化版初始化")
        print(f"   📁 資料夾: {self.base_folder}")
        print(f"   🧵 線程數: {self.max_workers}")
        print(f"   💾 目標記憶體: {target_memory_percent}%")
    
    def scan_raw_files(self) -> Dict[str, Any]:
        """彈性掃描raw資料夾"""
        print("🔍 彈性掃描raw資料夾...")
        
        if not self.raw_folder.exists():
            return {"exists": False, "file_count": 0, "files": []}
        
        # 掃描所有可能的檔案類型
        xml_files = list(self.raw_folder.rglob("*.xml"))
        txt_files = list(self.raw_folder.rglob("*.txt"))
        all_files = xml_files + txt_files
        
        # 篩選VD相關檔案
        vd_files = []
        for file_path in all_files:
            if self._is_vd_file(file_path):
                vd_files.append(file_path)
        
        # 檢查已處理檔案
        archived_files = self._get_archived_files()
        unprocessed_files = []
        
        for file_path in vd_files:
            if file_path.name not in archived_files:
                unprocessed_files.append(file_path)
        
        result = {
            "exists": True,
            "file_count": len(vd_files),
            "unprocessed_count": len(unprocessed_files),
            "unprocessed_files": unprocessed_files,
            "processed_count": len(vd_files) - len(unprocessed_files)
        }
        
        print(f"   📊 掃描結果:")
        print(f"      • 總VD檔案: {result['file_count']}")
        print(f"      • 待處理: {result['unprocessed_count']}")
        print(f"      • 已處理: {result['processed_count']}")
        
        return result
    
    def _is_vd_file(self, file_path: Path) -> bool:
        """檢查是否為VD檔案"""
        name_lower = file_path.name.lower()
        keywords = ['vd', 'traffic', 'detector', '靜態', '動態']
        return any(keyword in name_lower for keyword in keywords)
    
    def _get_archived_files(self) -> set:
        """獲取已歸檔的檔案名稱"""
        archived_files = set()
        
        if not self.archive_folder.exists():
            return archived_files
        
        for date_folder in self.archive_folder.iterdir():
            if date_folder.is_dir():
                for archived_file in date_folder.iterdir():
                    if archived_file.is_file():
                        # 提取原始檔案名稱
                        original_name = self._extract_original_filename(archived_file.name)
                        if original_name:
                            archived_files.add(original_name)
        
        return archived_files
    
    def _extract_original_filename(self, archived_name: str) -> str:
        """從歸檔檔案名提取原始檔案名"""
        # 移除時間戳前綴 (格式: YYYYMMDD_HHMMSS_原檔名)
        parts = archived_name.split('_', 2)
        if len(parts) >= 3:
            return parts[2]
        return archived_name
    
    def process_all_files_flexible(self) -> pd.DataFrame:
        """彈性處理所有檔案"""
        print("🚀 開始彈性處理")
        print("=" * 50)
        
        # 掃描檔案
        scan_result = self.scan_raw_files()
        
        if not scan_result["exists"] or scan_result["unprocessed_count"] == 0:
            print("📂 載入現有數據...")
            return self.load_existing_data()
        
        unprocessed_files = scan_result["unprocessed_files"]
        total_files = len(unprocessed_files)
        
        print(f"📋 處理計劃: {total_files} 檔案")
        
        # 分段處理
        processed_count = 0
        all_date_data = {}
        
        start_time = time_module.time()
        
        while processed_count < total_files:
            # 檢查記憶體狀況
            memory_status = self.resource_manager.get_memory_status()
            
            if self.resource_manager.should_pause_processing():
                print(f"   ⚠️ 記憶體使用過高({memory_status['percent']:.1f}%)，執行清理...")
                self._aggressive_cleanup()
                continue
            
            # 動態調整批次大小
            batch_size = self.resource_manager.adjust_batch_size(memory_status['percent'])
            
            # 處理批次
            batch_start = processed_count
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = unprocessed_files[batch_start:batch_end]
            
            if self.verbose:
                print(f"   📦 批次處理 {batch_start+1}-{batch_end}/{total_files} "
                      f"(記憶體: {memory_status['percent']:.1f}%, 批次: {len(batch_files)})")
            
            # 處理批次檔案
            batch_data = self._process_batch_safe(batch_files)
            
            # 合併批次數據
            for date_str, data_list in batch_data.items():
                if date_str not in all_date_data:
                    all_date_data[date_str] = []
                all_date_data[date_str].extend(data_list)
            
            processed_count = batch_end
            
            # 進度報告
            if processed_count % 100 == 0 or processed_count == total_files:
                elapsed = time_module.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                total_records = sum(len(data_list) for data_list in all_date_data.values())
                
                print(f"   📈 進度: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%) "
                      f"| 速度: {speed:.1f} 檔案/秒 | 記錄: {total_records:,} "
                      f"| 記憶體: {memory_status['percent']:.1f}%")
                
                # 進度回調
                if self.progress_callback:
                    try:
                        self.progress_callback({
                            'progress': processed_count/total_files*100,
                            'memory_usage': memory_status['percent'],
                            'records': total_records
                        })
                    except:
                        pass
            
            # 清理批次數據
            del batch_data
            
            # 檢查是否需要垃圾回收
            if self.resource_manager.should_force_gc():
                gc.collect()
            
            # 防止處理過快導致系統負載過高
            time_module.sleep(0.1)
        
        # 保存最終數據
        return self._save_final_data_flexible(all_date_data)
    
    def _process_batch_safe(self, batch_files: List[Path]) -> Dict[str, List]:
        """安全的批次處理"""
        batch_data = {}
        
        # 較保守的線程數
        safe_workers = min(self.max_workers, 4)
        
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file_safe, file_path): file_path
                for file_path in batch_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result(timeout=60)  # 增加超時時間
                    
                    if result and result.get("success") and "data" in result:
                        date_str = result["xml_timestamp"].strftime('%Y-%m-%d')
                        
                        if date_str not in batch_data:
                            batch_data[date_str] = []
                        batch_data[date_str].extend(result["data"])
                        
                        # 立即歸檔（使用原檔名）
                        self._archive_file_original_name(file_path, result["xml_timestamp"])
                        
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ 檔案處理失敗: {file_path.name} - {e}")
        
        return batch_data
    
    def _process_single_file_safe(self, file_path: Path) -> Dict[str, Any]:
        """安全的單檔處理"""
        try:
            # 讀取檔案內容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 解析XML
            root = ET.fromstring(content)
            
            # 提取時間
            update_time = self._extract_update_time(root)
            date_str = update_time.strftime('%Y-%m-%d')
            
            # 提取目標路段數據
            target_data = []
            
            for vd_live in root.findall('.//traffic:VDLive', self.namespace):
                vd_id_element = vd_live.find('traffic:VDID', self.namespace)
                if vd_id_element is None:
                    continue
                
                vd_id = vd_id_element.text or ""
                
                # 檢查是否為目標路段
                if not self._is_target_route(vd_id):
                    continue
                
                # 提取車道數據
                for lane in vd_live.findall('.//traffic:Lane', self.namespace):
                    try:
                        lane_data = self._extract_lane_data(lane, vd_id, date_str, update_time)
                        target_data.append(lane_data)
                    except:
                        continue
            
            # 清理XML對象
            del root, content
            
            return {
                "success": True,
                "file_name": file_path.name,
                "record_count": len(target_data),
                "xml_timestamp": update_time,
                "data": target_data
            }
            
        except Exception as e:
            return {"success": False, "file_name": file_path.name, "error": str(e)}
    
    def _is_target_route(self, vd_id: str) -> bool:
        """檢查是否為目標路段"""
        if not isinstance(vd_id, str):
            return False
        
        # 檢查關鍵字
        for keyword in self.target_keywords:
            if keyword in vd_id:
                return True
        
        # 檢查國道1號圓山-三重路段的里程數
        if 'N1' in vd_id:
            # 提取里程數
            parts = vd_id.split('-')
            for part in parts:
                try:
                    if '.' in part:
                        km = float(part)
                    else:
                        km = int(part)
                    
                    # 圓山-三重路段大約在20-30公里
                    if 20 <= km <= 30:
                        return True
                except:
                    continue
        
        return False
    
    def _extract_update_time(self, root) -> datetime:
        """提取更新時間"""
        try:
            update_time_element = root.find('traffic:UpdateTime', self.namespace)
            if update_time_element is not None:
                time_str = update_time_element.text.replace('+08:00', '')
                return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except:
            pass
        return datetime.now()
    
    def _extract_lane_data(self, lane_element, vd_id: str, date_str: str, 
                          update_time: datetime) -> Dict[str, Any]:
        """提取車道數據"""
        def safe_get_int(element, tag_name, default=0):
            elem = element.find(f'traffic:{tag_name}', self.namespace)
            try:
                return int(elem.text) if elem is not None and elem.text else default
            except:
                return default
        
        # 基本資訊
        lane_id = safe_get_int(lane_element, 'LaneID')
        lane_type = safe_get_int(lane_element, 'LaneType')
        speed = safe_get_int(lane_element, 'Speed')
        occupancy = safe_get_int(lane_element, 'Occupancy')
        
        # 車種數據
        volume_small = volume_large = volume_truck = 0
        speed_small = speed_large = speed_truck = 0
        
        for vehicle in lane_element.findall('traffic:Vehicles/traffic:Vehicle', self.namespace):
            vehicle_type_elem = vehicle.find('traffic:VehicleType', self.namespace)
            if vehicle_type_elem is None:
                continue
            
            vehicle_type = vehicle_type_elem.text
            volume = safe_get_int(vehicle, 'Volume')
            v_speed = safe_get_int(vehicle, 'Speed')
            
            if vehicle_type == 'S':  # 小車
                volume_small, speed_small = volume, v_speed
            elif vehicle_type == 'L':  # 大車
                volume_large, speed_large = volume, v_speed
            elif vehicle_type == 'T':  # 卡車
                volume_truck, speed_truck = volume, v_speed
        
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
    
    def _archive_file_original_name(self, file_path: Path, xml_timestamp: datetime) -> str:
        """使用原檔名歸檔檔案"""
        try:
            archive_date_folder = self.archive_folder / xml_timestamp.strftime("%Y-%m-%d")
            archive_date_folder.mkdir(exist_ok=True)
            
            # 使用原檔名，只加時間戳前綴避免重名
            new_filename = f"{xml_timestamp.strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
            archive_path = archive_date_folder / new_filename
            
            shutil.move(str(file_path), str(archive_path))
            return str(archive_path)
        except:
            return ""
    
    def _save_final_data_flexible(self, all_date_data: Dict[str, List]) -> pd.DataFrame:
        """彈性保存最終數據"""
        print(f"\n📊 保存處理結果...")
        
        all_data = []
        
        for date_str, data_list in all_date_data.items():
            if not data_list:
                continue
            
            print(f"   💾 處理 {date_str}: {len(data_list):,} 筆記錄")
            
            # 創建DataFrame
            df = pd.DataFrame(data_list)
            
            # 優化記憶體
            df = self._optimize_dataframe_memory(df)
            
            # 分類並保存
            self._save_date_target_files(df, date_str)
            
            all_data.extend(data_list)
            
            # 清理
            del df, data_list
            
            # 垃圾回收
            if self.resource_manager.should_force_gc():
                gc.collect()
        
        # 返回合併結果
        if all_data:
            final_df = pd.DataFrame(all_data)
            final_df = self._optimize_dataframe_memory(final_df)
            
            print(f"🎯 處理完成: {len(final_df):,} 筆目標路段記錄")
            
            # 最終清理
            del all_data
            gc.collect()
            
            return final_df
        
        return pd.DataFrame()
    
    def _save_date_target_files(self, df: pd.DataFrame, date_str: str):
        """保存指定日期的目標檔案"""
        date_folder = self.processed_base_folder / date_str
        date_folder.mkdir(parents=True, exist_ok=True)
        
        # 添加時間分類
        df['time_category'] = self._classify_peak_hours(df['update_time'])
        
        # 1. 目標路段數據 (所有目標路段記錄)
        target_route_csv = date_folder / "target_route_data.csv"
        df.to_csv(target_route_csv, index=False, encoding='utf-8-sig')
        
        # 2. 目標路段尖峰
        peak_mask = df['time_category'].str.contains('尖峰', na=False)
        if peak_mask.any():
            peak_df = df[peak_mask].copy()
            target_peak_csv = date_folder / "target_route_peak.csv"
            peak_df.to_csv(target_peak_csv, index=False, encoding='utf-8-sig')
            del peak_df
        
        # 3. 目標路段離峰
        offpeak_mask = ~peak_mask
        if offpeak_mask.any():
            offpeak_df = df[offpeak_mask].copy()
            target_offpeak_csv = date_folder / "target_route_offpeak.csv"
            offpeak_df.to_csv(target_offpeak_csv, index=False, encoding='utf-8-sig')
            del offpeak_df
        
        # 生成摘要
        summary = {
            "date": date_str,
            "total_records": len(df),
            "peak_records": peak_mask.sum(),
            "offpeak_records": offpeak_mask.sum(),
            "avg_speed": df['speed'].mean(),
            "avg_volume": df['volume_total'].mean(),
            "unique_vd_count": df['vd_id'].nunique()
        }
        
        summary_path = date_folder / "target_route_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    
    def _classify_peak_hours(self, datetime_series: pd.Series) -> pd.Series:
        """分類尖峰離峰時段"""
        # 確保是datetime類型
        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            datetime_series = pd.to_datetime(datetime_series, errors='coerce')
        
        hours = datetime_series.dt.hour
        weekdays = datetime_series.dt.weekday
        
        # 週末判斷
        is_weekend = weekdays >= 5
        
        # 尖峰時段判斷
        weekday_peak = ((hours >= 7) & (hours < 9)) | ((hours >= 17) & (hours < 20))
        weekend_peak = ((hours >= 9) & (hours < 12)) | ((hours >= 15) & (hours < 19))
        
        # 分類
        result = pd.Series('平日離峰', index=datetime_series.index)
        result[weekday_peak & ~is_weekend] = '平日尖峰'
        result[weekend_peak & is_weekend] = '假日尖峰'
        result[~weekend_peak & is_weekend] = '假日離峰'
        
        return result
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """優化DataFrame記憶體使用"""
        if df.empty:
            return df
        
        # 數值類型優化
        numeric_columns = ['lane_id', 'lane_type', 'speed', 'occupancy',
                          'volume_total', 'volume_small', 'volume_large', 'volume_truck',
                          'speed_small', 'speed_large', 'speed_truck']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        
        # 類別類型優化
        if 'vd_id' in df.columns:
            df['vd_id'] = df['vd_id'].astype('category')
        
        if 'time_category' in df.columns:
            df['time_category'] = df['time_category'].astype('category')
        
        # 時間類型確保
        if 'update_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['update_time']):
            df['update_time'] = pd.to_datetime(df['update_time'], errors='coerce')
        
        return df
    
    def _aggressive_cleanup(self):
        """積極清理記憶體"""
        print("   🧹 執行積極記憶體清理...")
        
        # 多次垃圾回收
        for _ in range(3):
            gc.collect()
            time_module.sleep(0.5)
        
        # 檢查清理效果
        memory_after = psutil.virtual_memory().percent
        print(f"   💾 清理後記憶體: {memory_after:.1f}%")
        
        # 如果記憶體仍然過高，暫停更長時間
        if memory_after > 75:
            print("   ⏳ 記憶體仍高，等待系統釋放...")
            time_module.sleep(5)
    
    def load_existing_data(self, target_date: str = None) -> pd.DataFrame:
        """載入現有數據"""
        print("📂 載入現有目標路段數據...")
        
        available_dates = self.list_available_dates()
        
        if not available_dates:
            print("   ⚠️ 沒有找到已處理數據")
            return pd.DataFrame()
        
        all_data = []
        
        dates_to_load = [target_date] if target_date else available_dates
        
        for date_str in dates_to_load:
            if date_str not in available_dates:
                continue
            
            date_folder = self.processed_base_folder / date_str
            target_csv = date_folder / "target_route_data.csv"
            
            if target_csv.exists():
                try:
                    df = pd.read_csv(target_csv, engine='c', low_memory=True)
                    df = self._optimize_dataframe_memory(df)
                    all_data.append(df)
                    print(f"   ✅ {date_str}: {len(df):,} 筆記錄")
                except Exception as e:
                    print(f"   ❌ {date_str}: 載入失敗 - {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = self._optimize_dataframe_memory(combined_df)
            print(f"   🎯 總計載入: {len(combined_df):,} 筆目標路段記錄")
            return combined_df
        
        return pd.DataFrame()
    
    def list_available_dates(self) -> List[str]:
        """列出可用日期"""
        if not self.processed_base_folder.exists():
            return []
        
        dates = []
        for date_folder in self.processed_base_folder.iterdir():
            if date_folder.is_dir() and date_folder.name.count('-') == 2:
                # 檢查是否有目標檔案
                target_csv = date_folder / "target_route_data.csv"
                if target_csv.exists():
                    dates.append(date_folder.name)
        
        return sorted(dates)
    
    def check_data_readiness(self) -> Dict[str, Any]:
        """檢查數據就緒度"""
        scan_result = self.scan_raw_files()
        available_dates = self.list_available_dates()
        memory_status = self.resource_manager.get_memory_status()
        
        readiness = {
            'timestamp': datetime.now().isoformat(),
            'raw_files': scan_result,
            'processed_dates': len(available_dates),
            'memory_status': memory_status,
            'overall_readiness': 'unknown',
            'next_action': 'unknown',
            'recommendations': []
        }
        
        # 判斷整體就緒度
        unprocessed_count = scan_result.get('unprocessed_count', 0)
        
        if unprocessed_count > 0:
            readiness['overall_readiness'] = 'raw_processing_needed'
            readiness['next_action'] = 'process_raw_files'
            readiness['recommendations'].append(f"發現 {unprocessed_count} 個待處理檔案")
            
            # 記憶體建議
            if memory_status['percent'] > 70:
                readiness['recommendations'].append("建議先關閉其他程序釋放記憶體")
        elif len(available_dates) > 0:
            readiness['overall_readiness'] = 'ready_for_analysis'
            readiness['next_action'] = 'proceed_to_analysis'
            readiness['recommendations'].append(f"已有 {len(available_dates)} 個日期的目標路段數據")
        else:
            readiness['overall_readiness'] = 'no_data'
            readiness['next_action'] = 'add_raw_files'
            readiness['recommendations'].append("請將VD XML檔案放入 data/raw/ 資料夾")
        
        return readiness
    
    def auto_process_if_needed(self, progress_callback: Callable = None) -> Dict[str, Any]:
        """智能自動處理"""
        self.progress_callback = progress_callback
        
        readiness = self.check_data_readiness()
        result = {'action_taken': 'none', 'success': False, 'message': '', 'data': None}
        
        if readiness['overall_readiness'] == 'raw_processing_needed':
            try:
                df = self.process_all_files_flexible()
                result['action_taken'] = 'processed_raw_files'
                result['success'] = not df.empty
                result['message'] = f"成功處理 {len(df):,} 筆目標路段記錄" if result['success'] else "處理失敗"
                result['data'] = df
            except Exception as e:
                result['message'] = f"自動處理失敗: {e}"
        elif readiness['overall_readiness'] == 'ready_for_analysis':
            result['action_taken'] = 'ready_for_analysis'
            result['success'] = True
            result['message'] = "目標路段數據已就緒，可進行AI分析"
        else:
            result['message'] = readiness['recommendations'][0] if readiness['recommendations'] else "無需處理"
        
        return result
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """獲取處理摘要"""
        available_dates = self.list_available_dates()
        
        if not available_dates:
            return {'available_dates': 0, 'total_records': 0, 'date_details': {}}
        
        total_records = 0
        date_details = {}
        
        for date_str in available_dates:
            date_folder = self.processed_base_folder / date_str
            summary_file = date_folder / "target_route_summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    date_details[date_str] = summary
                    total_records += summary.get('total_records', 0)
                except:
                    pass
        
        return {
            'available_dates': len(available_dates),
            'total_records': total_records,
            'date_details': date_details,
            'date_range': {
                'start': available_dates[0] if available_dates else None,
                'end': available_dates[-1] if available_dates else None
            }
        }


# ============================================================
# 便利函數
# ============================================================

def process_target_route_data(folder_path: str = "data", 
                             target_memory_percent: float = 60.0) -> pd.DataFrame:
    """處理目標路段數據"""
    loader = VDDataLoader(base_folder=folder_path, 
                         target_memory_percent=target_memory_percent)
    return loader.process_all_files_flexible()


def load_target_route_data(folder_path: str = "data", 
                          target_date: str = None) -> pd.DataFrame:
    """載入目標路段數據"""
    loader = VDDataLoader(base_folder=folder_path)
    return loader.load_existing_data(target_date)


def check_system_readiness(folder_path: str = "data") -> Dict[str, Any]:
    """檢查系統就緒度"""
    loader = VDDataLoader(base_folder=folder_path)
    return loader.check_data_readiness()


def auto_process_data(folder_path: str = "data", 
                     progress_callback: Callable = None) -> Dict[str, Any]:
    """自動處理數據"""
    loader = VDDataLoader(base_folder=folder_path)
    return loader.auto_process_if_needed(progress_callback)


if __name__ == "__main__":
    print("🚀 VD數據載入器 - 強化版彈性處理")
    print("=" * 60)
    print("🎯 核心特性:")
    print("   🔄 彈性數據量檢測 - 自動檢測檔案數量")
    print("   💾 積極記憶體管理 - 分段處理，即時釋放")
    print("   🎯 精準路段篩選 - 圓山、台北、三重專項")
    print("   📁 標準化輸出 - target_route_data/peak/offpeak.csv")
    print("   🏷️ 原檔名歸檔 - 保持原始檔案名稱")
    print("=" * 60)
    
    # 示範使用
    loader = VDDataLoader(target_memory_percent=60.0, verbose=True)
    
    # 檢查就緒度
    readiness = loader.check_data_readiness()
    print(f"\n📊 系統狀態: {readiness['overall_readiness']}")
    
    # 顯示檔案掃描結果
    raw_files = readiness['raw_files']
    if raw_files['exists']:
        print(f"📁 Raw檔案狀況:")
        print(f"   • 總檔案數: {raw_files['file_count']}")
        print(f"   • 待處理: {raw_files['unprocessed_count']}")
        print(f"   • 已處理: {raw_files['processed_count']}")
    
    # 顯示記憶體狀況
    memory_status = readiness['memory_status']
    print(f"💾 記憶體狀況:")
    print(f"   • 使用率: {memory_status['percent']:.1f}%")
    print(f"   • 可用: {memory_status['available_gb']:.1f}GB")
    print(f"   • 總計: {memory_status['total_gb']:.1f}GB")
    
    # 建議行動
    if readiness['recommendations']:
        print(f"💡 系統建議:")
        for i, rec in enumerate(readiness['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    if readiness['overall_readiness'] == 'raw_processing_needed':
        print(f"\n🚀 可執行自動處理:")
        print(f"   loader = VDDataLoader()")
        print(f"   result = loader.auto_process_if_needed()")
    
    print(f"\n🎯 輸出檔案結構:")
    print(f"   data/processed/YYYY-MM-DD/")
    print(f"   ├── target_route_data.csv     # 目標路段數據")
    print(f"   ├── target_route_peak.csv     # 目標路段尖峰")
    print(f"   ├── target_route_offpeak.csv  # 目標路段離峰")
    print(f"   └── target_route_summary.json # 數據摘要")
    
    print(f"\n🔄 彈性處理特色:")
    print(f"   ✅ 自動檢測檔案數量（不限2880個）")
    print(f"   ✅ 分段處理防止記憶體溢出")
    print(f"   ✅ 專注目標路段（圓山、台北、三重）")
    print(f"   ✅ 標準化輸出格式")
    print(f"   ✅ 原檔名歸檔保存")
    print(f"   ✅ 積極記憶體清理")
    
    print(f"\n🚀 Ready for Flexible Processing! 🚀")