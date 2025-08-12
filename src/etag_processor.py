# src/etag_processor.py - 簡化修正版

"""
eTag數據處理器 - 簡化修正版
========================

核心功能：
1. 🕐 嚴格按資料夾日期篩選XML內容
2. 🎯 目標路段篩選（圓山-台北-三重）
3. 📊 生成旅行時間和流量數據
4. 🔧 簡潔高效的代碼結構

作者: 交通預測專案團隊
"""

import xml.etree.ElementTree as ET
import pandas as pd
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


class ETagProcessor:
    """eTag數據處理器"""
    
    def __init__(self, base_folder: str = "data", debug: bool = True):
        self.base_folder = Path(base_folder)
        self.raw_etag_folder = self.base_folder / "raw" / "etag"
        self.processed_etag_folder = self.base_folder / "processed" / "etag"
        self.archive_folder = self.base_folder / "archive" / "etag"  # 添加這一行
        self.debug = debug
        
        # 創建資料夾
        for folder in [self.processed_etag_folder, self.archive_folder]:  # 修正這一行
            folder.mkdir(parents=True, exist_ok=True)
        
        # 目標路段配對（圓山-台北-三重）
        self.target_pairs = {
            '01F0017N-01F0005N': {'segment': '台北→圓山', 'distance': 1.8},
            '01F0005S-01F0017S': {'segment': '圓山→台北', 'distance': 1.8},
            '01F0029N-01F0017N': {'segment': '三重→台北', 'distance': 2.0},
            '01F0017S-01F0029S': {'segment': '台北→三重', 'distance': 2.0},
            '01F0029N-01F0005N': {'segment': '三重→圓山', 'distance': 3.8},
            '01F0005S-01F0029S': {'segment': '圓山→三重', 'distance': 3.8}
        }
        
        if self.debug:
            print(f"🏷️ eTag處理器初始化 (修正版)")
            print(f"   📁 原始數據: {self.raw_etag_folder}")
            print(f"   🎯 目標配對: {len(self.target_pairs)} 個")
    
    def scan_date_folders(self) -> Dict[str, List[Path]]:
        """掃描按日期分類的檔案"""
        date_files = {}
        
        if not self.raw_etag_folder.exists():
            return date_files
        
        # 掃描日期資料夾
        for date_folder in self.raw_etag_folder.iterdir():
            if not date_folder.is_dir():
                continue
            
            # 提取日期
            date_str = self._extract_date_from_folder(date_folder.name)
            if not date_str:
                continue
            
            # 收集該日期的eTag檔案 - 擴展檔案匹配
            etag_files = []
            patterns = ["ETagPairLive_*.xml.gz", "*.xml.gz", "ETag*.xml.gz"]
            for pattern in patterns:
                files = list(date_folder.glob(pattern))
                etag_files.extend(files)
            
            # 去重
            etag_files = list(set(etag_files))
            
            if etag_files:
                date_files[date_str] = etag_files
                if self.debug:
                    print(f"   📁 {date_str}: {len(etag_files)} 檔案")
        
        if self.debug:
            total_files = sum(len(files) for files in date_files.values())
            print(f"   📊 總計: {len(date_files)} 日期, {total_files} 檔案")
        
        return date_files
    
    def _extract_date_from_folder(self, folder_name: str) -> str:
        """從資料夾名稱提取日期"""
        import re
        
        # YYYY-MM-DD格式
        match = re.search(r'(\d{4}-\d{2}-\d{2})', folder_name)
        if match:
            return match.group(1)
        
        # YYYYMMDD格式
        match = re.search(r'(\d{8})', folder_name)
        if match:
            date_str = match.group(1)
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return None
    
    def process_single_file(self, file_path: Path, target_date: str) -> List[Dict[str, Any]]:
        """處理單一檔案 - 修正版"""
        try:
            # 解壓並讀取XML
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) < 100:
                if self.debug:
                    print(f"   ⚠️ 檔案內容太短: {file_path.name}")
                return []
            
            root = ET.fromstring(content)
            
            if self.debug:
                print(f"   🔍 解析 {file_path.name}")
            
            # 提取更新時間
            update_time = self._extract_update_time(root, file_path, target_date)
            
            # 放寬時間檢查 - 允許相鄰日期
            target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            date_diff = abs((update_time.date() - target_date_obj.date()).days)
            
            if date_diff > 1:
                if self.debug:
                    print(f"      ⚠️ 時間差距過大: {update_time} vs {target_date}")
                return []
            
            # 提取目標路段數據
            target_data = []
            etag_pairs = self._find_etag_pairs_enhanced(root)
            
            if self.debug:
                print(f"      🎯 找到 {len(etag_pairs)} 個ETagPairLive")
            
            for etag_pair in etag_pairs:
                pair_id = self._get_text_enhanced(etag_pair, 'ETagPairId')
                
                if self.debug:
                    print(f"         檢查配對: {pair_id}")
                
                if not pair_id or pair_id not in self.target_pairs:
                    continue
                
                if self.debug:
                    print(f"         ✅ 目標配對: {pair_id}")
                
                # 查找Flows容器 - 修正版
                flows_containers = []
                
                # 方法1: 直接查找Flows
                flows_elem = etag_pair.find('Flows')
                if flows_elem is not None:
                    flows_containers.append(flows_elem)
                
                # 方法2: 深度搜尋Flows
                for elem in etag_pair.iter():
                    if elem.tag == 'Flows' and elem not in flows_containers:
                        flows_containers.append(elem)
                
                if self.debug:
                    print(f"            找到 {len(flows_containers)} 個Flows容器")
                
                for flows_container in flows_containers:
                    flows = flows_container.findall('Flow')
                    if self.debug:
                        print(f"               容器內有 {len(flows)} 個Flow")
                    
                    for flow in flows:
                        flow_data = self._extract_flow_data(flow, pair_id, update_time)
                        if flow_data:
                            target_data.append(flow_data)
                            if self.debug:
                                print(f"               ✅ 有效Flow: 車種={flow_data['vehicle_type']}, 時間={flow_data['travel_time']}s")
                
                # 如果沒找到Flows容器，直接搜尋Flow
                if not flows_containers:
                    if self.debug:
                        print(f"            備用：直接搜尋Flow元素")
                    
                    for elem in etag_pair.iter():
                        if elem.tag == 'Flow':
                            flow_data = self._extract_flow_data(elem, pair_id, update_time)
                            if flow_data:
                                target_data.append(flow_data)
                                if self.debug:
                                    print(f"               ✅ 備用Flow: 車種={flow_data['vehicle_type']}")
            
            if self.debug:
                print(f"      📊 總提取記錄: {len(target_data)}")
            
            return target_data
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ 處理失敗 {file_path.name}: {e}")
            return []
    
    def _find_etag_pairs_enhanced(self, root):
        """增強版ETagPair查找"""
        etag_pairs = []
        
        # 方法1: 標準命名空間
        try:
            pairs = root.findall('.//ETagPairLive')
            etag_pairs.extend(pairs)
        except:
            pass
        
        # 方法2: 遍歷所有元素查找
        for elem in root.iter():
            if 'ETagPair' in elem.tag and 'Live' in elem.tag:
                if elem not in etag_pairs:
                    etag_pairs.append(elem)
        
        return etag_pairs
    
    def _find_flows_enhanced(self, etag_pair):
        """增強版Flow查找"""
        flows = []
        
        # 查找Flows容器
        flows_containers = etag_pair.findall('.//Flows')
        for container in flows_containers:
            flows.extend(container.findall('Flow'))
        
        # 直接查找Flow
        flows.extend(etag_pair.findall('.//Flow'))
        
        return flows
    
    def _get_text_enhanced(self, element, tag: str) -> str:
        """增強版文本提取"""
        # 方法1: 直接查找
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        
        # 方法2: 遍歷查找
        for child in element:
            if tag in child.tag and child.text:
                return child.text.strip()
        
        return ""
    
    def _extract_update_time(self, root, file_path: Path, target_date: str) -> datetime:
        """提取更新時間 - 修正版"""
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        
        # 優先順序：DataCollectTime -> EndTime -> StartTime -> UpdateTime -> 檔案名
        time_fields = ['DataCollectTime', 'EndTime', 'StartTime', 'UpdateTime']
        
        for field in time_fields:
            for elem in root.iter():
                if field in elem.tag and elem.text:
                    try:
                        time_str = elem.text.replace('+08:00', '').replace('Z', '')
                        parsed_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                        
                        if self.debug:
                            print(f"      🕐 解析 {field}: {parsed_time}")
                        
                        # 檢查日期是否符合 - 放寬條件
                        date_diff = abs((parsed_time.date() - target_date_obj.date()).days)
                        if date_diff <= 1:  # 允許1天誤差
                            return parsed_time
                        
                    except Exception as e:
                        if self.debug:
                            print(f"      ⚠️ {field} 解析失敗: {e}")
                        continue
        
        # 從檔案名提取時間
        try:
            filename = file_path.stem.replace('.xml', '')
            if 'ETagPairLive_' in filename:
                time_part = filename.replace('ETagPairLive_', '')
                if len(time_part) == 4 and time_part.isdigit():
                    hour = int(time_part[:2])
                    minute = int(time_part[2:])
                    result_time = target_date_obj.replace(hour=hour, minute=minute)
                    if self.debug:
                        print(f"      🕐 檔案名時間: {result_time}")
                    return result_time
        except:
            pass
        
        # 預設使用目標日期的12:00
        default_time = target_date_obj.replace(hour=12, minute=0)
        if self.debug:
            print(f"      🕐 預設時間: {default_time}")
        return default_time
    
    def _get_text(self, element, tag: str) -> str:
        """安全獲取元素文本"""
        return self._get_text_enhanced(element, tag)
    
    def _extract_flow_data(self, flow, pair_id: str, update_time: datetime) -> Dict[str, Any]:
        """提取Flow數據 - 修正版"""
        try:
            travel_time = int(self._get_text_enhanced(flow, 'TravelTime') or '0')
            vehicle_count = int(self._get_text_enhanced(flow, 'VehicleCount') or '0')
            space_mean_speed = float(self._get_text_enhanced(flow, 'SpaceMeanSpeed') or '0')
            vehicle_type = self._get_text_enhanced(flow, 'VehicleType') or '31'
            standard_deviation = float(self._get_text_enhanced(flow, 'StandardDeviation') or '0')
            
            # 放寬條件：只要有TravelTime且VehicleCount > 0就接受
            # 這樣可以處理SpaceMeanSpeed為0的情況
            if travel_time <= 0 or vehicle_count <= 0:
                return None
            
            # 如果SpaceMeanSpeed為0，從distance和travel_time計算
            if space_mean_speed <= 0 and travel_time > 0:
                pair_info = self.target_pairs[pair_id]
                distance_km = pair_info['distance']
                travel_time_hours = travel_time / 3600
                space_mean_speed = distance_km / travel_time_hours if travel_time_hours > 0 else 0
            
            if self.debug:
                print(f"            Flow: 類型={vehicle_type}, 時間={travel_time}s, 數量={vehicle_count}, 速度={space_mean_speed:.1f}")
            
            pair_info = self.target_pairs[pair_id]
            
            return {
                'update_time': update_time,
                'etag_pair_id': pair_id,
                'vehicle_type': vehicle_type,
                'travel_time': travel_time,
                'vehicle_count': vehicle_count,
                'space_mean_speed': round(space_mean_speed, 1),
                'standard_deviation': standard_deviation,
                'travel_time_minutes': round(travel_time / 60, 2),
                'segment_name': pair_info['segment']
            }
            
        except Exception as e:
            if self.debug:
                print(f"            ❌ Flow提取失敗: {e}")
            return None
    
    def process_date_folder(self, date_str: str, file_list: List[Path]) -> bool:
        """處理單日期資料夾"""
        if self.debug:
            print(f"📅 處理 {date_str}: {len(file_list)} 檔案")
        
        all_data = []
        processed_files = 0
        
        for i, file_path in enumerate(file_list):
            if self.debug and i % 50 == 0:
                print(f"   進度: {i+1}/{len(file_list)}")
            
            file_data = self.process_single_file(file_path, date_str)
            if file_data:
                all_data.extend(file_data)
                processed_files += 1
        
        if self.debug:
            print(f"   📊 處理完成: {processed_files}/{len(file_list)} 檔案有效")
        
        if not all_data:
            if self.debug:
                print(f"   ⚠️ {date_str}: 無目標路段數據")
            return False
        
        # 檢查時間分布
        times = [record['update_time'] for record in all_data]
        time_span = (max(times) - min(times)).total_seconds() / 3600
        
        if self.debug:
            print(f"   📊 {date_str}: {len(all_data)} 記錄, 時間跨度: {time_span:.1f}h")
            print(f"       時間範圍: {min(times)} ~ {max(times)}")
        
        # 保存數據
        self._save_date_data(date_str, all_data)
        
        # 歸檔原始檔案
        self._archive_date_files(date_str, file_list)
        
        return True
    
    def _archive_date_files(self, date_str: str, file_list: List[Path]):
        """歸檔日期檔案"""
        if not file_list:
            return
        
        # 創建歸檔資料夾
        archive_date_folder = self.archive_folder / date_str
        archive_date_folder.mkdir(parents=True, exist_ok=True)
        
        archived_count = 0
        
        for file_path in file_list:
            try:
                if file_path.exists():
                    # 移動到歸檔
                    archive_path = archive_date_folder / file_path.name
                    file_path.rename(archive_path)
                    archived_count += 1
            except Exception as e:
                if self.debug:
                    print(f"   ⚠️ 歸檔失敗 {file_path.name}: {e}")
        
        if self.debug and archived_count > 0:
            print(f"   📦 歸檔: {archived_count} 檔案至 {archive_date_folder}")
        
        # 移除空的原始資料夾
        try:
            original_folder = file_list[0].parent
            if original_folder.exists() and not any(original_folder.iterdir()):
                original_folder.rmdir()
                if self.debug:
                    print(f"   🗑️ 移除空資料夾: {original_folder}")
        except:
            pass
    
    def _save_date_data(self, date_str: str, data_list: List[Dict[str, Any]]):
        """保存日期數據"""
        date_folder = self.processed_etag_folder / date_str
        date_folder.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(data_list)
        
        # 1. 旅行時間數據
        travel_time_csv = date_folder / "etag_travel_time.csv"
        df.to_csv(travel_time_csv, index=False, encoding='utf-8-sig')
        
        # 2. 摘要統計
        summary = {
            'date': date_str,
            'total_records': len(df),
            'unique_pairs': df['etag_pair_id'].nunique(),
            'time_range': {
                'start': df['update_time'].min().isoformat(),
                'end': df['update_time'].max().isoformat(),
                'span_hours': (df['update_time'].max() - df['update_time'].min()).total_seconds() / 3600
            }
        }
        
        summary_json = date_folder / "etag_summary.json"
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    
    def process_all_dates(self) -> Dict[str, Any]:
        """批次處理所有日期"""
        if self.debug:
            print("🚀 批次處理eTag數據")
        
        date_files = self.scan_date_folders()
        if not date_files:
            return {"success": False, "message": "無eTag檔案"}
        
        successful_dates = 0
        results = {}
        
        for date_str, file_list in date_files.items():
            success = self.process_date_folder(date_str, file_list)
            results[date_str] = success
            if success:
                successful_dates += 1
        
        return {
            "success": True,
            "total_dates": len(date_files),
            "successful_dates": successful_dates,
            "results": results
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """獲取處理摘要"""
        if not self.processed_etag_folder.exists():
            return {"processed_dates": 0, "total_records": 0}
        
        date_folders = [d for d in self.processed_etag_folder.iterdir() 
                       if d.is_dir() and d.name.count('-') == 2]
        
        total_records = 0
        for date_folder in date_folders:
            summary_file = date_folder / "etag_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    total_records += summary.get('total_records', 0)
                except:
                    pass
        
        return {
            "processed_dates": len(date_folders),
            "total_records": total_records
        }


# 便利函數
def process_etag_data(base_folder: str = "data", debug: bool = True) -> Dict[str, Any]:
    """處理eTag數據"""
    processor = ETagProcessor(base_folder, debug)
    return processor.process_all_dates()


def get_etag_summary(base_folder: str = "data") -> Dict[str, Any]:
    """獲取eTag摘要"""
    processor = ETagProcessor(base_folder, debug=False)
    return processor.get_processing_summary()


if __name__ == "__main__":
    print("🏷️ eTag處理器 - 修正版")
    processor = ETagProcessor(debug=True)
    result = processor.process_all_dates()
    if result["success"]:
        summary = processor.get_processing_summary()
        print(f"✅ 處理完成: {summary['processed_dates']} 日期, {summary['total_records']} 記錄")