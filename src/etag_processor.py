# src/etag_processor.py - 精簡高效版

"""
eTag數據處理器 - 精簡高效版
========================

核心功能：
1. 🎯 準確解析XML，處理目標路段（圓山-台北-三重）
2. 📊 詳細提取Flow數據（包含VehicleType分類）
3. 📁 自動歸檔處理完成的檔案
4. 💾 輸出符合VD融合格式的CSV數據

檔案路徑：data/raw/etag/ETag_Data_YYYYMMDD/ETagPairLive_HHMM.xml.gz
輸出格式：data/processed/etag/YYYY-MM-DD/etag_travel_time.csv

作者: 交通預測專案團隊
"""

import xml.etree.ElementTree as ET
import pandas as pd
import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


class ETagProcessor:
    """eTag數據處理器 - 精簡版"""
    
    def __init__(self, base_folder: str = "data", debug: bool = True):
        self.base_folder = Path(base_folder)
        self.raw_etag_folder = self.base_folder / "raw" / "etag"
        self.processed_etag_folder = self.base_folder / "processed" / "etag"
        self.archive_folder = self.base_folder / "archive" / "etag"
        self.debug = debug
        
        # 創建必要資料夾
        for folder in [self.processed_etag_folder, self.archive_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # 目標路段配對 - 根據實際XML調整（移除不存在的配對）
        self.target_pairs = {
            '01F0017N-01F0005N': {'segment': '台北→圓山', 'distance': 1.8, 'direction': 'N'},
            '01F0005S-01F0017S': {'segment': '圓山→台北', 'distance': 1.8, 'direction': 'S'},
            '01F0029N-01F0017N': {'segment': '三重→台北', 'distance': 2.0, 'direction': 'N'},
            '01F0017S-01F0029S': {'segment': '台北→三重', 'distance': 2.0, 'direction': 'S'}
            # 註：01F0029N-01F0005N 和 01F0005S-01F0029S 在實際XML中不存在
        }
        
        # XML命名空間
        self.namespace = {'ns': 'http://traffic.transportdata.tw/standard/traffic/schema/'}
        
        if self.debug:
            print(f"🏷️ eTag處理器初始化 (修正版)")
            print(f"   📁 原始數據: {self.raw_etag_folder}")
            print(f"   🎯 目標配對: {len(self.target_pairs)} 個 (實際存在)")
            print(f"   📝 註：移除XML中不存在的三重↔圓山配對")
    
    def scan_date_folders(self) -> Dict[str, List[Path]]:
        """掃描ETag_Data_YYYYMMDD格式的資料夾"""
        date_files = {}
        
        if not self.raw_etag_folder.exists():
            return date_files
        
        # 掃描 ETag_Data_YYYYMMDD 格式的資料夾
        for date_folder in self.raw_etag_folder.glob("ETag_Data_*"):
            if not date_folder.is_dir():
                continue
            
            # 提取日期 (ETag_Data_20250621 -> 2025-06-21)
            folder_name = date_folder.name
            if folder_name.startswith('ETag_Data_') and len(folder_name) == 18:
                date_part = folder_name[10:]  # 20250621
                if date_part.isdigit() and len(date_part) == 8:
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    
                    # 收集ETagPairLive_HHMM.xml.gz檔案
                    etag_files = list(date_folder.glob("ETagPairLive_*.xml.gz"))
                    if etag_files:
                        date_files[date_str] = etag_files
                        if self.debug:
                            print(f"   📁 {date_str}: {len(etag_files)} 檔案")
        
        if self.debug:
            total_files = sum(len(files) for files in date_files.values())
            print(f"   📊 總計: {len(date_files)} 日期, {total_files} 檔案")
        
        return date_files
    
    def process_single_file(self, file_path: Path, target_date: str) -> List[Dict[str, Any]]:
        """處理單一XML檔案"""
        try:
            # 讀取並解析XML
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                content = f.read()
            
            root = ET.fromstring(content)
            
            # 提取UpdateTime（優先使用資料夾日期）
            update_time = self._extract_update_time(root, file_path, target_date)
            
            # 寬鬆的日期檢查：允許±1天的時間差異
            target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            time_diff = abs((update_time.date() - target_date_obj.date()).days)
            
            if time_diff > 1:  # 超過1天差異才過濾
                if self.debug:
                    print(f"      ⚠️ 時間差異過大: XML={update_time.date()}, 目標={target_date}, 差異={time_diff}天")
                return []
            
            # 使用資料夾日期作為主要時間（更可靠）
            folder_date = datetime.strptime(target_date, '%Y-%m-%d')
            update_time = folder_date.replace(
                hour=update_time.hour, 
                minute=update_time.minute,
                second=update_time.second
            )
            
            # 提取目標路段數據
            target_data = []
            
            # 查找所有ETagPairLive
            etag_pairs = root.findall('.//ns:ETagPairLive', self.namespace)
            if not etag_pairs:  # 備援查找
                etag_pairs = root.findall('.//ETagPairLive')
            
            if self.debug:
                print(f"      🔍 找到 {len(etag_pairs)} 個ETagPairLive")
                # 快速掃描所有配對ID（僅debug模式）
                if len(etag_pairs) > 0:
                    sample_pair = etag_pairs[0]
                    pair_id_elem = sample_pair.find('ns:ETagPairId', self.namespace)
                    if pair_id_elem is None:
                        pair_id_elem = sample_pair.find('ETagPairId')
                    if pair_id_elem is not None and pair_id_elem.text:
                        print(f"      📋 示例配對: {pair_id_elem.text.strip()}")
            
            for etag_pair in etag_pairs:
                # 提取ETagPairId
                pair_id_elem = etag_pair.find('ns:ETagPairId', self.namespace)
                if pair_id_elem is None:
                    pair_id_elem = etag_pair.find('ETagPairId')
                
                if pair_id_elem is None or not pair_id_elem.text:
                    continue
                
                pair_id = pair_id_elem.text.strip()
                
                # 檢查是否為目標路段
                if pair_id not in self.target_pairs:
                    continue
                
                if self.debug:
                    print(f"      ✅ 處理目標配對: {pair_id}")
                
                # 提取Flows
                flows_elem = etag_pair.find('ns:Flows', self.namespace)
                if flows_elem is None:
                    flows_elem = etag_pair.find('Flows')
                
                if flows_elem is not None:
                    flows = flows_elem.findall('ns:Flow', self.namespace)
                    if not flows:
                        flows = flows_elem.findall('Flow')
                    
                    flow_count = 0
                    for flow in flows:
                        flow_data = self._extract_flow_data(flow, pair_id, update_time)
                        if flow_data:
                            target_data.append(flow_data)
                            flow_count += 1
                    
                    if self.debug:
                        print(f"         📊 提取 {flow_count} 個Flow記錄")
            
            if self.debug and target_data:
                print(f"      📊 總提取: {len(target_data)} 筆記錄")
            elif self.debug:
                print(f"      ⚠️ 沒有提取到任何目標記錄")
            
            return target_data
            
        except Exception as e:
            if self.debug:
                print(f"      ❌ 處理失敗 {file_path.name}: {e}")
            return []
    
    def _extract_update_time(self, root, file_path: Path, target_date: str) -> datetime:
        """提取更新時間 - 優先使用檔案名時間"""
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        
        # 方法1: 從檔案名ETagPairLive_HHMM.xml.gz提取時間（最可靠）
        filename = file_path.stem.replace('.xml', '')  # ETagPairLive_1620
        if 'ETagPairLive_' in filename:
            time_part = filename.split('_')[-1]  # 1620
            if len(time_part) == 4 and time_part.isdigit():
                hour = int(time_part[:2])
                minute = int(time_part[2:])
                return target_date_obj.replace(hour=hour, minute=minute)
        
        # 方法2: 從XML的UpdateTime提取時間部分
        update_elem = root.find('ns:UpdateTime', self.namespace)
        if update_elem is None:
            update_elem = root.find('UpdateTime')
        
        if update_elem is not None and update_elem.text:
            try:
                time_str = update_elem.text.replace('+08:00', '').replace('Z', '')
                parsed_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                # 只使用時間部分，日期使用資料夾日期
                return target_date_obj.replace(
                    hour=parsed_time.hour, 
                    minute=parsed_time.minute,
                    second=parsed_time.second
                )
            except Exception as e:
                if self.debug:
                    print(f"      ⚠️ XML時間解析失敗: {e}")
        
        # 方法3: 預設使用中午12:00
        return target_date_obj.replace(hour=12, minute=0)
    
    def _extract_flow_data(self, flow, pair_id: str, update_time: datetime) -> Dict[str, Any]:
        """提取Flow數據 - 包含所有車種"""
        try:
            # 提取基本數據
            vehicle_type = self._get_text(flow, 'VehicleType', '31')
            travel_time = int(self._get_text(flow, 'TravelTime', '0'))
            vehicle_count = int(self._get_text(flow, 'VehicleCount', '0'))
            space_mean_speed = float(self._get_text(flow, 'SpaceMeanSpeed', '0'))
            std_deviation = float(self._get_text(flow, 'StandardDeviation', '0'))
            
            # 保留所有Flow記錄（包括TravelTime=0的記錄）
            # 這些記錄在後續分析中有重要意義
            
            pair_info = self.target_pairs[pair_id]
            
            # 車種對應表
            vehicle_type_mapping = {
                '31': '小客車', '32': '小貨車',
                '41': '大客車', '42': '大貨車',
                '5': '聯結車'
            }
            
            return {
                'update_time': update_time,
                'etag_pair_id': pair_id,
                'vehicle_type_code': vehicle_type,
                'vehicle_type_name': vehicle_type_mapping.get(vehicle_type, f'未知({vehicle_type})'),
                'travel_time_seconds': travel_time,
                'travel_time_minutes': round(travel_time / 60, 2) if travel_time > 0 else 0,
                'vehicle_count': vehicle_count,
                'space_mean_speed_kmh': space_mean_speed,
                'standard_deviation': std_deviation,
                'segment_name': pair_info['segment'],
                'direction': pair_info['direction'],
                'distance_km': pair_info['distance'],
                'data_valid': 1 if travel_time > 0 and vehicle_count > 0 else 0
            }
            
        except Exception as e:
            if self.debug:
                print(f"         ❌ Flow提取失敗: {e}")
            return None
    
    def _get_text(self, element, tag: str, default: str = "") -> str:
        """安全獲取元素文本"""
        # 命名空間方式
        child = element.find(f'ns:{tag}', self.namespace)
        if child is not None and child.text:
            return child.text.strip()
        
        # 直接方式
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        
        return default
    
    def process_date_folder(self, date_str: str, file_list: List[Path]) -> bool:
        """處理單日期資料夾"""
        if self.debug:
            print(f"📅 處理 {date_str}: {len(file_list)} 檔案")
        
        all_data = []
        processed_files = 0
        skipped_files = 0
        
        # 處理所有檔案
        for i, file_path in enumerate(file_list):
            if self.debug and (i + 1) % 50 == 0:
                print(f"      進度: {i+1}/{len(file_list)}")
            
            file_data = self.process_single_file(file_path, date_str)
            if file_data:
                all_data.extend(file_data)
                processed_files += 1
            else:
                skipped_files += 1
        
        if self.debug:
            print(f"   📊 檔案處理結果:")
            print(f"      有效檔案: {processed_files}/{len(file_list)}")
            print(f"      跳過檔案: {skipped_files}")
        
        if not all_data:
            if self.debug:
                print(f"   ⚠️ {date_str}: 無目標路段數據")
                print(f"   💡 可能原因: XML內時間與資料夾日期不匹配")
            return False
        
        # 時間分析
        times = [record['update_time'] for record in all_data]
        time_span = (max(times) - min(times)).total_seconds() / 3600
        valid_records = sum(1 for record in all_data if record['data_valid'] == 1)
        
        # 檢查目標路段分布
        pair_counts = {}
        for record in all_data:
            pair_id = record['etag_pair_id']
            if pair_id not in pair_counts:
                pair_counts[pair_id] = 0
            pair_counts[pair_id] += 1
        
        if self.debug:
            print(f"   📊 結果統計:")
            print(f"      總記錄數: {len(all_data):,}")
            print(f"      有效記錄: {valid_records:,} ({valid_records/len(all_data)*100:.1f}%)")
            print(f"      時間跨度: {time_span:.1f} 小時")
            print(f"      時間範圍: {min(times)} ~ {max(times)}")
            print(f"   🎯 目標路段分布:")
            for pair_id, count in pair_counts.items():
                segment_name = self.target_pairs[pair_id]['segment']
                print(f"      {pair_id} ({segment_name}): {count:,} 記錄")
        
        # 保存數據
        self._save_date_data(date_str, all_data)
        
        # 歸檔檔案
        self._archive_date_files(date_str, file_list)
        
        return True
    
    def _save_date_data(self, date_str: str, data_list: List[Dict[str, Any]]):
        """保存日期數據"""
        date_folder = self.processed_etag_folder / date_str
        date_folder.mkdir(parents=True, exist_ok=True)
        
        # 轉換為DataFrame
        df = pd.DataFrame(data_list)
        
        # 1. 完整旅行時間數據（所有Flow記錄）
        travel_time_csv = date_folder / "etag_travel_time.csv"
        df.to_csv(travel_time_csv, index=False, encoding='utf-8-sig')
        
        # 2. 有效數據摘要（僅非零記錄）
        valid_df = df[df['data_valid'] == 1]
        if not valid_df.empty:
            valid_csv = date_folder / "etag_valid_data.csv"
            valid_df.to_csv(valid_csv, index=False, encoding='utf-8-sig')
        
        # 3. 按車種統計
        vehicle_stats = df.groupby(['etag_pair_id', 'vehicle_type_name']).agg({
            'vehicle_count': 'sum',
            'travel_time_seconds': 'mean',
            'space_mean_speed_kmh': 'mean',
            'data_valid': 'sum'
        }).reset_index()
        
        vehicle_csv = date_folder / "etag_vehicle_stats.csv"
        vehicle_stats.to_csv(vehicle_csv, index=False, encoding='utf-8-sig')
        
        # 4. 統計摘要
        summary = self._create_summary(df, date_str)
        summary_json = date_folder / "etag_summary.json"
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        if self.debug:
            print(f"   💾 保存完成: {travel_time_csv}")
    
    def _create_summary(self, df: pd.DataFrame, date_str: str) -> Dict[str, Any]:
        """創建統計摘要"""
        times = pd.to_datetime(df['update_time'])
        valid_df = df[df['data_valid'] == 1]
        
        return {
            'date': date_str,
            'total_records': len(df),
            'valid_records': len(valid_df),
            'validity_rate': len(valid_df) / len(df) * 100 if len(df) > 0 else 0,
            'unique_pairs': df['etag_pair_id'].nunique(),
            'time_range': {
                'start': times.min().isoformat(),
                'end': times.max().isoformat(),
                'span_hours': (times.max() - times.min()).total_seconds() / 3600
            },
            'vehicle_type_distribution': df['vehicle_type_name'].value_counts().to_dict(),
            'pair_statistics': df.groupby('etag_pair_id').agg({
                'vehicle_count': 'sum',
                'data_valid': 'sum'
            }).to_dict(),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _archive_date_files(self, date_str: str, file_list: List[Path]):
        """歸檔處理完的檔案"""
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
                    shutil.move(str(file_path), str(archive_path))
                    archived_count += 1
            except Exception as e:
                if self.debug:
                    print(f"      ⚠️ 歸檔失敗 {file_path.name}: {e}")
        
        if self.debug:
            print(f"   📦 歸檔: {archived_count}/{len(file_list)} 檔案")
        
        # 清理空的原始資料夾
        if archived_count > 0:
            try:
                original_folder = file_list[0].parent
                if original_folder.exists() and not any(original_folder.iterdir()):
                    original_folder.rmdir()
                    if self.debug:
                        print(f"   🗑️ 清理空資料夾: {original_folder.name}")
            except Exception as e:
                if self.debug:
                    print(f"   ⚠️ 清理資料夾失敗: {e}")
    
    def process_all_dates(self) -> Dict[str, Any]:
        """批次處理所有日期"""
        if self.debug:
            print("🚀 批次處理eTag數據")
            print("=" * 40)
        
        date_files = self.scan_date_folders()
        if not date_files:
            return {"success": False, "message": "無eTag檔案"}
        
        successful_dates = 0
        total_records = 0
        results = {}
        
        for date_str, file_list in date_files.items():
            success = self.process_date_folder(date_str, file_list)
            results[date_str] = success
            
            if success:
                successful_dates += 1
                # 統計記錄數
                summary_file = self.processed_etag_folder / date_str / "etag_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        total_records += summary.get('total_records', 0)
                    except:
                        pass
        
        if self.debug:
            print(f"\n🏁 批次處理完成:")
            print(f"   成功: {successful_dates}/{len(date_files)} 日期")
            print(f"   記錄: {total_records:,} 筆")
        
        return {
            "success": True,
            "total_dates": len(date_files),
            "successful_dates": successful_dates,
            "total_records": total_records,
            "results": results
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """獲取處理摘要"""
        if not self.processed_etag_folder.exists():
            return {"processed_dates": 0, "total_records": 0}
        
        date_folders = [d for d in self.processed_etag_folder.iterdir() 
                       if d.is_dir() and d.name.count('-') == 2]
        
        total_records = 0
        date_details = {}
        
        for date_folder in date_folders:
            summary_file = date_folder / "etag_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    date_details[date_folder.name] = {
                        "total_records": summary.get('total_records', 0),
                        "valid_records": summary.get('valid_records', 0),
                        "validity_rate": summary.get('validity_rate', 0),
                        "time_span": summary.get('time_range', {}).get('span_hours', 0)
                    }
                    total_records += summary.get('total_records', 0)
                except:
                    pass
        
        return {
            "processed_dates": len(date_folders),
            "total_records": total_records,
            "date_details": date_details
        }


# 便利函數
def process_etag_data(base_folder: str = "data", debug: bool = True) -> Dict[str, Any]:
    """一鍵處理eTag數據"""
    processor = ETagProcessor(base_folder, debug)
    return processor.process_all_dates()


def get_etag_summary(base_folder: str = "data") -> Dict[str, Any]:
    """獲取eTag處理摘要"""
    processor = ETagProcessor(base_folder, debug=False)
    return processor.get_processing_summary()


if __name__ == "__main__":
    print("🏷️ eTag處理器 - 精簡高效版")
    print("=" * 50)
    print("🎯 目標路段: 圓山(23K)-台北(25K)-三重(27K)")
    print("📁 檔案格式: ETag_Data_YYYYMMDD/ETagPairLive_HHMM.xml.gz")
    print("=" * 50)
    
    processor = ETagProcessor(debug=True)
    result = processor.process_all_dates()
    
    if result["success"]:
        print(f"\n🎉 處理完成！")
        summary = processor.get_processing_summary()
        
        print(f"\n📊 處理摘要:")
        for date_str, details in summary['date_details'].items():
            valid_rate = details['validity_rate']
            time_span = details['time_span']
            total = details['total_records']
            valid = details['valid_records']
            print(f"   {date_str}: {total:,} 記錄 ({valid:,} 有效, {valid_rate:.1f}%), {time_span:.1f}h")
        
        print(f"\n📁 輸出位置:")
        print(f"   data/processed/etag/YYYY-MM-DD/")
        print(f"   ├── etag_travel_time.csv      # 完整數據")
        print(f"   ├── etag_valid_data.csv       # 有效數據") 
        print(f"   ├── etag_vehicle_stats.csv    # 車種統計")
        print(f"   └── etag_summary.json         # 統計摘要")
        
        print(f"\n📦 歸檔位置:")
        print(f"   data/archive/etag/YYYY-MM-DD/")
    else:
        print(f"\n❌ 處理失敗: {result.get('message', '未知錯誤')}")