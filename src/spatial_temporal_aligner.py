# src/spatial_temporal_aligner.py - 簡化修正版

"""
VD與eTag數據時空對齊器 - 簡化版
================================

核心功能：
1. 動態檢測可用資料天數
2. VD站點 ↔ eTag配對時空對齊
3. 智能資料同步 (1分鐘VD → 5分鐘eTag)
4. 對齊品質評估

修正重點：
- 簡化程式碼，移除冗餘功能
- 動態讀取實際可用資料
- 保留核心對齊邏輯和debug功能

作者: 交通預測專案團隊
日期: 2025-01-23 (簡化版)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SpatialTemporalAligner:
    """VD與eTag時空對齊器 - 簡化版"""
    
    def __init__(self, base_folder: str = "data", debug: bool = True):
        self.base_folder = Path(base_folder)
        self.debug = debug
        
        # 空間映射配置 - 基於實際eTag配對
        self.spatial_mapping = {
            'VD-N1-N-23-圓山': {
                'etag_pairs': ['01F0017N-01F0005N', '01F0005S-01F0017S'],
                'description': '圓山與台北間'
            },
            'VD-N1-N-25-台北': {
                'etag_pairs': ['01F0017N-01F0005N', '01F0005S-01F0017S', 
                              '01F0029N-01F0017N', '01F0017S-01F0029S'],
                'description': '台北樞紐路段'
            },
            'VD-N1-N-27-三重': {
                'etag_pairs': ['01F0029N-01F0017N', '01F0017S-01F0029S'],
                'description': '三重與台北間'
            }
        }
        
        # 時間同步參數
        self.time_window = 5  # 分鐘
        self.sync_tolerance = 2  # 分鐘容差
        
        if self.debug:
            print("🔗 VD+eTag時空對齊器初始化 (簡化版)")
            print(f"   📁 數據目錄: {self.base_folder}")
            print(f"   🗺️ 空間映射: {len(self.spatial_mapping)} 個VD站點")
    
    def get_available_dates(self) -> Dict[str, List[str]]:
        """動態檢測可用資料日期"""
        vd_dates = self._scan_vd_dates()
        etag_dates = self._scan_etag_dates()
        common_dates = sorted(set(vd_dates).intersection(set(etag_dates)))
        
        result = {
            'vd_dates': vd_dates,
            'etag_dates': etag_dates,
            'common_dates': common_dates
        }
        
        if self.debug:
            print(f"📅 可用資料檢測:")
            print(f"   VD資料: {len(vd_dates)} 天")
            print(f"   eTag資料: {len(etag_dates)} 天")
            print(f"   共同日期: {len(common_dates)} 天")
            for date in common_dates:
                print(f"     • {date}")
        
        return result
    
    def _scan_vd_dates(self) -> List[str]:
        """掃描VD數據日期"""
        dates = []
        search_paths = [
            self.base_folder / "processed" / "vd",
            self.base_folder / "cleaned"
        ]
        
        for folder in search_paths:
            if not folder.exists():
                continue
                
            for date_folder in folder.iterdir():
                if date_folder.is_dir() and date_folder.name.count('-') == 2:
                    target_files = [
                        date_folder / "target_route_data.csv",
                        date_folder / "target_route_data_cleaned.csv"
                    ]
                    if any(f.exists() for f in target_files):
                        dates.append(date_folder.name)
        
        return sorted(list(set(dates)))
    
    def _scan_etag_dates(self) -> List[str]:
        """掃描eTag數據日期"""
        dates = []
        etag_folder = self.base_folder / "processed" / "etag"
        
        if etag_folder.exists():
            for date_folder in etag_folder.iterdir():
                if date_folder.is_dir() and date_folder.name.count('-') == 2:
                    etag_files = [
                        date_folder / "etag_travel_time.csv",
                        date_folder / "etag_flow_data.csv"
                    ]
                    if any(f.exists() for f in etag_files):
                        dates.append(date_folder.name)
        
        return sorted(dates)
    
    def align_date_data(self, date_str: str) -> Dict:
        """對齊單日資料"""
        if self.debug:
            print(f"🔗 對齊 {date_str} 資料...")
        
        # 載入資料
        vd_data = self._load_vd_data(date_str)
        etag_data = self._load_etag_data(date_str)
        
        if vd_data.empty:
            return {'error': 'VD資料載入失敗'}
        if etag_data.empty:
            return {'error': 'eTag資料載入失敗'}
        
        if self.debug:
            print(f"   📊 VD資料: {len(vd_data):,} 筆")
            print(f"   📊 eTag資料: {len(etag_data):,} 筆")
        
        # 執行對齊
        aligned_data = self._perform_alignment(vd_data, etag_data)
        
        # 保存結果
        if aligned_data and 'aligned' in aligned_data:
            self._save_aligned_data(aligned_data, date_str)
        
        return aligned_data
    
    def _load_vd_data(self, date_str: str) -> pd.DataFrame:
        """載入VD資料"""
        vd_files = [
            self.base_folder / "processed" / "vd" / date_str / "target_route_data.csv",
            self.base_folder / "cleaned" / date_str / "target_route_data_cleaned.csv"
        ]
        
        for vd_file in vd_files:
            if vd_file.exists():
                try:
                    df = pd.read_csv(vd_file)
                    df['update_time'] = pd.to_datetime(df['update_time'])
                    if self.debug:
                        print(f"   📄 VD來源: {vd_file.name}")
                    return df
                except Exception as e:
                    if self.debug:
                        print(f"   ❌ VD載入失敗: {e}")
        
        return pd.DataFrame()
    
    def _load_etag_data(self, date_str: str) -> pd.DataFrame:
        """載入eTag資料"""
        etag_files = [
            self.base_folder / "processed" / "etag" / date_str / "etag_travel_time.csv",
            self.base_folder / "processed" / "etag" / date_str / "etag_flow_data.csv"
        ]
        
        for etag_file in etag_files:
            if etag_file.exists():
                try:
                    df = pd.read_csv(etag_file)
                    df['update_time'] = pd.to_datetime(df['update_time'])
                    if self.debug:
                        print(f"   📄 eTag來源: {etag_file.name}")
                    return df
                except Exception as e:
                    if self.debug:
                        print(f"   ❌ eTag載入失敗: {e}")
        
        return pd.DataFrame()
    
    def _perform_alignment(self, vd_data: pd.DataFrame, etag_data: pd.DataFrame) -> Dict:
        """執行時空對齊核心邏輯"""
        aligned_records = []
        
        if self.debug:
            print(f"   🎯 開始時空對齊...")
        
        for vd_station, mapping in self.spatial_mapping.items():
            # 匹配VD站點
            vd_subset = self._match_vd_station(vd_data, vd_station)
            if vd_subset.empty:
                continue
            
            # 處理eTag配對
            for etag_pair in mapping['etag_pairs']:
                etag_subset = etag_data[etag_data['etag_pair_id'] == etag_pair]
                if etag_subset.empty:
                    continue
                
                # 時間對齊
                aligned_pair = self._temporal_alignment(vd_subset, etag_subset, vd_station, etag_pair)
                aligned_records.extend(aligned_pair)
        
        if aligned_records:
            aligned_df = pd.DataFrame(aligned_records)
            if self.debug:
                print(f"   ✅ 總對齊記錄: {len(aligned_df)} 筆")
            
            return {
                'aligned': aligned_df,
                'summary': self._generate_summary(aligned_df)
            }
        else:
            return {'error': '沒有成功對齊的記錄'}
    
    def _match_vd_station(self, vd_data: pd.DataFrame, vd_station: str) -> pd.DataFrame:
        """匹配VD站點"""
        station_name = vd_station.split('-')[-1]  # 例如 "圓山"
        mask = vd_data['vd_id'].str.contains(station_name, na=False, regex=False)
        return vd_data[mask] if mask.any() else pd.DataFrame()
    
    def _temporal_alignment(self, vd_data: pd.DataFrame, etag_data: pd.DataFrame, 
                           vd_station: str, etag_pair: str) -> List[Dict]:
        """時間對齊：1分鐘VD → 5分鐘eTag"""
        aligned_records = []
        
        try:
            # VD資料聚合到5分鐘窗口
            vd_data = vd_data.copy()
            vd_data['time_window'] = vd_data['update_time'].dt.floor(f'{self.time_window}min')
            
            vd_aggregated = vd_data.groupby('time_window').agg({
                'speed': 'mean',
                'volume_total': 'sum',
                'occupancy': 'mean'
            }).reset_index()
            
            # 時間匹配
            for _, etag_row in etag_data.iterrows():
                etag_time = etag_row['update_time']
                
                # 找最接近的VD時間窗口
                time_diffs = abs((vd_aggregated['time_window'] - etag_time).dt.total_seconds() / 60)
                
                if len(time_diffs) == 0:
                    continue
                    
                min_diff_idx = time_diffs.idxmin()
                min_diff = time_diffs.iloc[min_diff_idx]
                
                if min_diff <= self.sync_tolerance:
                    vd_row = vd_aggregated.iloc[min_diff_idx]
                    
                    aligned_record = {
                        'update_time': etag_time,
                        'vd_station': vd_station,
                        'etag_pair': etag_pair,
                        'vd_speed': float(vd_row['speed']) if pd.notna(vd_row['speed']) else 0,
                        'vd_volume': int(vd_row['volume_total']) if pd.notna(vd_row['volume_total']) else 0,
                        'vd_occupancy': float(vd_row['occupancy']) if pd.notna(vd_row['occupancy']) else 0,
                        'etag_speed': float(etag_row.get('space_mean_speed', 0)) or 0,
                        'etag_volume': int(etag_row.get('vehicle_count', 0)) or 0,
                        'time_diff_minutes': float(min_diff)
                    }
                    
                    aligned_records.append(aligned_record)
        
        except Exception as e:
            if self.debug:
                print(f"   ❌ 對齊錯誤: {e}")
        
        return aligned_records
    
    def _generate_summary(self, aligned_df: pd.DataFrame) -> Dict:
        """生成對齊摘要"""
        return {
            'total_records': len(aligned_df),
            'vd_stations': int(aligned_df['vd_station'].nunique()),
            'etag_pairs': int(aligned_df['etag_pair'].nunique()),
            'speed_correlation': float(aligned_df['vd_speed'].corr(aligned_df['etag_speed'])),
            'sync_quality_percent': float((aligned_df['time_diff_minutes'] <= self.sync_tolerance).mean() * 100)
        }
    
    def _save_aligned_data(self, aligned_data: Dict, date_str: str):
        """保存對齊結果"""
        output_folder = self.base_folder / "processed" / "fusion" / date_str
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # 保存對齊數據
        aligned_df = aligned_data['aligned']
        aligned_csv = output_folder / "vd_etag_aligned.csv"
        aligned_df.to_csv(aligned_csv, index=False, encoding='utf-8-sig')
        
        # 保存摘要
        summary_json = output_folder / "alignment_summary.json"
        with open(summary_json, 'w', encoding='utf-8') as f:
            import json
            json.dump(aligned_data['summary'], f, ensure_ascii=False, indent=2, default=str)
        
        if self.debug:
            print(f"   💾 保存至: {aligned_csv}")
    
    def batch_align_all_available(self) -> Dict:
        """批次對齊所有可用資料"""
        available = self.get_available_dates()
        common_dates = available['common_dates']
        
        if not common_dates:
            return {'error': '沒有共同日期資料'}
        
        results = {}
        for date_str in common_dates:
            if self.debug:
                print(f"\n📅 處理 {date_str}...")
            result = self.align_date_data(date_str)
            results[date_str] = result
        
        successful = sum(1 for r in results.values() if 'aligned' in r)
        if self.debug:
            print(f"\n🏁 批次完成: {successful}/{len(common_dates)} 成功")
        
        return results
    
    def validate_alignment(self, date_str: str) -> Dict:
        """驗證對齊品質"""
        aligned_file = self.base_folder / "processed" / "fusion" / date_str / "vd_etag_aligned.csv"
        
        if not aligned_file.exists():
            return {'error': '對齊檔案不存在'}
        
        try:
            df = pd.read_csv(aligned_file)
            
            validation = {
                'record_count': len(df),
                'time_sync_quality': (df['time_diff_minutes'] <= self.sync_tolerance).mean() * 100,
                'speed_correlation': df['vd_speed'].corr(df['etag_speed']),
                'volume_correlation': df['vd_volume'].corr(df['etag_volume']),
                'data_completeness': df.notna().all(axis=1).mean() * 100
            }
            
            if self.debug:
                print(f"📊 {date_str} 驗證結果:")
                print(f"   記錄數: {validation['record_count']:,}")
                print(f"   時間同步品質: {validation['time_sync_quality']:.1f}%")
                print(f"   速度相關性: {validation['speed_correlation']:.3f}")
            
            return validation
            
        except Exception as e:
            return {'error': f'驗證失敗: {str(e)}'}


# 便利函數
def align_all_available_data(debug: bool = True) -> Dict:
    """對齊所有可用資料"""
    aligner = SpatialTemporalAligner(debug=debug)
    return aligner.batch_align_all_available()


def get_available_data_status(debug: bool = True) -> Dict:
    """獲取可用資料狀態"""
    aligner = SpatialTemporalAligner(debug=debug)
    return aligner.get_available_dates()


if __name__ == "__main__":
    print("🔗 VD+eTag時空對齊模組 (簡化版)")
    print("=" * 50)
    
    aligner = SpatialTemporalAligner(debug=True)
    
    # 檢查可用資料
    available = aligner.get_available_dates()
    
    if available['common_dates']:
        print(f"\n🎯 開始批次對齊 {len(available['common_dates'])} 天資料...")
        results = aligner.batch_align_all_available()
        
        # 驗證結果
        for date_str in available['common_dates']:
            if 'aligned' in results.get(date_str, {}):
                validation = aligner.validate_alignment(date_str)
                if 'error' not in validation:
                    print(f"✅ {date_str}: 品質評分 {validation['speed_correlation']:.3f}")
    else:
        print("\n⚠️ 沒有可用的共同日期資料")