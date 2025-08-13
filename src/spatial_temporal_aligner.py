# src/spatial_temporal_aligner.py - 簡化版

"""
VD與eTag數據時空對齊器 - 簡化版
================================

核心功能：
1. VD(1分鐘) ↔ eTag(5分鐘) 時間對齊
2. 圓山-台北-三重路段空間映射
3. 數據品質驗證與保存

簡化原則：
- 移除冗餘功能，保留核心邏輯
- 動態檢測可用資料
- 清晰的錯誤處理

作者: 交通預測專案團隊
日期: 2025-01-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class SpatialTemporalAligner:
    """VD與eTag時空對齊器 - 簡化版"""
    
    def __init__(self, base_folder: str = "data", debug: bool = False):
        self.base_folder = Path(base_folder)
        self.debug = debug
        
        # 空間映射：VD站點 ↔ eTag配對
        self.spatial_mapping = {
            '圓山': ['01F0017N-01F0005N', '01F0005S-01F0017S'],
            '台北': ['01F0017N-01F0005N', '01F0005S-01F0017S', 
                    '01F0029N-01F0017N', '01F0017S-01F0029S'],
            '三重': ['01F0029N-01F0017N', '01F0017S-01F0029S']
        }
        
        # 時間同步參數
        self.time_window = 5  # 分鐘
        self.sync_tolerance = 2  # 容差
        
        if self.debug:
            print("🔗 VD+eTag時空對齊器初始化")
    
    def get_available_dates(self) -> Dict[str, List[str]]:
        """檢測可用資料日期"""
        vd_dates = self._scan_vd_dates()
        etag_dates = self._scan_etag_dates()
        common_dates = sorted(set(vd_dates) & set(etag_dates))
        
        if self.debug:
            print(f"📅 可用資料: VD={len(vd_dates)}, eTag={len(etag_dates)}, 共同={len(common_dates)}")
        
        return {
            'vd_dates': vd_dates,
            'etag_dates': etag_dates,
            'common_dates': common_dates
        }
    
    def _scan_vd_dates(self) -> List[str]:
        """掃描VD數據日期"""
        dates = []
        vd_folders = [
            self.base_folder / "processed" / "vd",
            self.base_folder / "cleaned"
        ]
        
        for folder in vd_folders:
            if folder.exists():
                for date_folder in folder.iterdir():
                    if date_folder.is_dir() and self._is_valid_date(date_folder.name):
                        target_file = date_folder / "target_route_data.csv"
                        cleaned_file = date_folder / "target_route_data_cleaned.csv"
                        if target_file.exists() or cleaned_file.exists():
                            dates.append(date_folder.name)
        
        return sorted(list(set(dates)))
    
    def _scan_etag_dates(self) -> List[str]:
        """掃描eTag數據日期"""
        dates = []
        etag_folder = self.base_folder / "processed" / "etag"
        
        if etag_folder.exists():
            for date_folder in etag_folder.iterdir():
                if date_folder.is_dir() and self._is_valid_date(date_folder.name):
                    etag_file = date_folder / "etag_travel_time.csv"
                    if etag_file.exists():
                        dates.append(date_folder.name)
        
        return sorted(dates)
    
    def _is_valid_date(self, date_str: str) -> bool:
        """檢查是否為有效日期格式 YYYY-MM-DD"""
        return len(date_str.split('-')) == 3 and len(date_str) == 10
    
    def align_date_data(self, date_str: str) -> Dict:
        """對齊單日資料"""
        if self.debug:
            print(f"🔗 開始對齊 {date_str}")
        
        try:
            # 載入資料
            vd_data = self._load_vd_data(date_str)
            etag_data = self._load_etag_data(date_str)
            
            if vd_data.empty:
                return {'error': f'VD資料不存在: {date_str}'}
            if etag_data.empty:
                return {'error': f'eTag資料不存在: {date_str}'}
            
            # 執行對齊
            aligned_records = self._perform_alignment(vd_data, etag_data)
            
            if not aligned_records:
                return {'error': '無法產生對齊記錄'}
            
            # 建立結果
            aligned_df = pd.DataFrame(aligned_records)
            summary = self._generate_summary(aligned_df)
            
            # 保存結果
            self._save_results(aligned_df, summary, date_str)
            
            if self.debug:
                print(f"✅ 對齊完成: {len(aligned_df)} 筆記錄")
            
            return {
                'aligned': aligned_df,
                'summary': summary
            }
            
        except Exception as e:
            error_msg = f'對齊失敗: {str(e)}'
            if self.debug:
                print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def _load_vd_data(self, date_str: str) -> pd.DataFrame:
        """載入VD資料"""
        # 優先載入清理後的檔案
        vd_files = [
            self.base_folder / "cleaned" / date_str / "target_route_data_cleaned.csv",
            self.base_folder / "processed" / "vd" / date_str / "target_route_data.csv"
        ]
        
        for vd_file in vd_files:
            if vd_file.exists():
                try:
                    df = pd.read_csv(vd_file)
                    df['update_time'] = pd.to_datetime(df['update_time'])
                    return df
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ VD載入失敗: {vd_file.name} - {e}")
        
        return pd.DataFrame()
    
    def _load_etag_data(self, date_str: str) -> pd.DataFrame:
        """載入eTag資料"""
        etag_file = self.base_folder / "processed" / "etag" / date_str / "etag_travel_time.csv"
        
        if etag_file.exists():
            try:
                df = pd.read_csv(etag_file)
                df['update_time'] = pd.to_datetime(df['update_time'])
                return df
            except Exception as e:
                if self.debug:
                    print(f"⚠️ eTag載入失敗: {e}")
        
        return pd.DataFrame()
    
    def _perform_alignment(self, vd_data: pd.DataFrame, etag_data: pd.DataFrame) -> List[Dict]:
        """執行時空對齊"""
        aligned_records = []
        
        # 遍歷每個區域
        for region, etag_pairs in self.spatial_mapping.items():
            # 篩選VD站點
            vd_subset = vd_data[vd_data['vd_id'].str.contains(region, na=False)]
            if vd_subset.empty:
                continue
            
            # 對每個eTag配對進行時間對齊
            for etag_pair in etag_pairs:
                etag_subset = etag_data[etag_data['etag_pair_id'] == etag_pair]
                if etag_subset.empty:
                    continue
                
                # 時間對齊
                records = self._temporal_alignment(vd_subset, etag_subset, region, etag_pair)
                aligned_records.extend(records)
        
        return aligned_records
    
    def _temporal_alignment(self, vd_data: pd.DataFrame, etag_data: pd.DataFrame, 
                           region: str, etag_pair: str) -> List[Dict]:
        """時間對齊：VD(1分鐘) → eTag(5分鐘)"""
        records = []
        
        try:
            # VD數據聚合到5分鐘
            vd_grouped = vd_data.copy()
            vd_grouped['time_bin'] = vd_grouped['update_time'].dt.floor('5min')
            
            vd_agg = vd_grouped.groupby('time_bin').agg({
                'speed': 'mean',
                'volume_total': 'sum',
                'occupancy': 'mean'
            }).reset_index()
            
            # 與eTag時間匹配
            for _, etag_row in etag_data.iterrows():
                etag_time = etag_row['update_time']
                
                # 找最接近的VD時間窗口
                time_diffs = abs((vd_agg['time_bin'] - etag_time).dt.total_seconds() / 60)
                
                if len(time_diffs) > 0:
                    min_idx = time_diffs.idxmin()
                    min_diff = time_diffs.iloc[min_idx]
                    
                    if min_diff <= self.sync_tolerance:
                        vd_row = vd_agg.iloc[min_idx]
                        
                        record = {
                            'datetime': etag_time,
                            'region': region,
                            'etag_pair': etag_pair,
                            'vd_speed': float(vd_row['speed']) if pd.notna(vd_row['speed']) else 0,
                            'vd_volume': int(vd_row['volume_total']) if pd.notna(vd_row['volume_total']) else 0,
                            'vd_occupancy': float(vd_row['occupancy']) if pd.notna(vd_row['occupancy']) else 0,
                            'etag_speed': float(etag_row.get('space_mean_speed_kmh', 0)) or 0,
                            'etag_volume': int(etag_row.get('vehicle_count', 0)) or 0,
                            'time_diff_min': float(min_diff)
                        }
                        
                        records.append(record)
        
        except Exception as e:
            if self.debug:
                print(f"⚠️ 時間對齊錯誤: {e}")
        
        return records
    
    def _generate_summary(self, aligned_df: pd.DataFrame) -> Dict:
        """生成對齊摘要"""
        return {
            'total_records': len(aligned_df),
            'regions': int(aligned_df['region'].nunique()),
            'etag_pairs': int(aligned_df['etag_pair'].nunique()),
            'speed_correlation': float(aligned_df['vd_speed'].corr(aligned_df['etag_speed'])) 
                                if len(aligned_df) > 1 else 0,
            'sync_quality_percent': float((aligned_df['time_diff_min'] <= self.sync_tolerance).mean() * 100)
        }
    
    def _save_results(self, aligned_df: pd.DataFrame, summary: Dict, date_str: str):
        """保存對齊結果"""
        output_folder = self.base_folder / "processed" / "fusion" / date_str
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # 保存對齊數據
        aligned_file = output_folder / "vd_etag_aligned.csv"
        aligned_df.to_csv(aligned_file, index=False, encoding='utf-8-sig')
        
        # 保存摘要
        summary_file = output_folder / "alignment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        if self.debug:
            print(f"💾 結果已保存: {aligned_file}")
    
    def batch_align_all_available(self) -> Dict:
        """批次對齊所有可用資料"""
        available = self.get_available_dates()
        common_dates = available['common_dates']
        
        if not common_dates:
            return {'error': '沒有可用的共同日期'}
        
        if self.debug:
            print(f"🚀 批次處理 {len(common_dates)} 天")
        
        results = {}
        for date_str in common_dates:
            result = self.align_date_data(date_str)
            results[date_str] = result
        
        successful = sum(1 for r in results.values() if 'aligned' in r)
        if self.debug:
            print(f"🏁 批次完成: {successful}/{len(common_dates)} 成功")
        
        return results
    
    def validate_alignment(self, date_str: str) -> Dict:
        """驗證對齊品質"""
        aligned_file = self.base_folder / "processed" / "fusion" / date_str / "vd_etag_aligned.csv"
        
        if not aligned_file.exists():
            return {'error': '對齊檔案不存在'}
        
        try:
            df = pd.read_csv(aligned_file)
            
            return {
                'record_count': len(df),
                'time_sync_quality': (df['time_diff_min'] <= self.sync_tolerance).mean() * 100,
                'speed_correlation': df['vd_speed'].corr(df['etag_speed']),
                'data_completeness': df.notna().all(axis=1).mean() * 100
            }
            
        except Exception as e:
            return {'error': f'驗證失敗: {str(e)}'}


# 便利函數
def align_all_available_data(debug: bool = False) -> Dict:
    """對齊所有可用資料的便利函數"""
    aligner = SpatialTemporalAligner(debug=debug)
    return aligner.batch_align_all_available()


def get_available_data_status(debug: bool = False) -> Dict:
    """獲取可用資料狀態的便利函數"""
    aligner = SpatialTemporalAligner(debug=debug)
    return aligner.get_available_dates()


if __name__ == "__main__":
    print("🔗 VD+eTag時空對齊器 - 簡化版")
    print("=" * 40)
    
    # 初始化對齊器
    aligner = SpatialTemporalAligner(debug=True)
    
    # 檢查可用資料
    available = aligner.get_available_dates()
    print(f"\n📊 資料狀態: {len(available['common_dates'])} 天可對齊")
    
    if available['common_dates']:
        # 批次對齊
        print("\n🚀 開始批次對齊...")
        results = aligner.batch_align_all_available()
        
        # 顯示結果
        for date_str, result in results.items():
            if 'aligned' in result:
                summary = result['summary']
                print(f"✅ {date_str}: {summary['total_records']} 筆, "
                      f"相關性 {summary['speed_correlation']:.3f}")
            else:
                print(f"❌ {date_str}: {result.get('error', '未知錯誤')}")
    else:
        print("\n⚠️ 沒有可用的共同日期資料")
        print("請確認 data/processed/vd/ 和 data/processed/etag/ 目錄下有相同日期的資料")