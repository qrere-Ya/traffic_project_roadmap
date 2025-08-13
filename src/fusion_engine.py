# src/fusion_engine.py - VD+eTag融合引擎

"""
VD+eTag多源數據融合引擎
=======================

核心功能：
1. 載入時空對齊數據
2. 多源特徵融合
3. 智能特徵選擇
4. 融合數據品質評估

簡化原則：
- 清晰的數據融合邏輯
- 自動化特徵工程
- 有效的品質控制

作者: 交通預測專案團隊
日期: 2025-01-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class FusionEngine:
    """VD+eTag數據融合引擎"""
    
    def __init__(self, base_folder: str = "data", debug: bool = False):
        self.base_folder = Path(base_folder)
        self.debug = debug
        
        # 融合參數
        self.feature_threshold = 0.01  # 特徵重要性閾值
        self.correlation_threshold = 0.95  # 高相關性閾值
        
        # 存儲器 - 每次批次處理重置
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.global_feature_template = None  # 全局特徵模板
        
        if self.debug:
            print("🔧 VD+eTag融合引擎初始化")
    
    def get_available_fusion_dates(self) -> List[str]:
        """獲取可用的融合數據日期"""
        fusion_folder = self.base_folder / "processed" / "fusion"
        dates = []
        
        if fusion_folder.exists():
            for date_folder in fusion_folder.iterdir():
                if date_folder.is_dir() and self._is_valid_date(date_folder.name):
                    aligned_file = date_folder / "vd_etag_aligned.csv"
                    if aligned_file.exists():
                        dates.append(date_folder.name)
        
        return sorted(dates)
    
    def _is_valid_date(self, date_str: str) -> bool:
        """檢查日期格式"""
        return len(date_str.split('-')) == 3 and len(date_str) == 10
    
    def load_aligned_data(self, date_str: str) -> pd.DataFrame:
        """載入對齊數據"""
        aligned_file = self.base_folder / "processed" / "fusion" / date_str / "vd_etag_aligned.csv"
        
        if not aligned_file.exists():
            raise FileNotFoundError(f"對齊數據不存在: {date_str}")
        
        try:
            df = pd.read_csv(aligned_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            if self.debug:
                print(f"📄 載入 {date_str}: {len(df)} 筆對齊記錄")
            
            return df
            
        except Exception as e:
            raise Exception(f"數據載入失敗: {str(e)}")
    
    def create_fusion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建融合特徵 - 確保一致性"""
        df = df.copy()
        
        # 基礎特徵檢查
        required_cols = ['vd_speed', 'vd_volume', 'vd_occupancy', 'etag_speed', 'etag_volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要欄位: {missing_cols}")
        
        if self.debug:
            print("🔧 創建融合特徵...")
        
        # 1. 時間特徵
        df = self._create_time_features(df)
        
        # 2. 速度融合特徵
        df = self._create_speed_fusion_features(df)
        
        # 3. 流量融合特徵
        df = self._create_volume_fusion_features(df)
        
        # 4. 交通狀態特徵
        df = self._create_traffic_state_features(df)
        
        # 5. 區域特徵 - 確保一致性
        df = self._create_region_features_consistent(df)
        
        # 6. 滯後特徵 - 限制生成
        df = self._create_lag_features_limited(df)
        
        # 7. 填補缺失值確保一致性
        df = self._ensure_feature_consistency(df)
        
        if self.debug:
            print(f"✅ 融合特徵創建完成: {len(df.columns)} 個特徵")
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間特徵"""
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 尖峰時段
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        # 周期性特徵
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def _create_speed_fusion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """速度融合特徵"""
        # 速度差異
        df['speed_diff'] = df['vd_speed'] - df['etag_speed']
        df['speed_diff_abs'] = abs(df['speed_diff'])
        df['speed_diff_ratio'] = df['speed_diff'] / (df['etag_speed'] + 1)
        
        # 速度統計
        df['speed_mean'] = (df['vd_speed'] + df['etag_speed']) / 2
        df['speed_max'] = df[['vd_speed', 'etag_speed']].max(axis=1)
        df['speed_min'] = df[['vd_speed', 'etag_speed']].min(axis=1)
        
        # 速度等級
        df['vd_speed_level'] = pd.cut(df['vd_speed'], bins=[0, 40, 60, 80, 120], labels=[1, 2, 3, 4]).astype(float)
        df['etag_speed_level'] = pd.cut(df['etag_speed'], bins=[0, 40, 60, 80, 120], labels=[1, 2, 3, 4]).astype(float)
        
        return df
    
    def _create_volume_fusion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """流量融合特徵"""
        # 流量差異
        df['volume_diff'] = df['vd_volume'] - df['etag_volume']
        df['volume_diff_abs'] = abs(df['volume_diff'])
        df['volume_diff_ratio'] = df['volume_diff'] / (df['etag_volume'] + 1)
        
        # 流量統計
        df['volume_mean'] = (df['vd_volume'] + df['etag_volume']) / 2
        df['volume_max'] = df[['vd_volume', 'etag_volume']].max(axis=1)
        
        # 流量密度關係
        df['vd_density'] = df['vd_volume'] / (df['vd_speed'] + 1)
        df['etag_density'] = df['etag_volume'] / (df['etag_speed'] + 1)
        df['density_ratio'] = df['vd_density'] / (df['etag_density'] + 1)
        
        return df
    
    def _create_traffic_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交通狀態特徵"""
        # 擁堵程度 (基於速度)
        df['vd_congestion'] = (100 - df['vd_speed']).clip(0, 100) / 100
        df['etag_congestion'] = (100 - df['etag_speed']).clip(0, 100) / 100
        df['congestion_mean'] = (df['vd_congestion'] + df['etag_congestion']) / 2
        
        # 交通效率 (速度/流量比)
        df['vd_efficiency'] = df['vd_speed'] / (df['vd_volume'] + 1)
        df['etag_efficiency'] = df['etag_speed'] / (df['etag_volume'] + 1)
        df['efficiency_ratio'] = df['vd_efficiency'] / (df['etag_efficiency'] + 1)
        
        # 佔有率相關 (如果有)
        if 'vd_occupancy' in df.columns:
            df['occupancy_speed_ratio'] = df['vd_occupancy'] / (df['vd_speed'] + 1)
            df['occupancy_volume_ratio'] = df['vd_occupancy'] / (df['vd_volume'] + 1)
        
        return df
    
    def _create_region_features_consistent(self, df: pd.DataFrame) -> pd.DataFrame:
        """區域特徵 - 確保一致性"""
        # 確保所有必要的區域特徵都存在
        df['region_code'] = 0  # 預設值
        df['is_yuanshan'] = 0
        df['is_taipei'] = 0  
        df['is_sanchong'] = 0
        
        if 'region' in df.columns:
            # 區域編碼
            region_mapping = {'圓山': 1, '台北': 2, '三重': 3}
            df['region_code'] = df['region'].map(region_mapping).fillna(0).astype(int)
            
            # 區域標記
            df['is_yuanshan'] = (df['region'] == '圓山').astype(int)
            df['is_taipei'] = (df['region'] == '台北').astype(int)
            df['is_sanchong'] = (df['region'] == '三重').astype(int)
        
        return df
    
    def _create_lag_features_limited(self, df: pd.DataFrame, periods: List[int] = [1, 2]) -> pd.DataFrame:
        """滯後特徵 - 限制生成確保一致性"""
        # 確保基礎目標特徵存在
        if 'speed_mean' not in df.columns:
            df['speed_mean'] = (df['vd_speed'] + df['etag_speed']) / 2
        if 'volume_mean' not in df.columns:
            df['volume_mean'] = (df['vd_volume'] + df['etag_volume']) / 2
        
        # 初始化滯後特徵為0
        for period in periods:
            df[f'speed_mean_lag_{period}'] = 0.0
            df[f'vd_speed_lag_{period}'] = 0.0
            df[f'volume_mean_lag_{period}'] = 0.0
        
        # 如果有足夠數據，計算實際滯後特徵
        if len(df) > max(periods) and 'region' in df.columns and 'etag_pair' in df.columns:
            try:
                df = df.sort_values(['region', 'etag_pair', 'datetime'])
                
                for period in periods:
                    # 速度滯後
                    df[f'speed_mean_lag_{period}'] = df.groupby(['region', 'etag_pair'])['speed_mean'].shift(period).fillna(0)
                    df[f'vd_speed_lag_{period}'] = df.groupby(['region', 'etag_pair'])['vd_speed'].shift(period).fillna(0)
                    
                    # 流量滯後
                    df[f'volume_mean_lag_{period}'] = df.groupby(['region', 'etag_pair'])['volume_mean'].shift(period).fillna(0)
            except Exception as e:
                if self.debug:
                    print(f"⚠️ 滯後特徵計算警告: {e}")
                # 保持預設值0
        
        return df
    
    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """確保特徵一致性"""
        # 定義核心特徵集合
        core_features = [
            # 時間特徵
            'hour', 'day_of_week', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 
            'is_peak_hour', 'hour_sin', 'hour_cos',
            
            # 速度特徵
            'speed_diff', 'speed_diff_abs', 'speed_diff_ratio', 'speed_mean', 
            'speed_max', 'speed_min', 'vd_speed_level', 'etag_speed_level',
            
            # 流量特徵
            'volume_diff', 'volume_diff_abs', 'volume_diff_ratio', 'volume_mean', 
            'volume_max', 'vd_density', 'etag_density', 'density_ratio',
            
            # 交通狀態特徵
            'vd_congestion', 'etag_congestion', 'congestion_mean',
            'vd_efficiency', 'etag_efficiency', 'efficiency_ratio',
            
            # 區域特徵
            'region_code', 'is_yuanshan', 'is_taipei', 'is_sanchong',
            
            # 滯後特徵
            'speed_mean_lag_1', 'speed_mean_lag_2', 'vd_speed_lag_1', 'vd_speed_lag_2',
            'volume_mean_lag_1', 'volume_mean_lag_2'
        ]
        
        # 確保所有核心特徵都存在
        for feature in core_features:
            if feature not in df.columns:
                df[feature] = 0.0  # 預設值
        
        # 填補佔有率相關特徵
        if 'vd_occupancy' in df.columns:
            if 'occupancy_speed_ratio' not in df.columns:
                df['occupancy_speed_ratio'] = df['vd_occupancy'] / (df['vd_speed'] + 1)
            if 'occupancy_volume_ratio' not in df.columns:
                df['occupancy_volume_ratio'] = df['vd_occupancy'] / (df['vd_volume'] + 1)
        else:
            df['occupancy_speed_ratio'] = 0.0
            df['occupancy_volume_ratio'] = 0.0
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'speed_mean', k: int = 20) -> pd.DataFrame:
        """智能特徵選擇"""
        if self.debug:
            print("🎯 執行特徵選擇...")
        
        # 準備特徵和目標
        feature_cols = [col for col in df.columns if col not in 
                       ['datetime', 'region', 'etag_pair', target_col]]
        
        # 移除非數值特徵
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].fillna(0)
        y = df[target_col].fillna(df[target_col].mean())
        
        # 移除高相關性特徵
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        X = X.drop(columns=to_drop)
        
        if self.debug:
            print(f"   移除高相關特徵: {len(to_drop)} 個")
        
        # SelectKBest 特徵選擇
        if len(X.columns) > k:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            selected_features = X.columns.tolist()
        
        self.feature_names = selected_features
        
        if self.debug:
            print(f"   最終選擇特徵: {len(selected_features)} 個")
        
        # 回傳包含選擇特徵的數據
        result_cols = ['datetime', 'region', 'etag_pair', target_col] + selected_features
        return df[result_cols]
    
    def normalize_features(self, df: pd.DataFrame, target_col: str = 'speed_mean') -> pd.DataFrame:
        """特徵標準化"""
        if not self.feature_names:
            raise ValueError("請先執行特徵選擇")
        
        df = df.copy()
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            df[self.feature_names] = self.scaler.fit_transform(df[self.feature_names].fillna(0))
        else:
            df[self.feature_names] = self.scaler.transform(df[self.feature_names].fillna(0))
        
        if self.debug:
            print(f"✅ 特徵標準化完成: {len(self.feature_names)} 個特徵")
        
        return df
    
    def process_single_date(self, date_str: str, target_col: str = 'speed_mean') -> Dict:
        """處理單日融合數據"""
        try:
            if self.debug:
                print(f"🔧 處理融合數據: {date_str}")
            
            # 載入對齊數據
            df = self.load_aligned_data(date_str)
            
            # 創建融合特徵
            df = self.create_fusion_features(df)
            
            # 特徵選擇
            df = self.select_features(df, target_col)
            
            # 特徵標準化
            df = self.normalize_features(df, target_col)
            
            # 品質評估
            quality = self._assess_quality(df, target_col)
            
            # 保存結果
            self._save_fusion_data(df, quality, date_str)
            
            if self.debug:
                print(f"✅ 融合處理完成: {len(df)} 筆記錄")
            
            return {
                'fusion_data': df,
                'quality': quality,
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            error_msg = f'融合處理失敗: {str(e)}'
            if self.debug:
                print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def _assess_quality(self, df: pd.DataFrame, target_col: str) -> Dict:
        """評估融合數據品質"""
        quality = {
            'record_count': len(df),
            'feature_count': len(self.feature_names),
            'data_completeness': df.notna().all(axis=1).mean() * 100,
            'target_std': float(df[target_col].std()),
            'target_range': float(df[target_col].max() - df[target_col].min())
        }
        
        # 數據變異性檢查
        feature_std = df[self.feature_names].std()
        quality['feature_variance'] = float(feature_std.mean())
        quality['low_variance_features'] = int((feature_std < 0.01).sum())
        
        return quality
    
    def _save_fusion_data(self, df: pd.DataFrame, quality: Dict, date_str: str):
        """保存融合數據"""
        output_folder = self.base_folder / "processed" / "fusion" / date_str
        
        # 保存融合特徵數據
        fusion_file = output_folder / "fusion_features.csv"
        df.to_csv(fusion_file, index=False, encoding='utf-8-sig')
        
        # 保存品質報告
        quality_file = output_folder / "fusion_quality.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality, f, ensure_ascii=False, indent=2, default=str)
        
        if self.debug:
            print(f"💾 融合數據已保存: {fusion_file}")
    
    def batch_process_all_dates(self, target_col: str = 'speed_mean') -> Dict:
        """批次處理所有可用日期 - 修正版"""
        available_dates = self.get_available_fusion_dates()
        
        if not available_dates:
            return {'error': '沒有可用的融合數據'}
        
        if self.debug:
            print(f"🚀 批次處理 {len(available_dates)} 天融合數據")
        
        # 重置全局狀態
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.global_feature_template = None
        
        results = {}
        total_records = 0
        
        # 第一階段：建立特徵模板
        if self.debug:
            print("📋 第一階段：建立特徵模板...")
        
        template_established = False
        for date_str in available_dates:
            try:
                # 載入並處理數據建立模板
                df = self.load_aligned_data(date_str)
                df = self.create_fusion_features(df)
                
                # 建立特徵模板
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols if col not in 
                               ['datetime', 'region', 'etag_pair', target_col]]
                
                if not template_established:
                    self.global_feature_template = feature_cols
                    template_established = True
                    if self.debug:
                        print(f"✅ 特徵模板建立: {len(self.global_feature_template)} 個特徵")
                    break
                    
            except Exception as e:
                if self.debug:
                    print(f"⚠️ {date_str} 模板建立失敗: {e}")
                continue
        
        if not template_established:
            return {'error': '無法建立特徵模板'}
        
        # 第二階段：使用第一個成功的日期訓練特徵選擇器
        if self.debug:
            print("🎯 第二階段：訓練特徵選擇器...")
            
        for date_str in available_dates:
            try:
                result = self._process_single_date_with_template(date_str, target_col, is_template=True)
                if 'fusion_data' in result:
                    if self.debug:
                        print(f"✅ 特徵選擇器訓練完成")
                    break
            except Exception as e:
                if self.debug:
                    print(f"⚠️ {date_str} 選擇器訓練失敗: {e}")
                continue
        
        # 第三階段：批次處理所有日期
        if self.debug:
            print("🔄 第三階段：批次處理...")
            
        for date_str in available_dates:
            try:
                result = self._process_single_date_with_template(date_str, target_col, is_template=False)
                results[date_str] = result
                
                if 'fusion_data' in result:
                    total_records += len(result['fusion_data'])
                    
            except Exception as e:
                error_msg = f'融合處理失敗: {str(e)}'
                results[date_str] = {'error': error_msg}
                if self.debug:
                    print(f"❌ {date_str}: {error_msg}")
        
        successful = sum(1 for r in results.values() if 'fusion_data' in r)
        
        if self.debug:
            print(f"🏁 批次處理完成: {successful}/{len(available_dates)} 成功")
            print(f"📊 總融合記錄: {total_records:,} 筆")
        
        return results
    
    def _process_single_date_with_template(self, date_str: str, target_col: str, is_template: bool = False) -> Dict:
        """使用模板處理單日數據"""
        try:
            # 載入數據
            df = self.load_aligned_data(date_str)
            
            # 創建融合特徵
            df = self.create_fusion_features(df)
            
            # 確保符合特徵模板
            df = self._align_to_feature_template(df, target_col)
            
            # 特徵選擇和標準化
            if is_template or self.feature_selector is None:
                # 第一次處理，建立選擇器
                df = self.select_features(df, target_col, k=15)
                df = self.normalize_features(df, target_col)
            else:
                # 後續處理，使用已建立的選擇器
                df = self._apply_existing_feature_processing(df, target_col)
            
            # 品質評估
            quality = self._assess_quality(df, target_col)
            
            # 保存結果
            self._save_fusion_data(df, quality, date_str)
            
            return {
                'fusion_data': df,
                'quality': quality,
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _align_to_feature_template(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """對齊到特徵模板"""
        if not self.global_feature_template:
            return df
        
        # 確保所有模板特徵都存在
        for feature in self.global_feature_template:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # 保留模板特徵和必要欄位
        keep_cols = ['datetime', 'region', 'etag_pair', target_col] + self.global_feature_template
        available_cols = [col for col in keep_cols if col in df.columns]
        
        return df[available_cols]
    
    def _apply_existing_feature_processing(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """使用已建立的特徵處理器"""
        # 選擇特徵
        if self.feature_names:
            feature_cols = [col for col in self.feature_names if col in df.columns]
            result_cols = ['datetime', 'region', 'etag_pair', target_col] + feature_cols
            df = df[result_cols]
        
        # 標準化
        if self.scaler and self.feature_names:
            available_features = [col for col in self.feature_names if col in df.columns]
            if available_features:
                df[available_features] = self.scaler.transform(df[available_features].fillna(0))
        
        return df


# 便利函數
def process_all_fusion_data(debug: bool = False) -> Dict:
    """處理所有融合數據的便利函數"""
    engine = FusionEngine(debug=debug)
    return engine.batch_process_all_dates()


def get_fusion_data_status(debug: bool = False) -> Dict:
    """獲取融合數據狀態的便利函數"""
    engine = FusionEngine(debug=debug)
    dates = engine.get_available_fusion_dates()
    
    return {
        'available_dates': dates,
        'total_days': len(dates)
    }


if __name__ == "__main__":
    print("🔧 VD+eTag融合引擎")
    print("=" * 30)
    
    # 初始化融合引擎
    engine = FusionEngine(debug=True)
    
    # 檢查可用數據
    dates = engine.get_available_fusion_dates()
    print(f"\n📊 可用融合數據: {len(dates)} 天")
    
    if dates:
        # 批次處理
        print("\n🚀 開始批次融合處理...")
        results = engine.batch_process_all_dates()
        
        # 顯示結果
        for date_str, result in results.items():
            if 'fusion_data' in result:
                quality = result['quality']
                print(f"✅ {date_str}: {quality['record_count']} 筆, "
                      f"{quality['feature_count']} 特徵, "
                      f"完整性 {quality['data_completeness']:.1f}%")
            else:
                print(f"❌ {date_str}: {result.get('error', '處理失敗')}")
    else:
        print("\n⚠️ 沒有可用的融合數據")
        print("請先執行時空對齊: python src/spatial_temporal_aligner.py")