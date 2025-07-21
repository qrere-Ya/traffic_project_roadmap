"""
簡化版交通流量分析器 - 核心功能
=====================================

專注核心功能：
1. 🎯 數據載入和基本分析
2. 🤖 AI模型適用性評估 
3. 📊 預測就緒度檢查
4. 📋 分析報告生成

作者: 交通預測專案團隊
日期: 2025-07-21 (簡化核心版)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class SimplifiedTrafficAnalyzer:
    """簡化版交通流量分析器 - 專注核心功能"""
    
    def __init__(self, base_folder: str = "data"):
        self.base_folder = Path(base_folder)
        self.cleaned_folder = self.base_folder / "cleaned"
        self.datasets = {}
        self.analysis_results = {}
        self.insights = []
        
        # 核心預測配置
        self.prediction_config = {
            'target_columns': ['speed', 'volume_total', 'occupancy'],
            'min_records_for_lstm': 50000,
            'min_time_span_days': 5
        }
        
        print("🔬 簡化版交通分析器初始化")
        print(f"   📁 數據目錄: {self.cleaned_folder}")
        print(f"   🎯 預測目標: {', '.join(self.prediction_config['target_columns'])}")
        
        self._detect_data_files()
    
    def _detect_data_files(self):
        """檢測可用的數據檔案"""
        if not self.cleaned_folder.exists():
            print(f"❌ 清理數據目錄不存在: {self.cleaned_folder}")
            self.available_dates = []
            return
        
        # 檢測日期資料夾
        date_folders = [d for d in self.cleaned_folder.iterdir() 
                       if d.is_dir() and d.name.count('-') == 2]
        
        if date_folders:
            self.available_dates = sorted([d.name for d in date_folders])
            print(f"✅ 發現 {len(self.available_dates)} 個已清理日期")
        else:
            print("⚠️ 未找到已清理的日期資料夾")
            self.available_dates = []
    
    def load_data(self, merge_dates: bool = True, sample_rate: float = 1.0) -> bool:
        """載入數據"""
        print("📊 載入數據...")
        
        try:
            if not self.available_dates:
                print("❌ 無可用數據")
                return False
            
            all_data = {}
            
            for date_str in self.available_dates:
                date_folder = self.cleaned_folder / date_str
                
                # 載入目標路段數據
                data_file = date_folder / "target_route_data_cleaned.csv"
                peak_file = date_folder / "target_route_peak_cleaned.csv"
                offpeak_file = date_folder / "target_route_offpeak_cleaned.csv"
                
                for file_path, key in [
                    (data_file, 'target_data'),
                    (peak_file, 'target_peak'),
                    (offpeak_file, 'target_offpeak')
                ]:
                    if file_path.exists():
                        if key not in all_data:
                            all_data[key] = []
                        
                        # 讀取數據
                        df = pd.read_csv(file_path, low_memory=True)
                        
                        # 採樣（如果需要）
                        if sample_rate < 1.0:
                            df = df.sample(frac=sample_rate, random_state=42)
                        
                        df['source_date'] = date_str
                        all_data[key].append(df)
            
            # 合併數據
            for key, df_list in all_data.items():
                if df_list:
                    self.datasets[key] = pd.concat(df_list, ignore_index=True)
                    print(f"   ✅ {key}: {len(self.datasets[key]):,} 筆記錄")
            
            return len(self.datasets) > 0
                
        except Exception as e:
            print(f"❌ 數據載入失敗: {e}")
            return False
    
    def analyze_data_characteristics(self) -> Dict[str, Any]:
        """分析數據特性"""
        print("🔍 分析數據特性...")
        
        characteristics = {
            'data_summary': {},
            'quality_metrics': {},
            'time_coverage': {},
            'prediction_readiness': {}
        }
        
        total_records = 0
        all_dates = set()
        
        for name, df in self.datasets.items():
            if not df.empty:
                # 基本統計
                characteristics['data_summary'][name] = {
                    'records': len(df),
                    'avg_speed': round(df['speed'].mean(), 2) if 'speed' in df.columns else 0,
                    'avg_volume': round(df['volume_total'].mean(), 2) if 'volume_total' in df.columns else 0,
                    'unique_vd_stations': df['vd_id'].nunique() if 'vd_id' in df.columns else 0,
                    'completeness': round(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
                }
                
                total_records += len(df)
                
                # 收集日期
                if 'source_date' in df.columns:
                    all_dates.update(df['source_date'].unique())
        
        # 時間覆蓋分析
        date_span_days = len(all_dates)
        characteristics['time_coverage'] = {
            'total_records': total_records,
            'unique_dates': len(all_dates),
            'date_span_days': date_span_days,
            'records_per_day': round(total_records / max(len(all_dates), 1), 0),
            'estimated_ai_training_hours': round(total_records / 720, 1)
        }
        
        # 預測就緒度評估
        main_df = self.datasets.get('target_data')
        if main_df is not None and not main_df.empty:
            prediction_readiness = self._assess_prediction_readiness(main_df, date_span_days)
            characteristics['prediction_readiness'] = prediction_readiness
        
        # 品質評估
        if total_records > 0:
            volume_score = min(100, total_records / 1000)
            time_score = min(100, date_span_days * 20)
            balance_score = self._calculate_balance_score()
            prediction_score = self._calculate_prediction_readiness_score(characteristics)
            
            overall_quality = (volume_score * 0.3 + time_score * 0.3 + 
                             balance_score * 0.2 + prediction_score * 0.2)
            
            characteristics['quality_metrics'] = {
                'volume_score': round(volume_score, 1),
                'time_score': round(time_score, 1),
                'balance_score': round(balance_score, 1),
                'prediction_score': round(prediction_score, 1),
                'overall_quality': round(overall_quality, 1)
            }
        
        self.analysis_results['data_characteristics'] = characteristics
        
        # 生成洞察
        self._generate_insights(characteristics)
        
        return characteristics
    
    def _assess_prediction_readiness(self, df: pd.DataFrame, date_span_days: int) -> Dict[str, Any]:
        """評估預測就緒度"""
        config = self.prediction_config
        
        total_records = len(df)
        unique_vds = df['vd_id'].nunique()
        
        # 計算時間跨度（安全方式）
        try:
            if 'update_time' in df.columns:
                df['update_time'] = pd.to_datetime(df['update_time'], errors='coerce')
                valid_times = df['update_time'].dropna()
                if len(valid_times) > 0:
                    time_span_hours = (valid_times.max() - valid_times.min()).total_seconds() / 3600
                else:
                    time_span_hours = date_span_days * 24
            else:
                time_span_hours = date_span_days * 24
        except:
            time_span_hours = date_span_days * 24
        
        # 缺失值評估
        target_completeness = {}
        for col in config['target_columns']:
            if col in df.columns:
                completeness = (1 - df[col].isna().sum() / len(df)) * 100
                target_completeness[col] = completeness
        
        avg_completeness = np.mean(list(target_completeness.values())) if target_completeness else 0
        
        # 模型就緒度評估
        lstm_ready = (
            total_records >= config['min_records_for_lstm'] and
            date_span_days >= config['min_time_span_days'] and
            avg_completeness >= 80 and
            unique_vds >= 3
        )
        
        xgboost_ready = (
            total_records >= 10000 and
            avg_completeness >= 70 and
            unique_vds >= 2
        )
        
        rf_ready = (
            total_records >= 5000 and
            avg_completeness >= 60
        )
        
        return {
            'total_records': total_records,
            'unique_vd_stations': unique_vds,
            'time_span_days': date_span_days,
            'time_span_hours': time_span_hours,
            'target_completeness': target_completeness,
            'avg_completeness': avg_completeness,
            'lstm_ready': lstm_ready,
            'xgboost_ready': xgboost_ready,
            'rf_ready': rf_ready,
            'prediction_ready': lstm_ready or xgboost_ready
        }
    
    def _calculate_prediction_readiness_score(self, characteristics: Dict[str, Any]) -> float:
        """計算預測準備度評分"""
        readiness = characteristics.get('prediction_readiness', {})
        
        if not readiness:
            return 50
        
        score = 0
        
        # 模型準備度評分
        if readiness.get('lstm_ready', False):
            score += 40
        elif readiness.get('xgboost_ready', False):
            score += 25
        elif readiness.get('rf_ready', False):
            score += 15
        
        # 數據完整性評分
        completeness = readiness.get('avg_completeness', 0)
        score += min(30, completeness * 0.3)
        
        # 時間跨度評分
        time_span = readiness.get('time_span_days', 0)
        score += min(20, time_span * 2)
        
        # VD站點數量評分
        vd_count = readiness.get('unique_vd_stations', 0)
        score += min(10, vd_count * 2)
        
        return score
    
    def _generate_insights(self, characteristics: Dict[str, Any]):
        """生成分析洞察"""
        total_records = characteristics['time_coverage']['total_records']
        prediction_readiness = characteristics.get('prediction_readiness', {})
        
        # 數據規模洞察
        if total_records > 50000:
            self.insights.append(f"🚀 優秀數據規模：{total_records:,}筆記錄支援深度學習模型")
        
        # 預測模型洞察
        if prediction_readiness.get('lstm_ready', False):
            self.insights.append("🎯 LSTM模型就緒：數據符合時間序列深度學習要求")
        elif prediction_readiness.get('xgboost_ready', False):
            self.insights.append("📊 XGBoost模型就緒：適合高精度梯度提升預測")
        elif prediction_readiness.get('rf_ready', False):
            self.insights.append("🌲 隨機森林模型就緒：適合作為預測基線")
        
        # 時間特性洞察
        time_span = prediction_readiness.get('time_span_days', 0)
        if time_span >= 7:
            self.insights.append(f"📅 週期性分析就緒：{time_span}天數據支援週期模式學習")
        
        # VD站點洞察
        vd_count = prediction_readiness.get('unique_vd_stations', 0)
        if vd_count >= 5:
            self.insights.append(f"🛣️ 多站點預測：{vd_count}個VD站點支援路段級預測")
    
    def evaluate_ai_model_suitability(self) -> Dict[str, Any]:
        """評估AI模型適用性"""
        print("🤖 評估AI模型適用性...")
        
        characteristics = self.analysis_results.get('data_characteristics', {})
        prediction_readiness = characteristics.get('prediction_readiness', {})
        
        # 模型評估
        model_suitability = {
            'lstm_time_series': self._evaluate_lstm(prediction_readiness),
            'xgboost_ensemble': self._evaluate_xgboost(prediction_readiness),
            'random_forest_baseline': self._evaluate_rf(prediction_readiness)
        }
        
        # 生成推薦
        recommendations = self._generate_recommendations(model_suitability)
        
        ai_evaluation = {
            'model_suitability': model_suitability,
            'recommendations': recommendations,
            'data_readiness': prediction_readiness
        }
        
        self.analysis_results['ai_evaluation'] = ai_evaluation
        return ai_evaluation
    
    def _evaluate_lstm(self, readiness: Dict[str, Any]) -> Dict[str, Any]:
        """評估LSTM模型"""
        records = readiness.get('total_records', 0)
        time_span = readiness.get('time_span_days', 0)
        completeness = readiness.get('avg_completeness', 0)
        
        # 評分計算
        data_score = min(50, records / 1000)
        time_score = min(30, time_span * 4)
        quality_score = min(20, completeness * 0.2)
        
        total_score = data_score + time_score + quality_score
        
        return {
            'score': round(total_score, 1),
            'suitable': readiness.get('lstm_ready', False),
            'pros': ['時間序列學習', '長期依賴捕捉', '15分鐘預測'],
            'cons': ['需要大量數據', '訓練時間長', '需要GPU'],
            'expected_accuracy': '85-92%'
        }
    
    def _evaluate_xgboost(self, readiness: Dict[str, Any]) -> Dict[str, Any]:
        """評估XGBoost模型"""
        records = readiness.get('total_records', 0)
        completeness = readiness.get('avg_completeness', 0)
        
        score = min(100, (records / 10000) * 40 + completeness * 0.6)
        
        return {
            'score': round(score, 1),
            'suitable': readiness.get('xgboost_ready', False),
            'pros': ['高預測精度', '特徵重要性', '快速訓練'],
            'cons': ['需要調參', '記憶體需求高'],
            'expected_accuracy': '80-88%'
        }
    
    def _evaluate_rf(self, readiness: Dict[str, Any]) -> Dict[str, Any]:
        """評估隨機森林模型"""
        records = readiness.get('total_records', 0)
        completeness = readiness.get('avg_completeness', 0)
        
        score = min(100, (records / 5000) * 35 + completeness * 0.65)
        
        return {
            'score': round(score, 1),
            'suitable': readiness.get('rf_ready', False),
            'pros': ['穩定基線', '易於理解', '抗過擬合'],
            'cons': ['精度較低', '預測速度慢'],
            'expected_accuracy': '75-82%'
        }
    
    def _generate_recommendations(self, suitability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成模型推薦"""
        scored_models = []
        
        for model_name, info in suitability.items():
            if info['suitable']:
                scored_models.append((model_name, info['score'], info))
        
        # 按評分排序
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        priorities = ['🥇 首選', '🥈 次選', '🥉 備選']
        
        for i, (model_name, score, info) in enumerate(scored_models[:3]):
            priority = priorities[i] if i < len(priorities) else f"#{i+1}"
            
            recommendations.append({
                'rank': i + 1,
                'model': model_name,
                'priority': priority,
                'score': score,
                'reason': f"適合交通預測的{model_name}模型",
                'pros': info['pros'][:2],
                'expected_accuracy': info['expected_accuracy']
            })
        
        return recommendations
    
    def _calculate_balance_score(self) -> float:
        """計算數據平衡性評分"""
        if 'target_peak' in self.datasets and 'target_offpeak' in self.datasets:
            peak_count = len(self.datasets['target_peak'])
            offpeak_count = len(self.datasets['target_offpeak'])
            
            if peak_count > 0 and offpeak_count > 0:
                balance_ratio = min(peak_count, offpeak_count) / max(peak_count, offpeak_count)
                return balance_ratio * 100
        
        return 50
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成綜合分析報告"""
        print("📋 生成綜合分析報告...")
        
        # 確保已完成必要分析
        if not self.analysis_results.get('data_characteristics'):
            self.analyze_data_characteristics()
        
        if not self.analysis_results.get('ai_evaluation'):
            self.evaluate_ai_model_suitability()
        
        # 建構報告
        report = {
            'metadata': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analyzer_version': 'simplified_core_v1.0',
                'total_insights': len(self.insights)
            },
            'data_summary': self.analysis_results['data_characteristics']['data_summary'],
            'quality_assessment': self.analysis_results['data_characteristics']['quality_metrics'],
            'time_coverage': self.analysis_results['data_characteristics']['time_coverage'],
            'prediction_readiness': self.analysis_results['data_characteristics']['prediction_readiness'],
            'ai_model_evaluation': self.analysis_results['ai_evaluation'],
            'key_insights': self.insights,
            'actionable_recommendations': self._generate_actionable_recommendations()
        }
        
        return report
    
    def _generate_actionable_recommendations(self) -> List[str]:
        """生成可行動建議"""
        recommendations = []
        
        ai_eval = self.analysis_results.get('ai_evaluation', {})
        recommendations_list = ai_eval.get('recommendations', [])
        
        # 基於頂級推薦的建議
        if recommendations_list:
            top_model = recommendations_list[0]
            recommendations.append(f"🎯 立即開發 {top_model['model']} 模型")
            recommendations.append(f"📈 預期準確率: {top_model['expected_accuracy']}")
        
        # 基於數據狀況的建議
        prediction_readiness = self.analysis_results.get('data_characteristics', {}).get('prediction_readiness', {})
        
        if prediction_readiness.get('lstm_ready', False):
            recommendations.append("🚀 數據符合LSTM要求 - 可開始深度學習開發")
        
        if prediction_readiness.get('unique_vd_stations', 0) >= 5:
            recommendations.append("🛣️ 多站點預測能力 - 可實現路段級交通預測")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """保存分析報告"""
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            filename = f"simplified_traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = output_dir / filename
        
        # 確保所有數據都可以JSON序列化
        safe_report = self._safe_json_convert(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(safe_report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 報告已保存: {output_path}")
    
    def _safe_json_convert(self, obj):
        """安全的JSON轉換"""
        if isinstance(obj, dict):
            return {str(k): self._safe_json_convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._safe_json_convert(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            try:
                return str(obj)
            except:
                return None
    
    def print_summary(self):
        """列印分析摘要"""
        print("\n" + "="*60)
        print("📊 簡化版交通流量分析摘要")
        print("="*60)
        
        # 數據概覽
        if 'data_characteristics' in self.analysis_results:
            char = self.analysis_results['data_characteristics']
            time_coverage = char['time_coverage']
            quality_metrics = char['quality_metrics']
            prediction_readiness = char['prediction_readiness']
            
            print(f"📈 數據規模:")
            print(f"   總記錄數: {time_coverage.get('total_records', 0):,}")
            print(f"   時間跨度: {time_coverage.get('date_span_days', 0)} 天")
            print(f"   整體品質: {quality_metrics.get('overall_quality', 0):.1f}/100")
            
            print(f"\n🎯 預測就緒度:")
            print(f"   LSTM就緒: {'✅ 是' if prediction_readiness.get('lstm_ready') else '❌ 否'}")
            print(f"   XGBoost就緒: {'✅ 是' if prediction_readiness.get('xgboost_ready') else '❌ 否'}")
            print(f"   隨機森林就緒: {'✅ 是' if prediction_readiness.get('rf_ready') else '❌ 否'}")
            print(f"   VD站點數: {prediction_readiness.get('unique_vd_stations', 0)} 個")
        
        # AI模型推薦
        if 'ai_evaluation' in self.analysis_results:
            recommendations = self.analysis_results['ai_evaluation']['recommendations']
            print(f"\n🤖 AI模型推薦:")
            
            for rec in recommendations:
                print(f"   {rec['priority']} {rec['model']}: {rec['score']:.1f}分")
                print(f"      預期準確率: {rec['expected_accuracy']}")
        
        # 關鍵洞察
        if self.insights:
            print(f"\n💡 關鍵發現:")
            for i, insight in enumerate(self.insights, 1):
                print(f"   {i}. {insight}")
        
        print("\n✅ 簡化版分析完成！")


def quick_analyze(base_folder: str = "data", sample_rate: float = 1.0) -> Dict[str, Any]:
    """快速分析函數"""
    print("🚀 啟動簡化版交通流量分析...")
    
    analyzer = SimplifiedTrafficAnalyzer(base_folder)
    
    if analyzer.load_data(sample_rate=sample_rate):
        analyzer.analyze_data_characteristics()
        analyzer.evaluate_ai_model_suitability()
        report = analyzer.generate_comprehensive_report()
        analyzer.print_summary()
        analyzer.save_report(report)
        return report
    else:
        print("❌ 數據載入失敗")
        return {}


if __name__ == "__main__":
    print("🚀 啟動簡化版交通流量分析器")
    print("=" * 60)
    print("🎯 核心功能:")
    print("   📊 數據載入和特性分析")
    print("   🤖 AI模型適用性評估")
    print("   📋 分析報告生成")
    print("=" * 60)
    
    quick_analyze()