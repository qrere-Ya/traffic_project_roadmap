# src/enhanced_predictor.py - 多源融合預測器

"""
多源融合預測器 - 統一預測接口
=============================

核心功能：
1. 🔗 整合VD單源預測器與VD+eTag融合引擎
2. 🧠 智能預測模式選擇（單源/多源/混合）
3. 📊 預測結果比較和驗證
4. 🎯 統一的15分鐘預測接口
5. 📈 預測性能監控和評估

預測模式：
- VD_ONLY: 僅使用VD數據預測（現有系統）
- FUSION_ONLY: 僅使用VD+eTag融合預測
- HYBRID: 智能混合預測（推薦模式）
- ENSEMBLE: 多模型集成預測

應用場景：
- 實時交通預測服務
- 交通狀況監控系統
- 智慧導航系統整合
- 交通管理決策支援

作者: 交通預測專案團隊
日期: 2025-07-29
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
import logging

# 導入現有的預測模組
try:
    from predictor import TrafficPredictionSystem
    VD_PREDICTOR_AVAILABLE = True
    print("✅ VD單源預測器已導入")
except ImportError:
    VD_PREDICTOR_AVAILABLE = False
    warnings.warn("VD單源預測器導入失敗")

try:
    from fusion_engine import VDETagFusionEngine
    FUSION_ENGINE_AVAILABLE = True
    print("✅ VD+eTag融合引擎已導入")
except ImportError:
    FUSION_ENGINE_AVAILABLE = False
    warnings.warn("融合引擎導入失敗")

warnings.filterwarnings('ignore')


class PredictionMode(Enum):
    """預測模式枚舉"""
    VD_ONLY = "vd_only"
    FUSION_ONLY = "fusion_only"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class DataAvailability(Enum):
    """數據可用性狀態"""
    VD_ONLY = "vd_only"
    VD_AND_ETAG = "vd_and_etag"
    INSUFFICIENT = "insufficient"


class EnhancedTrafficPredictor:
    """多源融合預測器主控制器"""
    
    def __init__(self, base_folder: str = "data", 
                 default_mode: PredictionMode = PredictionMode.HYBRID):
        self.base_folder = Path(base_folder)
        self.models_folder = Path("models")
        self.default_mode = default_mode
        
        # 初始化預測器
        self.vd_predictor = None
        self.fusion_engine = None
        
        # 性能監控
        self.prediction_history = []
        self.performance_metrics = {}
        
        # 配置日誌
        self._setup_logging()
        
        print("🚀 多源融合智能預測器初始化")
        print(f"   📁 數據目錄: {self.base_folder}")
        print(f"   🎯 默認模式: {default_mode.value}")
        
        # 檢測並載入可用預測器
        self._initialize_predictors()
    
    def _setup_logging(self):
        """設置日誌系統"""
        log_folder = Path("logs")
        log_folder.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_folder / 'enhanced_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EnhancedPredictor')
    
    def _initialize_predictors(self):
        """初始化預測器"""
        print("🔧 初始化預測器組件...")
        
        # 初始化VD單源預測器
        if VD_PREDICTOR_AVAILABLE:
            try:
                self.vd_predictor = TrafficPredictionSystem(str(self.base_folder))
                
                # 嘗試載入已訓練的VD模型
                vd_models_folder = self.models_folder / "vd_models"
                if vd_models_folder.exists():
                    try:
                        self.vd_predictor.load_models()
                        print("   ✅ VD單源預測器已載入")
                    except:
                        print("   ⚠️ VD模型載入失敗，需要重新訓練")
                else:
                    print("   ⚠️ VD模型不存在，需要先訓練")
                    
            except Exception as e:
                print(f"   ❌ VD預測器初始化失敗: {e}")
                self.vd_predictor = None
        
        # 初始化融合引擎
        if FUSION_ENGINE_AVAILABLE:
            try:
                self.fusion_engine = VDETagFusionEngine(str(self.base_folder))
                
                # 嘗試載入已訓練的融合模型
                fusion_models_folder = self.models_folder / "fusion_models"
                if fusion_models_folder.exists():
                    try:
                        self.fusion_engine.load_fusion_models()
                        print("   ✅ VD+eTag融合引擎已載入")
                    except:
                        print("   ⚠️ 融合模型載入失敗，需要重新訓練")
                else:
                    print("   ⚠️ 融合模型不存在，需要先訓練")
                    
            except Exception as e:
                print(f"   ❌ 融合引擎初始化失敗: {e}")
                self.fusion_engine = None
    
    def assess_data_availability(self, input_data: pd.DataFrame) -> DataAvailability:
        """評估輸入數據的可用性"""
        if input_data.empty:
            return DataAvailability.INSUFFICIENT
        
        # 檢查VD基本欄位
        vd_required_fields = ['vd_id', 'speed', 'volume_total', 'occupancy']
        has_vd = all(field in input_data.columns for field in vd_required_fields)
        
        # 檢查eTag相關欄位
        etag_fields = ['etag_travel_time_primary', 'etag_speed_primary']
        has_etag = any(field in input_data.columns for field in etag_fields)
        
        if has_vd and has_etag:
            return DataAvailability.VD_AND_ETAG
        elif has_vd:
            return DataAvailability.VD_ONLY
        else:
            return DataAvailability.INSUFFICIENT
    
    def select_prediction_mode(self, input_data: pd.DataFrame, 
                             requested_mode: Optional[PredictionMode] = None) -> PredictionMode:
        """智能選擇預測模式"""
        # 如果明確指定模式，優先使用
        if requested_mode:
            return requested_mode
        
        # 評估數據可用性
        data_availability = self.assess_data_availability(input_data)
        
        # 根據數據可用性和模型可用性選擇模式
        if data_availability == DataAvailability.VD_AND_ETAG:
            if self.fusion_engine and hasattr(self.fusion_engine, 'fusion_xgboost') and self.fusion_engine.fusion_xgboost.is_trained:
                if self.vd_predictor and (self.vd_predictor.xgboost_predictor.is_trained or self.vd_predictor.rf_predictor.is_trained):
                    return PredictionMode.HYBRID  # 最佳模式
                else:
                    return PredictionMode.FUSION_ONLY
            else:
                return PredictionMode.VD_ONLY
        elif data_availability == DataAvailability.VD_ONLY:
            return PredictionMode.VD_ONLY
        else:
            # 數據不足，返回默認模式
            return self.default_mode
    
    def predict_15_minutes(self, input_data: pd.DataFrame, 
                          mode: Optional[PredictionMode] = None,
                          include_diagnostics: bool = True) -> Dict[str, Any]:
        """統一的15分鐘預測接口"""
        prediction_start_time = datetime.now()
        
        try:
            # 輸入數據驗證
            if input_data.empty:
                return self._create_error_response("輸入數據為空", prediction_start_time)
            
            # 選擇預測模式
            selected_mode = self.select_prediction_mode(input_data, mode)
            data_availability = self.assess_data_availability(input_data)
            
            self.logger.info(f"執行預測 - 模式: {selected_mode.value}, 數據: {data_availability.value}")
            
            # 根據模式執行預測
            if selected_mode == PredictionMode.VD_ONLY:
                result = self._predict_vd_only(input_data)
            elif selected_mode == PredictionMode.FUSION_ONLY:
                result = self._predict_fusion_only(input_data)
            elif selected_mode == PredictionMode.HYBRID:
                result = self._predict_hybrid(input_data)
            elif selected_mode == PredictionMode.ENSEMBLE:
                result = self._predict_ensemble(input_data)
            else:
                return self._create_error_response(f"不支援的預測模式: {selected_mode}", prediction_start_time)
            
            # 添加診斷信息
            if include_diagnostics:
                result['diagnostics'] = self._generate_diagnostics(
                    input_data, selected_mode, data_availability, prediction_start_time
                )
            
            # 記錄預測歷史
            self._record_prediction(result, selected_mode, data_availability)
            
            return result
            
        except Exception as e:
            self.logger.error(f"預測失敗: {e}")
            return self._create_error_response(str(e), prediction_start_time)
    
    def _predict_vd_only(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """VD單源預測"""
        if not self.vd_predictor:
            raise ValueError("VD預測器未初始化")
        
        if not (self.vd_predictor.xgboost_predictor.is_trained or self.vd_predictor.rf_predictor.is_trained):
            raise ValueError("VD預測器模型未訓練")
        
        # 使用現有的VD預測系統
        prediction = self.vd_predictor.predict_15_minutes(input_data)
        
        # 包裝結果
        if 'error' not in prediction:
            enhanced_result = {
                'prediction_mode': PredictionMode.VD_ONLY.value,
                'predicted_speed': prediction['predicted_speed'],
                'traffic_status': prediction['traffic_status'],
                'confidence': prediction['confidence'],
                'prediction_time': prediction['prediction_time'],
                'data_sources': ['VD車輛偵測器'],
                'model_info': {
                    'primary_model': self._get_vd_primary_model(),
                    'models_used': prediction['metadata']['models_used'],
                    'features_used': 'VD瞬時特徵'
                },
                'individual_predictions': prediction.get('individual_predictions', {}),
                'advantages': [
                    '高頻率更新(1分鐘)',
                    '瞬時交通狀態',
                    '成熟穩定技術'
                ]
            }
        else:
            enhanced_result = prediction
            enhanced_result['prediction_mode'] = PredictionMode.VD_ONLY.value
        
        return enhanced_result
    
    def _predict_fusion_only(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """融合預測"""
        if not self.fusion_engine:
            raise ValueError("融合引擎未初始化")
        
        if not self.fusion_engine.fusion_xgboost.is_trained:
            raise ValueError("融合引擎模型未訓練")
        
        # 使用融合引擎預測
        prediction = self.fusion_engine.predict_15_minutes(input_data)
        
        # 包裝結果
        if 'error' not in prediction:
            enhanced_result = {
                'prediction_mode': PredictionMode.FUSION_ONLY.value,
                'predicted_speed': prediction['predicted_speed'],
                'traffic_status': prediction['traffic_status'],
                'confidence': prediction['confidence'],
                'prediction_time': prediction['prediction_time'],
                'data_sources': ['VD車輛偵測器', 'eTag電子標籤'],
                'model_info': {
                    'primary_model': 'fusion_xgboost',
                    'models_used': prediction['fusion_models_used'],
                    'features_used': 'VD+eTag融合特徵'
                },
                'fusion_advantages': prediction.get('fusion_advantages', {}),
                'individual_predictions': prediction.get('individual_predictions', {}),
                'advantages': [
                    '多源數據驗證',
                    '空間一致性檢查', 
                    '區間+瞬時特徵融合',
                    '預測穩定性提升'
                ]
            }
        else:
            enhanced_result = prediction
            enhanced_result['prediction_mode'] = PredictionMode.FUSION_ONLY.value
        
        return enhanced_result
    
    def _predict_hybrid(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """混合預測 - 智能選擇最佳結果"""
        hybrid_results = {
            'prediction_mode': PredictionMode.HYBRID.value,
            'prediction_time': datetime.now().isoformat(),
            'data_sources': ['VD車輛偵測器', 'eTag電子標籤'],
            'hybrid_strategy': 'intelligent_selection'
        }
        
        predictions = {}
        
        # 獲取VD預測
        try:
            vd_result = self._predict_vd_only(input_data)
            if 'error' not in vd_result:
                predictions['vd'] = {
                    'predicted_speed': vd_result['predicted_speed'],
                    'confidence': vd_result['confidence'],
                    'source': 'VD單源',
                    'weight': 0.4
                }
        except Exception as e:
            self.logger.warning(f"VD預測失敗: {e}")
        
        # 獲取融合預測
        try:
            fusion_result = self._predict_fusion_only(input_data)
            if 'error' not in fusion_result:
                predictions['fusion'] = {
                    'predicted_speed': fusion_result['predicted_speed'],
                    'confidence': fusion_result['confidence'],
                    'source': 'VD+eTag融合',
                    'weight': 0.6  # 融合預測權重更高
                }
        except Exception as e:
            self.logger.warning(f"融合預測失敗: {e}")
        
        if not predictions:
            return self._create_error_response("所有預測模式都失敗", datetime.now())
        
        # 智能混合策略
        if len(predictions) == 1:
            # 只有一個預測可用
            single_pred = list(predictions.values())[0]
            hybrid_results.update({
                'predicted_speed': single_pred['predicted_speed'],
                'traffic_status': self._classify_traffic_status(single_pred['predicted_speed']),
                'confidence': single_pred['confidence'],
                'selected_model': single_pred['source'],
                'reason': '僅有一個模型可用'
            })
        else:
            # 多個預測可用，進行智能選擇
            hybrid_results.update(self._intelligent_hybrid_selection(predictions, input_data))
        
        hybrid_results['individual_predictions'] = predictions
        hybrid_results['advantages'] = [
            '智能模型選擇',
            '多源預測驗證',
            '自適應預測策略',
            '最優結果選擇'
        ]
        
        return hybrid_results
    
    def _predict_ensemble(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """集成預測 - 多模型結果融合"""
        ensemble_results = {
            'prediction_mode': PredictionMode.ENSEMBLE.value,
            'prediction_time': datetime.now().isoformat(),
            'data_sources': ['VD車輛偵測器', 'eTag電子標籤'],
            'ensemble_strategy': 'weighted_average'
        }
        
        predictions = []
        weights = []
        
        # 收集所有可用預測
        try:
            vd_result = self._predict_vd_only(input_data)
            if 'error' not in vd_result:
                predictions.append(vd_result['predicted_speed'])
                weights.append(vd_result['confidence'] * 0.01)  # 轉換為0-1權重
        except:
            pass
        
        try:
            fusion_result = self._predict_fusion_only(input_data)
            if 'error' not in fusion_result:
                predictions.append(fusion_result['predicted_speed'])
                weights.append(fusion_result['confidence'] * 0.01 * 1.2)  # 融合預測額外加權
        except:
            pass
        
        if not predictions:
            return self._create_error_response("沒有可用的預測結果", datetime.now())
        
        # 加權平均集成
        if len(predictions) == 1:
            ensemble_speed = predictions[0]
            ensemble_confidence = weights[0] * 100
        else:
            total_weight = sum(weights)
            ensemble_speed = sum(p * w for p, w in zip(predictions, weights)) / total_weight
            ensemble_confidence = min(95, total_weight * 100 / len(predictions))
        
        ensemble_results.update({
            'predicted_speed': round(ensemble_speed, 1),
            'traffic_status': self._classify_traffic_status(ensemble_speed),
            'confidence': round(ensemble_confidence, 0),
            'models_used': len(predictions),
            'ensemble_variance': round(np.var(predictions), 2),
            'prediction_consistency': self._calculate_prediction_consistency(predictions),
            'advantages': [
                '多模型集成降低風險',
                '預測結果更穩定',
                '集成學習效應',
                '異常值抑制'
            ]
        })
        
        return ensemble_results
    
    def _intelligent_hybrid_selection(self, predictions: Dict, input_data: pd.DataFrame) -> Dict[str, Any]:
        """智能混合選擇策略"""
        
        # 提取預測值和置信度
        speeds = [pred['predicted_speed'] for pred in predictions.values()]
        confidences = [pred['confidence'] for pred in predictions.values()]
        
        # 計算預測一致性
        speed_variance = np.var(speeds)
        consistency_score = max(0, 1 - speed_variance / 100)  # 歸一化一致性分數
        
        # 選擇策略
        if speed_variance < 5:  # 預測非常一致
            # 使用置信度最高的預測
            best_idx = np.argmax(confidences)
            best_pred = list(predictions.values())[best_idx]
            strategy = "高一致性-選擇最高置信度"
            
            return {
                'predicted_speed': best_pred['predicted_speed'],
                'traffic_status': self._classify_traffic_status(best_pred['predicted_speed']),
                'confidence': best_pred['confidence'],
                'selected_model': best_pred['source'],
                'selection_strategy': strategy,
                'consistency_score': round(consistency_score, 3),
                'speed_variance': round(speed_variance, 2)
            }
        
        elif speed_variance < 15:  # 預測中等一致
            # 加權平均，偏向融合預測
            if 'fusion' in predictions:
                fusion_weight = 0.7
                vd_weight = 0.3
            else:
                fusion_weight = 0.5
                vd_weight = 0.5
            
            weighted_speed = (predictions.get('fusion', {}).get('predicted_speed', 0) * fusion_weight +
                            predictions.get('vd', {}).get('predicted_speed', 0) * vd_weight)
            weighted_confidence = max(confidences) * consistency_score
            
            return {
                'predicted_speed': round(weighted_speed, 1),
                'traffic_status': self._classify_traffic_status(weighted_speed),
                'confidence': round(weighted_confidence, 0),
                'selected_model': '智能加權平均',
                'selection_strategy': "中等一致性-加權融合",
                'consistency_score': round(consistency_score, 3),
                'speed_variance': round(speed_variance, 2)
            }
        
        else:  # 預測差異較大
            # 優先選擇融合預測（更穩定）
            if 'fusion' in predictions:
                selected = predictions['fusion']
                strategy = "預測分歧-選擇融合模型"
            else:
                selected = predictions['vd']
                strategy = "預測分歧-選擇VD模型"
            
            return {
                'predicted_speed': selected['predicted_speed'],
                'traffic_status': self._classify_traffic_status(selected['predicted_speed']),
                'confidence': max(50, selected['confidence'] * consistency_score),  # 降低置信度
                'selected_model': selected['source'],
                'selection_strategy': strategy,
                'consistency_score': round(consistency_score, 3),
                'speed_variance': round(speed_variance, 2),
                'warning': '預測模型間存在較大差異'
            }
    
    def _get_vd_primary_model(self) -> str:
        """獲取VD主要模型類型"""
        if not self.vd_predictor:
            return "unknown"
        
        if self.vd_predictor.xgboost_predictor.is_trained:
            return "xgboost"
        elif self.vd_predictor.rf_predictor.is_trained:
            return "random_forest"
        elif hasattr(self.vd_predictor, 'lstm_predictor') and self.vd_predictor.lstm_predictor:
            return "lstm"
        else:
            return "unknown"
    
    def _classify_traffic_status(self, speed: float) -> str:
        """交通狀態分類"""
        if speed >= 80:
            return "暢通🟢"
        elif speed >= 50:
            return "緩慢🟡"
        else:
            return "擁堵🔴"
    
    def _calculate_prediction_consistency(self, predictions: List[float]) -> str:
        """計算預測一致性等級"""
        if len(predictions) < 2:
            return "單一預測"
        
        variance = np.var(predictions)
        if variance < 5:
            return "高度一致"
        elif variance < 15:
            return "中等一致"
        else:
            return "存在分歧"
    
    def _generate_diagnostics(self, input_data: pd.DataFrame, mode: PredictionMode, 
                            availability: DataAvailability, start_time: datetime) -> Dict[str, Any]:
        """生成診斷信息"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        diagnostics = {
            'prediction_mode': mode.value,
            'data_availability': availability.value,
            'processing_time_ms': round(processing_time, 2),
            'input_data_summary': {
                'records': len(input_data),
                'columns': len(input_data.columns),
                'vd_features': sum(1 for col in input_data.columns if col.startswith('vd_')),
                'etag_features': sum(1 for col in input_data.columns if col.startswith('etag_')),
                'fusion_features': sum(1 for col in input_data.columns if col.startswith('fusion_'))
            },
            'predictor_status': {
                'vd_predictor_available': self.vd_predictor is not None,
                'vd_models_trained': self._check_vd_models_trained(),
                'fusion_engine_available': self.fusion_engine is not None,
                'fusion_models_trained': self._check_fusion_models_trained()
            },
            'system_recommendations': self._generate_system_recommendations(availability)
        }
        
        return diagnostics
    
    def _check_vd_models_trained(self) -> bool:
        """檢查VD模型訓練狀態"""
        if not self.vd_predictor:
            return False
        return (self.vd_predictor.xgboost_predictor.is_trained or 
                self.vd_predictor.rf_predictor.is_trained)
    
    def _check_fusion_models_trained(self) -> bool:
        """檢查融合模型訓練狀態"""
        if not self.fusion_engine:
            return False
        return self.fusion_engine.fusion_xgboost.is_trained
    
    def _generate_system_recommendations(self, availability: DataAvailability) -> List[str]:
        """生成系統建議"""
        recommendations = []
        
        if availability == DataAvailability.INSUFFICIENT:
            recommendations.append("輸入數據不足，請確保包含基本VD特徵")
        
        if availability == DataAvailability.VD_ONLY:
            recommendations.append("建議整合eTag數據以啟用融合預測功能")
        
        if not self._check_vd_models_trained():
            recommendations.append("VD模型需要訓練或重新載入")
        
        if not self._check_fusion_models_trained() and availability == DataAvailability.VD_AND_ETAG:
            recommendations.append("融合模型需要訓練以發揮多源數據優勢")
        
        if len(recommendations) == 0:
            recommendations.append("系統運行良好，所有功能已就緒")
        
        return recommendations
    
    def _create_error_response(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
        """創建錯誤響應"""
        return {
            'error': error_message,
            'prediction_time': datetime.now().isoformat(),
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'system_status': {
                'vd_predictor_available': self.vd_predictor is not None,
                'fusion_engine_available': self.fusion_engine is not None
            }
        }
    
    def _record_prediction(self, result: Dict[str, Any], mode: PredictionMode, 
                          availability: DataAvailability):
        """記錄預測歷史"""
        if 'error' not in result:
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'mode': mode.value,
                'data_availability': availability.value,
                'predicted_speed': result.get('predicted_speed'),
                'confidence': result.get('confidence'),
                'processing_time_ms': result.get('diagnostics', {}).get('processing_time_ms', 0)
            })
            
            # 保持歷史記錄在合理範圍內
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """獲取預測統計信息"""
        if not self.prediction_history:
            return {'message': '暫無預測歷史'}
        
        df = pd.DataFrame(self.prediction_history)
        
        return {
            'total_predictions': len(df),
            'mode_distribution': df['mode'].value_counts().to_dict(),
            'average_confidence': round(df['confidence'].mean(), 1),
            'average_processing_time_ms': round(df['processing_time_ms'].mean(), 2),
            'speed_statistics': {
                'mean': round(df['predicted_speed'].mean(), 1),
                'std': round(df['predicted_speed'].std(), 1),
                'min': round(df['predicted_speed'].min(), 1),
                'max': round(df['predicted_speed'].max(), 1)
            },
            'recent_predictions': df.tail(5).to_dict('records')
        }
    
    def compare_prediction_modes(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """比較不同預測模式的結果"""
        comparison_results = {
            'comparison_time': datetime.now().isoformat(),
            'data_availability': self.assess_data_availability(input_data).value,
            'mode_results': {},
            'analysis': {}
        }
        
        # 測試各種預測模式
        modes_to_test = [PredictionMode.VD_ONLY, PredictionMode.FUSION_ONLY, 
                        PredictionMode.HYBRID, PredictionMode.ENSEMBLE]
        
        for mode in modes_to_test:
            try:
                start_time = datetime.now()
                result = self.predict_15_minutes(input_data, mode, include_diagnostics=False)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if 'error' not in result:
                    comparison_results['mode_results'][mode.value] = {
                        'predicted_speed': result['predicted_speed'],
                        'confidence': result['confidence'],
                        'traffic_status': result['traffic_status'],
                        'processing_time_ms': round(processing_time, 2),
                        'success': True
                    }
                else:
                    comparison_results['mode_results'][mode.value] = {
                        'error': result['error'],
                        'success': False
                    }
                    
            except Exception as e:
                comparison_results['mode_results'][mode.value] = {
                    'error': str(e),
                    'success': False
                }
        
        # 分析比較結果
        successful_predictions = {k: v for k, v in comparison_results['mode_results'].items() 
                                if v.get('success', False)}
        
        if len(successful_predictions) >= 2:
            speeds = [pred['predicted_speed'] for pred in successful_predictions.values()]
            confidences = [pred['confidence'] for pred in successful_predictions.values()]
            
            comparison_results['analysis'] = {
                'prediction_consistency': self._calculate_prediction_consistency(speeds),
                'speed_variance': round(np.var(speeds), 2),
                'confidence_range': [min(confidences), max(confidences)],
                'recommended_mode': self._recommend_best_mode(successful_predictions),
                'convergence_analysis': self._analyze_prediction_convergence(speeds)
            }
        
        return comparison_results
    
    def _recommend_best_mode(self, successful_predictions: Dict) -> str:
        """推薦最佳預測模式"""
        if 'hybrid' in successful_predictions:
            return 'hybrid'
        elif 'fusion_only' in successful_predictions:
            return 'fusion_only'
        elif 'ensemble' in successful_predictions:
            return 'ensemble'
        elif 'vd_only' in successful_predictions:
            return 'vd_only'
        else:
            return 'none_available'
    
    def _analyze_prediction_convergence(self, speeds: List[float]) -> str:
        """分析預測收斂性"""
        if len(speeds) < 2:
            return "單一預測"
        
        speed_range = max(speeds) - min(speeds)
        if speed_range < 3:
            return "高度收斂"
        elif speed_range < 8:
            return "中等收斂"
        else:
            return "發散預測"
    
    def export_prediction_history(self, filepath: Optional[str] = None) -> str:
        """導出預測歷史"""
        if not self.prediction_history:
            raise ValueError("無預測歷史可導出")
        
        df = pd.DataFrame(self.prediction_history)
        
        if not filepath:
            output_folder = Path("outputs/prediction_logs")
            output_folder.mkdir(parents=True, exist_ok=True)
            filepath = output_folder / f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return str(filepath)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能分析報告"""
        if not self.prediction_history:
            return {'error': '無預測歷史數據'}
        
        df = pd.DataFrame(self.prediction_history)
        
        # 基本統計
        basic_stats = {
            'total_predictions': len(df),
            'time_period': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'mode_usage': df['mode'].value_counts().to_dict(),
            'data_availability_distribution': df['data_availability'].value_counts().to_dict()
        }
        
        # 性能指標
        performance_metrics = {
            'average_confidence': round(df['confidence'].mean(), 1),
            'confidence_std': round(df['confidence'].std(), 1),
            'average_processing_time_ms': round(df['processing_time_ms'].mean(), 2),
            'processing_time_95th_percentile': round(df['processing_time_ms'].quantile(0.95), 2)
        }
        
        # 預測速度分析
        speed_analysis = {
            'speed_distribution': {
                'mean': round(df['predicted_speed'].mean(), 1),
                'std': round(df['predicted_speed'].std(), 1),
                'median': round(df['predicted_speed'].median(), 1),
                'min': round(df['predicted_speed'].min(), 1),
                'max': round(df['predicted_speed'].max(), 1)
            },
            'traffic_status_distribution': self._analyze_traffic_status_distribution(df)
        }
        
        # 模式效能比較
        mode_performance = {}
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode]
            mode_performance[mode] = {
                'count': len(mode_df),
                'avg_confidence': round(mode_df['confidence'].mean(), 1),
                'avg_processing_time': round(mode_df['processing_time_ms'].mean(), 2),
                'speed_consistency': round(mode_df['predicted_speed'].std(), 2)
            }
        
        return {
            'report_generation_time': datetime.now().isoformat(),
            'basic_statistics': basic_stats,
            'performance_metrics': performance_metrics,
            'speed_analysis': speed_analysis,
            'mode_performance_comparison': mode_performance,
            'system_health': self._assess_system_health(df)
        }
    
    def _analyze_traffic_status_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """分析交通狀態分布"""
        status_counts = {'暢通🟢': 0, '緩慢🟡': 0, '擁堵🔴': 0}
        
        for speed in df['predicted_speed']:
            status = self._classify_traffic_status(speed)
            if status in status_counts:
                status_counts[status] += 1
        
        return status_counts
    
    def _assess_system_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        """評估系統健康狀況"""
        recent_df = df.tail(100)  # 最近100次預測
        
        health_score = 100
        issues = []
        
        # 檢查處理時間
        avg_processing_time = recent_df['processing_time_ms'].mean()
        if avg_processing_time > 1000:  # 超過1秒
            health_score -= 20
            issues.append("預測響應時間較長")
        
        # 檢查置信度
        avg_confidence = recent_df['confidence'].mean()
        if avg_confidence < 70:
            health_score -= 15
            issues.append("預測置信度偏低")
        
        # 檢查預測穩定性
        speed_volatility = recent_df['predicted_speed'].std()
        if speed_volatility > 20:
            health_score -= 10
            issues.append("預測結果波動較大")
        
        # 檢查模式多樣性
        mode_diversity = len(recent_df['mode'].unique())
        if mode_diversity < 2:
            health_score -= 5
            issues.append("預測模式單一")
        
        health_level = "優秀" if health_score >= 90 else "良好" if health_score >= 70 else "需改善"
        
        return {
            'health_score': max(0, health_score),
            'health_level': health_level,
            'issues': issues,
            'recommendations': self._generate_health_recommendations(health_score, issues)
        }
    
    def _generate_health_recommendations(self, health_score: int, issues: List[str]) -> List[str]:
        """生成健康狀況改善建議"""
        recommendations = []
        
        if health_score < 70:
            recommendations.append("建議檢查系統資源和模型狀態")
        
        if "預測響應時間較長" in issues:
            recommendations.append("優化特徵工程流程或增加計算資源")
        
        if "預測置信度偏低" in issues:
            recommendations.append("重新訓練模型或檢查輸入數據品質")
        
        if "預測結果波動較大" in issues:
            recommendations.append("檢查數據源穩定性或調整預測參數")
        
        if "預測模式單一" in issues:
            recommendations.append("啟用多源數據融合以提升預測多樣性")
        
        if not recommendations:
            recommendations.append("系統運行良好，繼續保持")
        
        return recommendations


# ============================================================
# 便利函數和示範用法
# ============================================================

def create_enhanced_predictor(base_folder: str = "data", 
                            mode: PredictionMode = PredictionMode.HYBRID) -> EnhancedTrafficPredictor:
    """創建增強預測器"""
    return EnhancedTrafficPredictor(base_folder, mode)


def quick_enhanced_prediction_demo():
    """快速增強預測演示"""
    print("🎯 多源融合預測器演示")
    print("-" * 50)
    
    try:
        # 創建增強預測器
        predictor = EnhancedTrafficPredictor()
        
        # 創建模擬多源數據
        current_time = datetime.now()
        mock_multi_source_data = pd.DataFrame({
            'date': [current_time.strftime('%Y-%m-%d')],
            'update_time': [current_time],
            'vd_id': ['VD-N1-N-25-台北'],
            # VD數據
            'speed': [75],
            'volume_total': [25],
            'occupancy': [45],
            'volume_small': [20],
            'volume_large': [3],
            'volume_truck': [2],
            # eTag數據
            'etag_travel_time_primary': [90],
            'etag_speed_primary': [72],
            'etag_volume_primary': [85],
            # 融合特徵
            'spatial_consistency_score': [0.85],
            'speed_difference': [3.0],
            'flow_correlation_index': [0.78]
        })
        
        print("🔍 數據可用性評估:")
        availability = predictor.assess_data_availability(mock_multi_source_data)
        print(f"   數據狀態: {availability.value}")
        
        print(f"\n🎯 執行多模式預測比較:")
        comparison = predictor.compare_prediction_modes(mock_multi_source_data)
        
        for mode, result in comparison['mode_results'].items():
            if result.get('success'):
                print(f"   {mode}: {result['predicted_speed']}km/h (信心{result['confidence']}%)")
            else:
                print(f"   {mode}: 失敗 - {result.get('error', '未知錯誤')}")
        
        if 'analysis' in comparison:
            analysis = comparison['analysis']
            print(f"\n📊 預測分析:")
            print(f"   一致性: {analysis['prediction_consistency']}")
            print(f"   建議模式: {analysis['recommended_mode']}")
            print(f"   收斂性: {analysis['convergence_analysis']}")
        
        # 執行混合預測
        print(f"\n🔗 執行智能混合預測:")
        hybrid_result = predictor.predict_15_minutes(
            mock_multi_source_data, 
            mode=PredictionMode.HYBRID
        )
        
        if 'error' not in hybrid_result:
            print(f"   🚗 預測速度: {hybrid_result['predicted_speed']} km/h")
            print(f"   🚥 交通狀態: {hybrid_result['traffic_status']}")
            print(f"   🎯 置信度: {hybrid_result['confidence']}%")
            print(f"   🤖 選擇模型: {hybrid_result.get('selected_model', '集成')}")
            print(f"   💡 選擇策略: {hybrid_result.get('selection_strategy', '智能選擇')}")
        else:
            print(f"   ❌ 混合預測失敗: {hybrid_result['error']}")
        
        return hybrid_result
        
    except Exception as e:
        print(f"❌ 增強預測演示失敗: {e}")
        return None


def benchmark_prediction_modes(base_folder: str = "data", iterations: int = 10):
    """預測模式基準測試"""
    print("📊 預測模式性能基準測試")
    print("-" * 50)
    
    predictor = EnhancedTrafficPredictor(base_folder)
    
    # 創建測試數據
    test_data = pd.DataFrame({
        'date': ['2025-07-29'] * iterations,
        'update_time': [datetime.now() + timedelta(minutes=i*5) for i in range(iterations)],
        'vd_id': ['VD-N1-N-25-台北'] * iterations,
        'speed': np.random.normal(70, 15, iterations),
        'volume_total': np.random.poisson(25, iterations),
        'occupancy': np.random.uniform(30, 80, iterations),
        'etag_travel_time_primary': np.random.normal(95, 20, iterations),
        'etag_speed_primary': np.random.normal(68, 12, iterations),
        'spatial_consistency_score': np.random.uniform(0.7, 0.95, iterations)
    })
    
    benchmark_results = {}
    
    for mode in [PredictionMode.VD_ONLY, PredictionMode.FUSION_ONLY, 
                PredictionMode.HYBRID, PredictionMode.ENSEMBLE]:
        print(f"\n🧪 測試模式: {mode.value}")
        
        times = []
        successes = 0
        
        for i in range(iterations):
            try:
                start = datetime.now()
                result = predictor.predict_15_minutes(
                    test_data.iloc[i:i+1], mode, include_diagnostics=False
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000
                
                if 'error' not in result:
                    times.append(elapsed)
                    successes += 1
                    
            except Exception as e:
                print(f"   ⚠️ 第{i+1}次測試失敗: {e}")
        
        if times:
            benchmark_results[mode.value] = {
                'success_rate': f"{successes}/{iterations}",
                'avg_time_ms': round(np.mean(times), 2),
                'min_time_ms': round(min(times), 2),
                'max_time_ms': round(max(times), 2),
                'std_time_ms': round(np.std(times), 2)
            }
            
            print(f"   成功率: {successes}/{iterations}")
            print(f"   平均時間: {benchmark_results[mode.value]['avg_time_ms']}ms")
        else:
            benchmark_results[mode.value] = {'error': '所有測試失敗'}
    
    print(f"\n📊 基準測試完成")
    return benchmark_results


if __name__ == "__main__":
    print("🚀 多源融合智能預測器")
    print("=" * 70)
    print("🎯 整合預測模式:")
    print("   📡 VD_ONLY - VD單源預測")
    print("   🔗 FUSION_ONLY - VD+eTag融合預測")
    print("   🧠 HYBRID - 智能混合預測")
    print("   📊 ENSEMBLE - 多模型集成預測")
    print("=" * 70)
    
    # 創建增強預測器
    enhanced_predictor = EnhancedTrafficPredictor()
    
    # 檢查系統狀態
    print(f"🔧 系統狀態檢查:")
    print(f"   VD預測器: {'✅ 就緒' if enhanced_predictor.vd_predictor else '❌ 不可用'}")
    print(f"   融合引擎: {'✅ 就緒' if enhanced_predictor.fusion_engine else '❌ 不可用'}")
    
    if enhanced_predictor.vd_predictor:
        vd_trained = enhanced_predictor._check_vd_models_trained()
        print(f"   VD模型: {'✅ 已訓練' if vd_trained else '⚠️ 需訓練'}")
    
    if enhanced_predictor.fusion_engine:
        fusion_trained = enhanced_predictor._check_fusion_models_trained()
        print(f"   融合模型: {'✅ 已訓練' if fusion_trained else '⚠️ 需訓練'}")
    
    # 檢查是否可以進行演示
    can_demo = (enhanced_predictor.vd_predictor and 
                enhanced_predictor._check_vd_models_trained()) or \
               (enhanced_predictor.fusion_engine and 
                enhanced_predictor._check_fusion_models_trained())
    
    if can_demo:
        response = input(f"\n執行多源融合預測演示？(y/N): ")
        
        if response.lower() in ['y', 'yes']:
            demo_result = quick_enhanced_prediction_demo()
            
            if demo_result:
                print(f"\n📈 預測統計信息:")
                stats = enhanced_predictor.get_prediction_statistics()
                if 'message' not in stats:
                    print(f"   總預測次數: {stats['total_predictions']}")
                    print(f"   平均置信度: {stats['average_confidence']}%")
                    print(f"   平均處理時間: {stats['average_processing_time_ms']}ms")
                
                # 詢問是否進行基準測試
                benchmark_response = input(f"\n執行性能基準測試？(y/N): ")
                if benchmark_response.lower() in ['y', 'yes']:
                    benchmark_results = benchmark_prediction_modes(iterations=5)
                    
                    print(f"\n🏆 性能排行:")
                    sorted_modes = sorted(
                        [(k, v) for k, v in benchmark_results.items() if 'avg_time_ms' in v],
                        key=lambda x: x[1]['avg_time_ms']
                    )
                    
                    for i, (mode, metrics) in enumerate(sorted_modes, 1):
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                        print(f"   {emoji} {mode}: {metrics['avg_time_ms']}ms")
    else:
        print(f"\n💡 系統準備建議:")
        if not enhanced_predictor._check_vd_models_trained():
            print("   1. 訓練VD模型: python src/predictor.py")
        if not enhanced_predictor._check_fusion_models_trained():
            print("   2. 訓練融合模型: python src/fusion_engine.py")
        print("   3. 然後執行: python src/enhanced_predictor.py")
    
    print(f"\n🎯 多源融合預測器特色:")
    print("   🧠 智能模式選擇 - 根據數據自動選擇最佳預測模式")
    print("   🔗 混合預測策略 - VD單源+融合預測智能組合")
    print("   📊 集成學習 - 多模型結果融合降低預測風險")
    print("   📈 性能監控 - 實時追蹤預測準確率和響應時間")
    print("   🎯 統一接口 - 一個API支援所有預測模式")
    
    print(f"\n🚀 Ready for Enhanced Multi-Source Prediction! 🚀")