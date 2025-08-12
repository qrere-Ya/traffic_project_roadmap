# etag_quick_fix.py - 快速修正版

"""
eTag處理器快速修正版
==================

針對測試失敗的關鍵問題進行修正：
1. ✅ 添加 archive_folder 屬性
2. ✅ 修正 Flow 數據提取邏輯
3. ✅ 強化 XML 解析

使用方法：
python etag_quick_fix.py
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


def quick_test_xml_parsing():
    """快速測試XML解析"""
    print("🔍 快速測試XML解析功能")
    print("-" * 40)
    
    # 測試XML內容（基於您提供的格式）
    test_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<ETagPairLiveList>
<ETagPairLive>
<ETagPairId>01F0017N-01F0005N</ETagPairId>
<StartETagStatus>0</StartETagStatus>
<EndETagStatus>0</EndETagStatus>
<Flows>
<Flow>
<VehicleType>31</VehicleType>
<TravelTime>73</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>59</SpaceMeanSpeed>
<VehicleCount>37</VehicleCount>
</Flow>
<Flow>
<VehicleType>32</VehicleType>
<TravelTime>73</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>59</SpaceMeanSpeed>
<VehicleCount>18</VehicleCount>
</Flow>
<Flow>
<VehicleType>41</VehicleType>
<TravelTime>0</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>0</SpaceMeanSpeed>
<VehicleCount>0</VehicleCount>
</Flow>
</Flows>
<StartTime>2025-06-20T23:55:00+08:00</StartTime>
<EndTime>2025-06-21T00:00:00+08:00</EndTime>
<DataCollectTime>2025-06-21T00:00:00+08:00</DataCollectTime>
</ETagPairLive>
</ETagPairLiveList>'''
    
    try:
        # 解析XML
        root = ET.fromstring(test_xml)
        print(f"✅ XML解析成功")
        print(f"   根元素: {root.tag}")
        
        # 查找ETagPairLive
        etag_pairs = []
        for elem in root.iter():
            if 'ETagPair' in elem.tag and 'Live' in elem.tag:
                etag_pairs.append(elem)
        
        print(f"✅ 找到 {len(etag_pairs)} 個ETagPairLive")
        
        for etag_pair in etag_pairs:
            # 提取ETagPairId
            pair_id = None
            for child in etag_pair:
                if 'ETagPairId' in child.tag:
                    pair_id = child.text
                    break
            
            print(f"✅ ETagPairId: {pair_id}")
            
            # 查找Flows
            flows_containers = []
            for child in etag_pair:
                if child.tag == 'Flows':
                    flows_containers.append(child)
            
            print(f"✅ 找到 {len(flows_containers)} 個Flows容器")
            
            # 提取Flow
            flow_count = 0
            valid_flows = 0
            
            for flows_container in flows_containers:
                for flow in flows_container:
                    if flow.tag == 'Flow':
                        flow_count += 1
                        
                        # 提取Flow數據
                        travel_time = 0
                        vehicle_count = 0
                        vehicle_type = ""
                        
                        for flow_child in flow:
                            if flow_child.tag == 'TravelTime':
                                travel_time = int(flow_child.text or '0')
                            elif flow_child.tag == 'VehicleCount':
                                vehicle_count = int(flow_child.text or '0')
                            elif flow_child.tag == 'VehicleType':
                                vehicle_type = flow_child.text or ''
                        
                        print(f"   Flow {flow_count}: 車種={vehicle_type}, 時間={travel_time}s, 數量={vehicle_count}")
                        
                        if travel_time > 0 and vehicle_count > 0:
                            valid_flows += 1
                            print(f"      ✅ 有效Flow")
                        else:
                            print(f"      ❌ 無效Flow")
            
            print(f"📊 總計: {flow_count} 個Flow, {valid_flows} 個有效")
            
            # 提取時間
            for child in etag_pair:
                if child.tag in ['DataCollectTime', 'EndTime', 'StartTime']:
                    print(f"🕐 {child.tag}: {child.text}")
            
            return valid_flows > 0
        
    except Exception as e:
        print(f"❌ XML解析失敗: {e}")
        return False


def quick_test_etag_processor():
    """快速測試eTag處理器"""
    print("\n🧪 快速測試eTag處理器")
    print("-" * 40)
    
    try:
        # 導入並初始化
        import sys
        sys.path.append('src')
        from etag_processor import ETagProcessor
        
        processor = ETagProcessor(debug=True)
        
        # 檢查屬性
        attributes = ['raw_etag_folder', 'processed_etag_folder', 'archive_folder', 'target_pairs']
        for attr in attributes:
            if hasattr(processor, attr):
                value = getattr(processor, attr)
                print(f"✅ {attr}: {value}")
            else:
                print(f"❌ 缺少屬性: {attr}")
                return False
        
        print(f"✅ 處理器初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ 處理器測試失敗: {e}")
        return False


def create_test_xml_file():
    """創建測試XML檔案"""
    print("\n📁 創建測試XML檔案")
    print("-" * 40)
    
    test_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<ETagPairLiveList>
<ETagPairLive>
<ETagPairId>01F0017N-01F0005N</ETagPairId>
<Flows>
<Flow>
<VehicleType>31</VehicleType>
<TravelTime>73</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>59</SpaceMeanSpeed>
<VehicleCount>37</VehicleCount>
</Flow>
</Flows>
<DataCollectTime>2025-06-24T12:00:00+08:00</DataCollectTime>
</ETagPairLive>
</ETagPairLiveList>'''
    
    try:
        # 創建測試目錄
        test_dir = Path("data/raw/etag/2025-06-24")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建測試檔案
        test_file = test_dir / "ETagPairLive_1200.xml.gz"
        with gzip.open(test_file, 'wt', encoding='utf-8') as f:
            f.write(test_xml)
        
        print(f"✅ 測試檔案已創建: {test_file}")
        return test_file
        
    except Exception as e:
        print(f"❌ 創建測試檔案失敗: {e}")
        return None


def test_processing_with_real_file():
    """使用真實檔案測試處理"""
    print("\n🔄 測試實際檔案處理")
    print("-" * 40)
    
    # 創建測試檔案
    test_file = create_test_xml_file()
    if not test_file:
        return False
    
    try:
        import sys
        sys.path.append('src')
        from etag_processor import ETagProcessor
        
        processor = ETagProcessor(debug=True)
        
        # 測試單檔處理
        print("📝 測試單檔處理...")
        result = processor.process_single_file(test_file, '2025-06-24')
        
        if result:
            print(f"✅ 單檔處理成功: {len(result)} 筆記錄")
            for record in result:
                print(f"   記錄: {record}")
        else:
            print(f"❌ 單檔處理失敗")
            return False
        
        # 測試批次處理
        print("\n📦 測試批次處理...")
        batch_result = processor.process_all_dates()
        
        if batch_result.get('success'):
            print(f"✅ 批次處理成功")
            print(f"   成功日期: {batch_result.get('successful_dates', 0)}")
            
            # 檢查輸出檔案
            output_file = Path("data/processed/etag/2025-06-24/etag_travel_time.csv")
            if output_file.exists():
                df = pd.read_csv(output_file)
                print(f"✅ 輸出檔案已生成: {len(df)} 筆記錄")
                print(f"   欄位: {list(df.columns)}")
            else:
                print(f"❌ 輸出檔案不存在")
                return False
            
            # 檢查歸檔
            archive_file = processor.archive_folder / "2025-06-24" / test_file.name
            if archive_file.exists():
                print(f"✅ 檔案已歸檔: {archive_file}")
            else:
                print(f"⚠️ 檔案未歸檔")
            
        else:
            print(f"❌ 批次處理失敗")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 檔案處理測試失敗: {e}")
        return False


def main():
    """主程序"""
    print("🚀 eTag處理器快速修正測試")
    print("=" * 50)
    
    tests = [
        ("XML解析功能", quick_test_xml_parsing),
        ("處理器初始化", quick_test_etag_processor),
        ("實際檔案處理", test_processing_with_real_file)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 執行測試: {test_name}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name}: 通過")
        else:
            print(f"❌ {test_name}: 失敗")
    
    # 測試摘要
    print("\n" + "=" * 50)
    print("📋 快速修正測試摘要")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 測試結果: {passed}/{total} 通過 ({passed/total*100:.1f}%)")
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 所有測試通過！eTag處理器修正成功")
        
        print(f"\n💡 修正重點:")
        print("   ✅ 添加 archive_folder 屬性")
        print("   ✅ 修正 Flows/Flow XML 解析")
        print("   ✅ 強化時間欄位提取")
        print("   ✅ 完善調試輸出")
        
        print(f"\n🎯 現在可以執行:")
        print("   python src/etag_processor.py")
        print("   python test_etag_processor.py")
        
    else:
        print(f"\n⚠️ 仍有 {total-passed} 個問題需要解決")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 修正完成，準備處理真實數據！")
    else:
        print("\n🔧 請檢查並解決剩餘問題")