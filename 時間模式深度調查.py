"""
時間模式深度調查
===============
分析數據採樣的時間分布情況。
"""

import pandas as pd
import sys
sys.path.append('src')

def investigate_time_patterns():
    """深度調查時間模式"""
    print("🕵️ 時間模式深度調查")
    print("=" * 40)
    
    # 載入數據
    df = pd.read_csv("data/processed/vd_data_cleaned.csv")
    df['update_time'] = pd.to_datetime(df['update_time'])
    df['hour'] = df['update_time'].dt.hour
    df['date'] = df['update_time'].dt.date
    
    print(f"📊 總記錄數: {len(df):,}")
    print(f"📅 日期範圍: {df['date'].min()} 到 {df['date'].max()}")
    
    # 1. 檢查每小時的數據分布
    print("\n🕐 每小時數據量分布:")
    hourly_counts = df['hour'].value_counts().sort_index()
    for hour, count in hourly_counts.items():
        bar = "█" * (count // 1000)
        print(f"   {hour:2d}點: {count:6,} 筆 {bar}")
    
    # 2. 檢查每日的數據分布
    print("\n📅 每日數據量分布:")
    daily_counts = df['date'].value_counts().sort_index()
    for date, count in daily_counts.items():
        print(f"   {date}: {count:6,} 筆")
    
    # 3. 檢查流量的真實尖峰
    print("\n🚗 每小時平均流量（排除零值）:")
    hourly_flow = df[df['volume_total'] > 0].groupby('hour')['volume_total'].agg(['count', 'mean']).round(2)
    hourly_flow = hourly_flow.sort_values('mean', ascending=False)
    
    print("   排名 | 時段 | 記錄數 | 平均流量")
    print("   -----|------|--------|----------")
    for i, (hour, data) in enumerate(hourly_flow.head(10).iterrows(), 1):
        print(f"   {i:2d}   | {hour:2d}點 | {data['count']:6.0f} | {data['mean']:6.2f}")
    
    # 4. 檢查數據採樣時間
    print("\n⏰ 具體採樣時間檢查:")
    unique_times = df['update_time'].dt.strftime('%Y-%m-%d %H:%M').value_counts()
    print(f"   總共有 {len(unique_times)} 個不同的時間點")
    print("   最常見的5個時間點:")
    for time_str, count in unique_times.head().items():
        print(f"   {time_str}: {count} 筆")
    
    # 5. VD設備的數據採樣檢查
    print("\n📡 VD設備數據分布:")
    vd_counts = df['vd_id'].value_counts()
    print(f"   總VD設備數: {len(vd_counts)}")
    print(f"   平均每設備記錄數: {vd_counts.mean():.1f}")
    print("   記錄數最多的5個設備:")
    for vd_id, count in vd_counts.head().items():
        print(f"   {vd_id}: {count} 筆")

if __name__ == "__main__":
    investigate_time_patterns()