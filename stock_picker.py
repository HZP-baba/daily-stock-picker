import tushare as ts
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# 从环境变量获取Tushare token（GitHub Secrets自动注入）
ts.set_token(os.environ.get('TUSHARE_TOKEN'))
pro = ts.pro_api()

# -------------------------- 消息面选股：关键词匹配 --------------------------
KEYWORD_MAP = {
    '人工智能': ['AI', '人工智能', '大模型', 'ChatGPT', '深度学习', '机器视觉'],
    '新能源': ['新能源', '光伏', '锂电池', '储能', '风电', '氢能'],
    '半导体': ['半导体', '芯片', '集成电路', '光刻机', '先进制程'],
    '医药': ['医药', '创新药', '医疗器械', '医保', '生物医药'],
    '消费': ['消费', '白酒', '食品', '零售', '免税'],
    '汽车': ['汽车', '新能源汽车', '自动驾驶', '零部件', '智能驾驶'],
    '基建': ['基建', '水利', '铁路', '公路', '新型城镇化'],
    '军工': ['军工', '国防', '航天', '航空', '船舶']
}

def get_recent_news():
    """获取新浪财经最新新闻"""
    try:
        url = 'https://news.sina.com.cn/c/finance/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        
        import re
        news_list = []
        titles = re.findall(r'<h2 class="news-title"><a href="(.*?)" target="_blank">(.*?)</a></h2>', response.text)
        for link, title in titles[:30]:
            news_list.append({'title': title, 'link': link})
        return news_list
    except Exception as e:
        print(f"获取新闻失败: {e}")
        return []

def match_news_stocks(news_list):
    """根据新闻关键词匹配股票"""
    matched_stocks = []
    used_codes = set()
    
    for news in news_list:
        title = news['title']
        for concept, keywords in KEYWORD_MAP.items():
            for keyword in keywords:
                if keyword in title:
                    try:
                        df = pro.concept_detail(concept=concept, fields='ts_code,name,industry')
                        if df is None or len(df) == 0:
                            continue
                        
                        # 获取前一个交易日的市值数据
                        trade_date = (datetime.now()-timedelta(days=1)).strftime('%Y%m%d')
                        daily_basic = pro.daily_basic(
                            trade_date=trade_date,
                            fields='ts_code,circ_mv,pe'
                        )
                        df = df.merge(daily_basic, on='ts_code')
                        # 筛选流通市值50-500亿，市盈率<100
                        df = df[(df['circ_mv'] >= 500000) & (df['circ_mv'] <= 5000000) & (df['pe'] < 100)]
                        
                        if len(df) > 0 and df.iloc[0]['ts_code'] not in used_codes:
                            stock = df.iloc[0]
                            matched_stocks.append({
                                'ts_code': stock['ts_code'],
                                'name': stock['name'],
                                'industry': stock['industry'],
                                'reason': f"受益于：{title}"
                            })
                            used_codes.add(stock['ts_code'])
                            break
                    except Exception as e:
                        print(f"匹配股票失败: {e}")
                        continue
            if len(matched_stocks) >= 3:
                break
    
    return matched_stocks[:3]

# -------------------------- 技术面选股：融合四大理论 --------------------------
def technical_select_stocks():
    """技术面选股，选出3只符合突破条件的股票"""
    selected_stocks = []
    
    # 获取A股所有股票列表（排除ST和退市股）
    stock_list = pro.stock_basic(
        exchange='', 
        list_status='L', 
        fields='ts_code,symbol,name,industry,list_date'
    )
    stock_list = stock_list[~stock_list['name'].str.contains(r'ST|\*ST|退', na=False)]
    
    # 处理股票（为了速度，只处理最近30天有交易的前500只）
    for _, stock in stock_list.head(500).iterrows():
        ts_code = stock['ts_code']
        try:
            # 获取最近90个交易日数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now()-timedelta(days=120)).strftime('%Y%m%d')
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if len(df) < 60:
                continue
            
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 1. 均线多头排列（道氏理论+趋势理论）
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma60'] = df['close'].rolling(60).mean()
            df['ma120'] = df['close'].rolling(120).mean()
            
            last = df.iloc[-1]
            if not (last['ma20'] > last['ma60'] > last['ma120']):
                continue
            
            # 2. 股价在20日均线之上
            if last['close'] < last['ma20']:
                continue
            
            # 3. 突破3个月新高（技术突破）
            three_month_high = df['high'].max()
            if last['close'] < three_month_high * 0.995:  # 允许0.5%的误差
                continue
            
            # 4. 成交量放大1.5倍以上（量能确认）
            avg_volume = df['vol'].tail(20).mean()
            if last['vol'] < avg_volume * 1.5:
                continue
            
            # 5. 趋势健康（斜率30-60度）
            slope = (last['ma20'] - df.iloc[-20]['ma20']) / df.iloc[-20]['ma20'] * 100
            if slope < 5 or slope > 30:
                continue
            
            # 6. 排除近期涨幅过大（避免追高）
            month_gain = (last['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close'] * 100
            if month_gain > 30:
                continue
            
            selected_stocks.append({
                'ts_code': ts_code,
                'name': stock['name'],
                'industry': stock['industry'],
                'close': round(last['close'], 2),
                'volume_ratio': round(last['vol'] / avg_volume, 2),
                'slope': round(slope, 2),
                'month_gain': round(month_gain, 2)
            })
            
        except Exception as e:
            print(f"处理{ts_code}失败: {e}")
            continue
    
    # 按量比排序，取前3只
    selected_stocks = sorted(selected_stocks, key=lambda x: x['volume_ratio'], reverse=True)[:3]
    return selected_stocks

# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    print("开始选股...")
    
    news = get_recent_news()
    news_stocks = match_news_stocks(news)
    tech_stocks = technical_select_stocks()
    
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'news_stocks': news_stocks,
        'tech_stocks': tech_stocks,
        'hot_news': news[:5]
    }
    
    # 保存结果到JSON文件
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("选股完成！")
    print(f"消息面股票：{[s['name'] for s in news_stocks]}")
    print(f"技术面股票：{[s['name'] for s in tech_stocks]}")
