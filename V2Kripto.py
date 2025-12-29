import pandas as pd
import numpy as np
import requests
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import concurrent.futures
import time
from datetime import datetime
import colorama
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

# Console Başlat
console = Console()
colorama.init(autoreset=True)
warnings.filterwarnings("ignore")

# ==========================================
# 1. DEV LİSTE (50+ COIN)
# ==========================================
# Hem senin istediklerin hem de piyasa devleri
TARGET_COINS = [
    # --- SENİN ÖZEL LİSTEN ---
    "BEAT", "RAVE", "CHZ", 
    
    # --- MEME COINS (Yüksek Risk/Getiri) ---
    "PEPE", "BONK", "FLOKI", "DOGE", "SHIB", "WIF", "BOME", "MEME",
    
    # --- AI & DATA ---
    "FET", "RNDR", "TAO", "WLD", "ARKM", "GRT", "NEAR", "AGIX", "OCEAN",
    
    # --- MAJÖRLER (L1) ---
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "TRX", "DOT", "MATIC", "LTC", "BCH",
    
    # --- YENİ NESİL L1/L2 ---
    "SUI", "SEI", "TIA", "APT", "INJ", "ARB", "OP", "STRK", "IMX", "KAS",
    
    # --- OYUN & METAVERSE ---
    "GALA", "SAND", "MANA", "AXS", "GMT", "PIXEL", "PORTAL",
    
    # --- DEFI & ALTYAPI ---
    "LINK", "UNI", "AAVE", "FIL", "ATOM", "ICP"
]

# ==========================================
# 2. MOTOR 1: BINANCE (HIZLI)
# ==========================================
def get_binance_data(symbol):
    """Binance Spot piyasasından anlık veri çeker"""
    # Binance genelde sonuna USDT ister
    pair = symbol.upper() + "USDT"
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": pair, "interval": "1d", "limit": 365}
    
    try:
        resp = requests.get(url, params=params, timeout=2)
        data = resp.json()
        
        # Hata kontrolü
        if isinstance(data, dict) and ('code' in data or 'msg' in data):
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
        ])
        
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Date', inplace=True)
        return df[numeric_cols]
    except:
        return pd.DataFrame()

# ==========================================
# 3. MOTOR 2: COINGECKO (YAVAŞ AMA KAPSAMLI)
# ==========================================
def get_gecko_data(symbol):
    """Binance'te yoksa buraya bakar (Deep Search)"""
    # 1. ID Bul
    coin_id = None
    try:
        search_url = "https://api.coingecko.com/api/v3/search"
        resp = requests.get(search_url, params={"query": symbol}, timeout=3)
        data = resp.json()
        for c in data.get('coins', []):
            if c['symbol'].upper() == symbol.upper():
                coin_id = c['id']
                break
        if not coin_id and data.get('coins'): coin_id = data['coins'][0]['id']
    except: pass
    
    if not coin_id: return pd.DataFrame()

    # 2. Veri Çek
    try:
        # Rate Limit yememek için kısa bekleme
        time.sleep(1.2) 
        chart_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        resp = requests.get(chart_url, params={"vs_currency": "usd", "days": "365", "interval": "daily"}, timeout=5)
        
        if resp.status_code == 429: # Çok hızlısın hatası
            time.sleep(5)
            return pd.DataFrame()

        data = resp.json()
        if 'prices' not in data: return pd.DataFrame()
        
        prices = data['prices']
        volumes = data['total_volumes']
        
        df = pd.DataFrame(prices, columns=['Date', 'Close'])
        df_vol = pd.DataFrame(volumes, columns=['Date', 'Volume'])
        
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        df_vol['Date'] = pd.to_datetime(df_vol['Date'], unit='ms')
        df_vol.set_index('Date', inplace=True)
        
        df = df.join(df_vol)
        df['Open'] = df['Close']; df['High'] = df['Close']; df['Low'] = df['Close']
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 4. ORTAK ANALİZ MOTORU
# ==========================================
def prepare_features(df):
    df = df.copy()
    
    # --- MEVCUTLAR ---
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI Hesaplama
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- YENİ EKLENENLER (Mühendislik Katmanı) ---
    
    # 1. MACD (Moving Average Convergence Divergence) - İvme Ölçümü
    kisa_ema = df['Close'].ewm(span=12, adjust=False).mean()
    uzun_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = kisa_ema - uzun_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal'] # Histogram
    
    # 2. Bollinger Bantları (Volatilite) - Standart Sapma
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['Std_Dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['Std_Dev'])
    
    # Fiyatın bant içindeki konumu (0 ile 1 arası normalize eder)
    # 1'e yakınsa üst banda yapışmış (Düşüş riski), 0'a yakınsa alt banda yapışmış (Alım fırsatı)
    df['B_Position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # 3. ATR (Average True Range) - Oynaklık/Risk Ölçümü
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # ATR'yi fiyata oranla (Normalize et ki Bitcoin ile Pepe kıyaslanabilsin)
    df['ATR_Norm'] = df['ATR'] / df['Close']

    # --- AI İÇİN FEATURES HAZIRLIĞI ---
    df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Change'] = df['Volume'].pct_change()
    df['RSI_Norm'] = df['RSI'] / 100.0
    
    # Lag (Gecikmeli) Veriler - AI geçmişe bakıp öğrensin
    features_to_lag = ['Log_Ret', 'RSI_Norm', 'MACD_Hist', 'B_Position', 'ATR_Norm']
    for col in features_to_lag:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2) # 2 gün öncesine de baksın

    # --- TARGET (HEDEF) BELİRLEME ---
    # Hedef: Gelecek 3 gün içinde %3 üzeri getiri var mı?
    threshold = 0.03 
    future_ret = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = (future_ret > threshold).astype(int)
    
    # NaN temizliği
    df.dropna(inplace=True)
    return df

def analyze_hybrid(symbol):
    source = "Binance"
    
    # ADIM 1: Binance Dene (Hızlı)
    df = get_binance_data(symbol)
    
    # ADIM 2: Olmadıysa Gecko Dene (Yavaş)
    if df.empty or len(df) < 50:
        source = "Gecko"
        df = get_gecko_data(symbol)
        
    if df.empty or len(df) < 30: return None
    
    # Analiz
    df = prepare_features(df)
    last_row = df.iloc[-1]
    
    # Teknik Skor
    tech_score = 0
    if last_row['RSI'] < 30: tech_score += 2
    elif last_row['RSI'] > 70: tech_score -= 2
    if last_row['Close'] > last_row['EMA_50']: tech_score += 1
    else: tech_score -= 1
    
   # XGBoost Kısmı
    feature_cols = [
        'Dist_EMA50', 'RSI_Norm', 'Log_Ret', 'Vol_Change', 
        'MACD_Hist', 'B_Position', 'ATR_Norm', # Yeni eklediklerimiz
        'Log_Ret_Lag1', 'RSI_Norm_Lag1', 'MACD_Hist_Lag1'
    ]
    X = df.iloc[:-3][feature_cols]
    y = df.iloc[:-3]['Target']
    
    try:
        model = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=42)
        model.fit(X, y)
        prob = model.predict_proba(pd.DataFrame([last_row[feature_cols]]))[0][1]
        
        decision = "NÖTR"; color = "yellow"
        if prob > 0.60: decision = "GÜÇLÜ AL"; color = "bold green"
        elif prob > 0.52: decision = "AL"; color = "green"
        elif prob < 0.40: decision = "SAT"; color = "red"
        
        return {
            "Sembol": symbol,
            "Fiyat": f"${last_row['Close']:.5f}",
            "Kaynak": source,
            "AI Güven": f"%{prob*100:.0f}",
            "Karar": decision,
            "Color": color
        }
    except: return None

# ==========================================
# 5. ANA PROGRAM
# ==========================================
def main():
    console.print(Panel.fit("[bold cyan]HELIOS V5.0 - HYBRID ULTIMATE[/bold cyan]\n[dim]50+ Coin | Binance (Hız) + Gecko (Yedek) Motoru[/dim]", border_style="blue"))
    
    results = []
    start = time.time()
    
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        task = progress.add_task("[cyan]Hibrit Tarama Başladı...", total=len(TARGET_COINS))
        
        # Binance hızlı olduğu için Thread sayısını artırdık
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(analyze_hybrid, coin): coin for coin in TARGET_COINS}
            
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res: results.append(res)
                progress.update(task, advance=1)

    # Rapor
    if not results:
        rprint("[red]Sonuç yok.[/red]"); return

    order = {"GÜÇLÜ AL": 5, "AL": 4, "NÖTR": 3, "SAT": 2, "GÜÇLÜ SAT": 1}
    results.sort(key=lambda x: order.get(x["Karar"], 0), reverse=True)

    table = Table(title=f"HELIOS V5.0 Sonuçları ({len(results)} Coin)", header_style="bold magenta")
    table.add_column("SEMBOL", style="cyan")
    table.add_column("KAYNAK", style="dim blue")
    table.add_column("FİYAT", justify="right")
    table.add_column("AI GÜVEN", justify="center")
    table.add_column("KARAR", justify="center")

    for r in results:
        table.add_row(
            r['Sembol'], r['Kaynak'], r['Fiyat'], r['AI Güven'],
            f"[{r['Color']}]{r['Karar']}[/{r['Color']}]"
        )

    console.print(table)
    rprint(f"\n[dim]Süre: {time.time() - start:.2f} saniye[/dim]")

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        pd.DataFrame(results).drop(columns=['Color']).to_excel(f"Helios_V5_Hybrid_{ts}.xlsx", index=False)
        rprint(f"[bold green]✔ Excel Kaydedildi.[/bold green]")
    except: pass

if __name__ == "__main__":
    main()