import streamlit as st
import yfinance as yf
import altair as alt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import datetime
import json
import os

@st.cache_data
def fetch_stock_info(symbol): 
    stock = yf.Ticker(symbol)
    return stock.info

@st.cache_data
def fetch_quarterly_financials(symbol): 
    stock = yf.Ticker(symbol)
    return stock.quarterly_financials.T

@st.cache_data
def fetch_annual_financials(symbol): 
    stock = yf.Ticker(symbol)
    return stock.financials.T

@st.cache_data
def fetch_quarterly_balance_sheet(symbol): 
    stock = yf.Ticker(symbol)
    return stock.quarterly_balance_sheet.T

@st.cache_data
def fetch_annual_balance_sheet(symbol): 
    stock = yf.Ticker(symbol)
    return stock.balance_sheet.T

@st.cache_data
def fetch_quarterly_cashflow(symbol): 
    stock = yf.Ticker(symbol)
    return stock.quarterly_cashflow.T

@st.cache_data
def fetch_annual_cashflow(symbol): 
    stock = yf.Ticker(symbol)
    return stock.cashflow.T

@st.cache_data
def fetch_weekly_price_history(symbol): 
    stock = yf.Ticker(symbol)
    return stock.history(period='1y', interval='1wk')

@st.cache_data
def fetch_earnings_growth(symbol):
    """Fetch yearly earnings and calculate their growth rate"""
    stock = yf.Ticker(symbol)
    earnings = stock.earnings
    if earnings is None or earnings.empty:
        return None, None
    
    # Calculate growth rate (using the last available years)
    if len(earnings) >= 2:
        earnings_sorted = earnings.sort_index(ascending=True)
        earliest_year = earnings_sorted.index[0]
        latest_year = earnings_sorted.index[-1]
        
        if earnings_sorted.loc[earliest_year, 'Earnings'] > 0 and earnings_sorted.loc[latest_year, 'Earnings'] > 0:
            years_diff = latest_year - earliest_year
            if years_diff > 0:
                cagr = (earnings_sorted.loc[latest_year, 'Earnings'] / 
                        earnings_sorted.loc[earliest_year, 'Earnings']) ** (1 / years_diff) - 1
                return earnings, cagr
    
    return earnings, None

def analyze_stock_with_perplexity(symbol, company_name, api_key):
    """Analyze news and market sentiment about a stock using Perplexity Sonar API"""
    url = "https://api.perplexity.ai/chat/completions"
    
    prompt = f"""Analizza le ultime notizie e il sentiment di mercato per {company_name} (simbolo: {symbol}).
    
Fornisci l'analisi finale senza mostrare alcun ragionamento o processo di ricerca. L'analisi deve:
1. Descrivere i principali movimenti di prezzo recenti e le loro cause
2. Menzionare eventi significativi o annunci dell'azienda
3. Indicare il sentiment del mercato e previsioni degli analisti
4. Spiegare le possibili implicazioni per gli investitori

Rispondi direttamente con l'analisi finale, senza introduzioni né commenti sul tuo processo di pensiero."""
    

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "search_context": {
            "search_context_size": "high"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Errore nell'analisi: {str(e)}"

def get_balance_sheet_field(df, field_options):
    """Trova il primo campo disponibile da una lista di opzioni"""
    for field in field_options:
        if field in df.columns:
            return field
    return None

def calculate_book_value_per_share(df_balance, info_dict):
    """Calcola Book Value per Azione"""
    # Possibili nomi per Total Equity
    equity_field_options = [
        'Total Equity Gross Minority Interest',
        'Stockholders Equity',
        'Total Stockholder Equity',
        'Total Equity'
    ]
    
    equity_field = get_balance_sheet_field(df_balance, equity_field_options)
    shares_outstanding = info_dict.get('sharesOutstanding')
    
    if equity_field and shares_outstanding:
        # Calcola book value per share per ogni periodo
        df_balance['Book Value per Share'] = df_balance[equity_field] / shares_outstanding
        return 'Book Value per Share', equity_field
    return None, None

def calculate_cash_flow_per_share(df_cashflow, df_financials):
    """Calcola Cash Flow per Azione"""
    # Possibili nomi per Operating Cash Flow
    ocf_field_options = [
        'Operating Cash Flow',
        'Cash From Operations',
        'Cash Flow From Operations',
        'Net Cash From Operating Activities'
    ]
    
    ocf_field = get_balance_sheet_field(df_cashflow, ocf_field_options)
    
    # Possibili nomi per Diluted Average Shares
    shares_field_options = [
        'Diluted Average Shares',
        'Weighted Average Diluted Shares',
        'Diluted Shares Outstanding'
    ]
    
    shares_field = get_balance_sheet_field(df_financials, shares_field_options)
    
    if ocf_field and shares_field:
        # Assicurati che entrambi i dataframe abbiano gli stessi indici
        if set(df_cashflow.index) == set(df_financials.index):
            # Calcola cash flow per share per ogni periodo
            df_cashflow['Cash Flow per Share'] = df_cashflow[ocf_field] / df_financials[shares_field]
            return 'Cash Flow per Share', ocf_field
    return None, None

# NUOVE FUNZIONI PER IL CALCOLO DEL VALORE INTRINSECO

def calculate_dcf_value(symbol, info, annual_financials, annual_cashflow, annual_balance_sheet):
    """Calcolo valore intrinseco con metodo DCF (Discounted Cash Flow)"""
    try:
        # Parametri per DCF
        discount_rate = st.slider("Tasso di Sconto (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100
        growth_rate_initial = st.slider("Tasso di Crescita Iniziale (%)", min_value=1.0, max_value=30.0, value=15.0, step=0.5) / 100
        growth_rate_terminal = st.slider("Tasso di Crescita Terminale (%)", min_value=1.0, max_value=5.0, value=2.5, step=0.1) / 100
        forecast_period = st.slider("Periodo di Previsione (anni)", min_value=5, max_value=20, value=10)
        
        # Ottieni Free Cash Flow più recente
        fcf_field_options = [
            'Free Cash Flow',
            'Operating Cash Flow',
            'Cash From Operations'
        ]
        
        # Trova il campo FCF disponibile
        fcf_field = get_balance_sheet_field(annual_cashflow, fcf_field_options)
        
        if fcf_field is None:
            st.warning("Dati di Free Cash Flow non disponibili.")
            return None
            
        # Prendi l'ultimo valore disponibile di FCF
        fcf = annual_cashflow[fcf_field].iloc[0]
        
        # Se FCF è negativo, usa la media degli ultimi 3 anni (se disponibile)
        if fcf <= 0 and len(annual_cashflow) >= 3:
            fcf = annual_cashflow[fcf_field].iloc[:3].mean()
            if fcf <= 0:
                st.warning("Free Cash Flow negativo o zero, impossibile calcolare DCF.")
                return None
        elif fcf <= 0:
            st.warning("Free Cash Flow negativo o zero, impossibile calcolare DCF.")
            return None
        
        # Numero di azioni in circolazione
        shares_outstanding = info.get('sharesOutstanding')
        if not shares_outstanding:
            st.warning("Numero di azioni in circolazione non disponibile.")
            return None
        
        # Calcola i flussi di cassa proiettati
        projected_cash_flows = []
        
        for year in range(1, forecast_period + 1):
            # Diminuisci gradualmente il tasso di crescita verso il tasso terminale
            if forecast_period > 1:
                weight = (forecast_period - year) / (forecast_period - 1)
                growth_rate = weight * growth_rate_initial + (1 - weight) * growth_rate_terminal
            else:
                growth_rate = growth_rate_terminal
                
            # Calcola il flusso di cassa per questo anno
            projected_cf = fcf * (1 + growth_rate) ** year
            
            # Calcola il valore presente di questo flusso di cassa
            present_value = projected_cf / (1 + discount_rate) ** year
            
            projected_cash_flows.append(present_value)
        
        # Calcola il valore terminale
        terminal_value = (fcf * (1 + growth_rate_terminal) ** forecast_period * (1 + growth_rate_terminal)) / (discount_rate - growth_rate_terminal)
        present_terminal_value = terminal_value / (1 + discount_rate) ** forecast_period
        
        # Calcola il valore totale dell'azienda
        enterprise_value = sum(projected_cash_flows) + present_terminal_value
        
        # Ottieni il debito totale e la liquidità
        debt_field_options = [
            'Debt',
            'Total Debt',
            'Long Term Debt',
            'Total Debt and Capital Lease Obligation'
        ]
        
        cash_field_options = [
            'Cash and Cash Equivalents',
            'Cash Cash Equivalents And Short Term Investments',
            'Cash And Short Term Investments'
        ]
        
        # Get data from the most recent balance sheet
        balance_sheet = annual_balance_sheet.iloc[0] if not annual_balance_sheet.empty else None
        
        if balance_sheet is not None:
            debt_field = get_balance_sheet_field(annual_balance_sheet, debt_field_options)
            cash_field = get_balance_sheet_field(annual_balance_sheet, cash_field_options)
            
            total_debt = balance_sheet[debt_field] if debt_field else 0
            total_cash = balance_sheet[cash_field] if cash_field else 0
            
            # Calcola equity value
            equity_value = enterprise_value - total_debt + total_cash
            
            # Calcola valore intrinseco per azione
            intrinsic_value_per_share = equity_value / shares_outstanding
            
            # Mostra il valore intrinseco calcolato
            st.metric(
                label="Valore Intrinseco per Azione (DCF)",
                value=f"${intrinsic_value_per_share:.2f}",
                delta=f"{(intrinsic_value_per_share / info.get('currentPrice', 1) - 1) * 100:.1f}% vs prezzo attuale"
            )
            
            # Mostra informazioni di base sui parametri utilizzati
            st.info(f"""
            **Parametri utilizzati:**
            - Free Cash Flow: ${fcf/1000000:.2f}M
            - Tasso di crescita iniziale: {growth_rate_initial*100:.1f}%
            - Tasso di crescita terminale: {growth_rate_terminal*100:.1f}%
            - Tasso di sconto: {discount_rate*100:.1f}%
            - Periodo di previsione: {forecast_period} anni
            """)
            
            return intrinsic_value_per_share
        else:
            st.warning("Dati del bilancio non disponibili.")
            return None
        
    except Exception as e:
        st.error(f"Errore nel calcolo DCF: {str(e)}")
        return None

def calculate_graham_value(symbol, info, annual_financials, annual_earnings_growth):
    """Calcolo valore intrinseco con Formula di Graham"""
    try:
        # Prova a ottenere EPS da diverse fonti
        eps = info.get('trailingEPS')
        
        # Se EPS non è disponibile nel campo trailingEPS, prova a cercarlo nei dati finanziari
        if not eps or eps <= 0:
            if not annual_financials.empty and 'Diluted EPS' in annual_financials.columns:
                # Prendi l'EPS più recente
                eps = annual_financials['Diluted EPS'].iloc[0]
            elif not annual_financials.empty and 'Basic EPS' in annual_financials.columns:
                eps = annual_financials['Basic EPS'].iloc[0]
        
        # Se ancora non disponibile, consenti all'utente di inserirlo manualmente
        if not eps or eps <= 0:
            st.warning("EPS non disponibile dai dati. Inserisci un valore manualmente.")
            eps = st.number_input("Inserisci EPS manualmente:", value=1.0, step=0.1)
        
        # Determina il tasso di crescita
        if annual_earnings_growth is not None:
            # Assicurati che annual_earnings_growth sia un float
            try:
                growth_rate_default = float(annual_earnings_growth) * 100
            except (TypeError, ValueError):
                growth_rate_default = 10.0
                st.warning("Tasso di crescita degli utili non valido. Utilizzando 10.0% come valore predefinito.")
        else:
            growth_rate_default = 10.0
            st.warning("Tasso di crescita degli utili non disponibile. Utilizzando 10.0% come valore predefinito.")
        
        # Assicurati che il valore sia nell'intervallo consentito
        growth_rate_default = min(max(growth_rate_default, 0.0), 30.0)
        
        # Parametri per la formula di Graham
        growth_rate = st.slider("Tasso di Crescita Annuale (%)", 
                               min_value=0.0, 
                               max_value=30.0, 
                               value=float(growth_rate_default),
                               step=0.5)
        
        bond_yield = st.slider("Rendimento Bond AAA (%)", 
                              min_value=1.0, 
                              max_value=10.0, 
                              value=4.5, 
                              step=0.1)
        
        # Formula originale di Graham: V = EPS × (8.5 + 2g) × 4.4 / Y
        # dove:
        # V = valore intrinseco
        # EPS = utile per azione
        # g = tasso di crescita degli utili (previsto per i prossimi 7-10 anni)
        # Y = rendimento corrente dei bond AAA
        
        intrinsic_value = eps * (8.5 + 2 * (growth_rate / 100)) * (4.4 / bond_yield)
        
        # Mostra il valore intrinseco calcolato
        st.metric(
            label="Valore Intrinseco per Azione (Graham)",
            value=f"${intrinsic_value:.2f}",
            delta=f"{(intrinsic_value / info.get('currentPrice', 1) - 1) * 100:.1f}% vs prezzo attuale"
        )
        
        # Mostra la formula utilizzata
        st.info(f"""
        **Formula di Graham utilizzata:** V = EPS × (8.5 + 2g) × 4.4 / Y
        
        **Parametri:**
        - EPS: ${eps:.2f}
        - Tasso di crescita (g): {growth_rate:.1f}%
        - Rendimento Bond AAA (Y): {bond_yield:.1f}%
        
        **Calcolo:** ${eps:.2f} × ({8.5 + (2 * growth_rate / 100):.2f}) × ({4.4 / bond_yield:.2f}) = ${intrinsic_value:.2f}
        """)
        
        return intrinsic_value
        
    except Exception as e:
        st.error(f"Errore nel calcolo con la Formula di Graham: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Mostra lo stack trace completo per il debug
        return None

# Funzione per formattare i valori in modo sicuro
def safe_format(value, format_str):
    try:
        if value is None:
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

# Main application ---------------------------------------------------

st.title('Analisi Fondamentale Azioni')
symbol = st.text_input('Inserisci il Ticker del Titolo (Come da [Watchlist](https://diramco.com/watchlist-azioni-diramco/) ad esempio per la Borsa Italiana ISP.MI)', 'AAPL')

information = fetch_stock_info(symbol)

st.header('Informazioni Titolo')

# Aggiungo il prezzo e la variazione percentuale con colorazione
current_price = information["currentPrice"]
percent_change = information["regularMarketChangePercent"]
color = "green" if percent_change >= 0 else "red"
change_sign = "+" if percent_change >= 0 else ""

st.subheader(f'Nome: {information["longName"]}')
# Uso subheader con HTML per mantenere lo stesso stile degli altri sottotitoli
st.markdown(f'<h3>Prezzo: {current_price:.2f} <span style="color:{color}">({change_sign}{percent_change:.2f}%)</span></h3>', unsafe_allow_html=True)
st.subheader(f'Capitalizzazione: {information["marketCap"]/1000000000:.1f} Miliardi')
st.subheader(f'Settore: {information["sector"]}')

# Creazione dei dati per le due colonne ----------------------------------------------------
data_colonna1 = {
    'Indicatori di Prezzo': [
        'Rendimento Dividendo', 
        'PE: Prezzo Utili', 
        'FPE: Prezzo Utili Futuri',
        'PB: Prezzo/Book'
    ],
    'Valore': [
        safe_format(information.get("dividendYield", 0), "{:.2f}%"),
        safe_format(information.get("trailingPE", 0), "{:.2f}"),
        safe_format(information.get("forwardPE", 0), "{:.2f}"),
        safe_format(information.get("priceToBook", 0), "{:.2f}")
    ]
}

data_colonna2 = {
    'Indicatori Chiave': [
        'Payout Ratio',
        'ROE', 
        'Debito/Equity',
        'Beta'  
    ],
    'Valore': [
        safe_format(information.get("payoutRatio", 0)*100, "{:.2f}%"),
        safe_format(information.get("returnOnEquity", 0), "{:.2f}"),
        safe_format(information.get("debtToEquity", 0), "{:.2f}%"),
        safe_format(information.get("beta", 0), "{:.1f}")
    ]
}

# Creazione dei dataframe
df1 = pd.DataFrame(data_colonna1)
df2 = pd.DataFrame(data_colonna2)

# Layout a due colonne
st.markdown("### Principali Indicatori Finanziari")

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df1, use_container_width=True, hide_index=True)

with col2:
    st.dataframe(df2, use_container_width=True, hide_index=True)


# GRAFICO ----------------------------------------------------------------
price_history = fetch_weekly_price_history(symbol)

st.header('Grafico')
st.markdown('Puoi visualizzare il grafico con Analisi Tecnica dettagliata e indicatori tecnici avanzati spiegati da IA in [questa pagina](https://diramco.com/analisi-tecnica-ai/)')

price_history_reset = price_history.rename_axis('Data').reset_index()
candle_stick_chart = go.Figure(data=[go.Candlestick(x=price_history_reset['Data'], 
                                open=price_history['Open'], 
                                low=price_history_reset['Low'],
                                high=price_history_reset['High'],
                                close=price_history['Close'])])

st.plotly_chart(candle_stick_chart, use_container_width=True)

# ANALISI NEWS ---------------------------------------------------------------
st.header('Analisi delle Notizie tramite IA')

# La tua chiave API Perplexity
perplexity_api_key = os.getenv("MY_SONAR_API_KEY")  # Sostituisci con la tua chiave reale

# Ottieni l'analisi AI direttamente
with st.spinner("Analizzando le notizie e il sentiment di mercato con IA..."):
    ai_analysis = analyze_stock_with_perplexity(symbol, information['longName'], perplexity_api_key)
    st.markdown(ai_analysis)
    
    # Aggiungi una nota informativa
    st.info("Analisi generata tramite IA basata sulle informazioni di mercato più recenti.")

# DATI FINANZIARI ----------------------------------------------------------
st.header('Dati Finanziari')
selection = st.segmented_control(label='Periodo', options=['Trimestrale', 'Annuale'], default='Annuale')

quarterly_financials = fetch_quarterly_financials(symbol)
annual_financials = fetch_annual_financials(symbol)
quarterly_balance_sheet = fetch_quarterly_balance_sheet(symbol)
annual_balance_sheet = fetch_annual_balance_sheet(symbol)
quarterly_cashflow = fetch_quarterly_cashflow(symbol)
annual_cashflow = fetch_annual_cashflow(symbol)

# Possibili nomi per il campo debt nel balance sheet
debt_field_options = [
    'Debt', 
    'Total Debt', 
    'Long Term Debt', 
    'Total Debt and Capital Lease Obligation',
    'Net Debt',
    'Short Term Debt',
    'Corporate Debt',
    'Long Term Debt Noncurrent'
]

if selection == 'Trimestrale':
    quarterly_financials = quarterly_financials.rename_axis('Quarter').reset_index()
    quarterly_financials['Quarter'] = quarterly_financials['Quarter'].astype(str)
    
    revenue_chart = alt.Chart(quarterly_financials).mark_bar(color='green').encode(
        x='Quarter:O',
        y='Total Revenue'
    ).properties(
        title={
            'text': f'Ricavi Totali - Trimestrale',
            'align': 'center',
            'anchor': 'middle'
        }
    )
    
    eps_chart = alt.Chart(quarterly_financials).mark_bar(color='green').encode(
        x='Quarter:O',
        y='Diluted EPS'
    ).properties(
        title={
            'text': f'Earnings per Share - Trimestrale',
            'align': 'center',
            'anchor': 'middle'
        }
    )
    
    shares_chart = alt.Chart(quarterly_financials).mark_bar(color='green').encode(
        x='Quarter:O',
        y='Diluted Average Shares'
    ).properties(
        title={
            'text': f'Azioni in Circolazione - Trimestrale',
            'align': 'center',
            'anchor': 'middle'
        }
    )
    
    # Correzione per il grafico del balance sheet
    quarterly_balance_sheet = quarterly_balance_sheet.rename_axis('Quarter').reset_index()
    quarterly_balance_sheet['Quarter'] = quarterly_balance_sheet['Quarter'].astype(str)
    
    # Calcola Book Value per Share
    bvps_field, equity_field = calculate_book_value_per_share(quarterly_balance_sheet, information)
    
    # Grafico Book Value per Share
    if bvps_field:
        bvps_chart = alt.Chart(quarterly_balance_sheet).mark_bar(color='green').encode(
            x='Quarter:O',
            y=alt.Y(bvps_field + ':Q',
                   title='Book Value per Share',
                   scale=alt.Scale(zero=True))
        ).properties(
            title={
                'text': f'Book Value per Share - Trimestrale',
                'align': 'center',
                'anchor': 'middle'
            }
        )
    
    # Cash Flow per Share
    quarterly_cashflow = quarterly_cashflow.rename_axis('Quarter').reset_index()
    quarterly_cashflow['Quarter'] = quarterly_cashflow['Quarter'].astype(str)
    
    cfps_field, ocf_field = calculate_cash_flow_per_share(quarterly_cashflow, quarterly_financials)
    
    # Grafico Cash Flow per Share
    if cfps_field:
        cfps_chart = alt.Chart(quarterly_cashflow).mark_bar(color='green').encode(
            x='Quarter:O',
            y=alt.Y(cfps_field + ':Q',
                   title='Cash Flow per Share',
                   scale=alt.Scale(zero=True))
        ).properties(
            title={
                'text': f'Cash Flow per Share - Trimestrale',
                'align': 'center',
                'anchor': 'middle'
            }
        )
    
    # Trova il campo debt disponibile
    debt_field = get_balance_sheet_field(quarterly_balance_sheet, debt_field_options)
    
    if debt_field:
        # Rimuovi eventuali valori NaN
        quarterly_balance_sheet_clean = quarterly_balance_sheet.dropna(subset=[debt_field])
        
        equity_chart = alt.Chart(quarterly_balance_sheet_clean).mark_bar(color='green').encode(
            x='Quarter:O',
            y=alt.Y(debt_field + ':Q',
                   title='Debt',
                   scale=alt.Scale(zero=True))
        ).properties(
            title={
                'text': f'Debito - Trimestrale',
                'align': 'center',
                'anchor': 'middle'
            }
        )
        
        st.altair_chart(revenue_chart, use_container_width=True)
        st.altair_chart(eps_chart, use_container_width=True)
        st.altair_chart(shares_chart, use_container_width=True)
        st.altair_chart(equity_chart, use_container_width=True)
        
        # Aggiungi i nuovi grafici
        if bvps_field:
            st.altair_chart(bvps_chart, use_container_width=True)
        else:
            st.warning("Book Value per Share non disponibile")
            
        if cfps_field:
            st.altair_chart(cfps_chart, use_container_width=True)
        else:
            st.warning("Cash Flow per Share non disponibile")
    else:
        st.altair_chart(revenue_chart, use_container_width=True)
        st.altair_chart(eps_chart, use_container_width=True)
        st.altair_chart(shares_chart, use_container_width=True)
        st.warning("Dati del Debt non disponibili. Campi disponibili nel balance sheet:")
        st.write(quarterly_balance_sheet.columns.tolist())
        
        # Grafico alternativo con Total Equity se disponibile
        equity_field = get_balance_sheet_field(quarterly_balance_sheet, 
                                             ['Total Equity Gross Minority Interest', 
                                              'Stockholders Equity', 
                                              'Total Stockholder Equity'])
        
        if equity_field:
            equity_chart = alt.Chart(quarterly_balance_sheet).mark_bar().encode(
                x='Quarter:O',
                y=alt.Y(equity_field + ':Q',
                       title='Equity',
                       scale=alt.Scale(zero=True))
            ).properties(
                title=f'Equity Levels - Trimestrale'
            )
            st.altair_chart(equity_chart, use_container_width=True)

if selection == 'Annuale':
    annual_financials = annual_financials.rename_axis('Anno').reset_index()
    annual_financials['Anno'] = annual_financials['Anno'].astype(str).apply(lambda year: year.split('-')[0])
    
    revenue_chart = alt.Chart(annual_financials).mark_bar().encode(
        x='Anno:O',
        y='Total Revenue'
    ).properties(
        title={
            'text': f'Ricavi Totali - Annuale',
            'align': 'center',
            'anchor': 'middle'
        }
    )
    
    eps_chart = alt.Chart(annual_financials).mark_bar().encode(
        x='Anno:O',
        y='Diluted EPS'
    ).properties(
        title={
            'text': f'Earnings per Share - Annuale',
            'align': 'center',
            'anchor': 'middle'
        }
    )
    
    shares_chart = alt.Chart(annual_financials).mark_bar().encode(
        x='Anno:O',
        y='Diluted Average Shares'
    ).properties(
        title={
            'text': f'Numero di Azioni in Circolazione',
            'align': 'center',
            'anchor': 'middle'
        }
    )

    # Correzione per il grafico del balance sheet annuale
    annual_balance_sheet = annual_balance_sheet.rename_axis('Anno').reset_index()
    annual_balance_sheet['Anno'] = annual_balance_sheet['Anno'].astype(str).apply(lambda year: year.split('-')[0])
    
    # Calcola Book Value per Share
    bvps_field, equity_field = calculate_book_value_per_share(annual_balance_sheet, information)
    
    # Grafico Book Value per Share
    if bvps_field:
        bvps_chart = alt.Chart(annual_balance_sheet).mark_bar().encode(
            x='Anno:O',
            y=alt.Y(bvps_field + ':Q',
                   title='Book Value per Share',
                   scale=alt.Scale(zero=True))
        ).properties(
            title={
                'text': f'Book Value per Share - Annuale',
                'align': 'center',
                'anchor': 'middle'
            }
        )
    
    # Cash Flow per Share
    annual_cashflow = annual_cashflow.rename_axis('Anno').reset_index()
    annual_cashflow['Anno'] = annual_cashflow['Anno'].astype(str).apply(lambda year: year.split('-')[0])
    
    cfps_field, ocf_field = calculate_cash_flow_per_share(annual_cashflow, annual_financials)
    
    # Grafico Cash Flow per Share
    if cfps_field:
        cfps_chart = alt.Chart(annual_cashflow).mark_bar().encode(
            x='Anno:O',
            y=alt.Y(cfps_field + ':Q',
                   title='Cash Flow per Share',
                   scale=alt.Scale(zero=True))
        ).properties(
            title={
                'text': f'Cash Flow per Share - Annuale',
                'align': 'center',
                'anchor': 'middle'
            }
        )
    
    # Trova il campo debt disponibile
    debt_field = get_balance_sheet_field(annual_balance_sheet, debt_field_options)
    
    if debt_field:
        # Rimuovi eventuali valori NaN
        annual_balance_sheet_clean = annual_balance_sheet.dropna(subset=[debt_field])
        
        equity_chart = alt.Chart(annual_balance_sheet_clean).mark_bar().encode(
            x='Anno:O',
            y=alt.Y(debt_field + ':Q',
                   title='Debt',
                   scale=alt.Scale(zero=True))
        ).properties(
            title={
                'text': f'Debito - Annuale',
                'align': 'center',
                'anchor': 'middle'
            }
        )
        
        st.altair_chart(revenue_chart, use_container_width=True)
        st.altair_chart(eps_chart, use_container_width=True)
        st.altair_chart(shares_chart, use_container_width=True)
        st.altair_chart(equity_chart, use_container_width=True)
        
        # Aggiungi i nuovi grafici
        if bvps_field:
            st.altair_chart(bvps_chart, use_container_width=True)
        else:
            st.warning("Book Value per Share non disponibile")
            
        if cfps_field:
            st.altair_chart(cfps_chart, use_container_width=True)
        else:
            st.warning("Cash Flow per Share non disponibile")
    else:
        st.altair_chart(revenue_chart, use_container_width=True)
        st.altair_chart(eps_chart, use_container_width=True)
        st.altair_chart(shares_chart, use_container_width=True)
        st.warning("Dati del Debt non disponibili. Campi disponibili nel balance sheet:")
        st.write(annual_balance_sheet.columns.tolist())
        
        # Grafico alternativo con Total Equity se disponibile
        equity_field = get_balance_sheet_field(annual_balance_sheet, 
                                             ['Total Equity Gross Minority Interest', 
                                              'Stockholders Equity', 
                                              'Total Stockholder Equity'])
        
        if equity_field:
            equity_chart = alt.Chart(annual_balance_sheet).mark_bar().encode(
                x='Anno:O',
                y=alt.Y(equity_field + ':Q',
                       title='Equity',
                       scale=alt.Scale(zero=True))
            ).properties(
                title=f'Equity Levels - Annuale'
            )
            st.altair_chart(equity_chart, use_container_width=True)

# CALCOLO VALORE INTRINSECO -------------------------------------------------------
st.header('Calcolo Valore Intrinseco')

# Carica i dati finanziari annuali necessari per entrambi i metodi
annual_financials = fetch_annual_financials(symbol)
annual_balance_sheet = fetch_annual_balance_sheet(symbol)
annual_cashflow = fetch_annual_cashflow(symbol)
annual_earnings, annual_earnings_growth = fetch_earnings_growth(symbol)

# Crea pulsanti per selezionare il metodo di valutazione
valuation_method = st.radio(
    "Seleziona il Metodo di Valutazione",
    ["Discounted Cash Flow (DCF)", "Formula di Graham"],
    horizontal=True
)

# Mostra il metodo selezionato
if valuation_method == "Discounted Cash Flow (DCF)":
    st.subheader("Metodo DCF (Discounted Cash Flow)")
    intrinsic_value = calculate_dcf_value(symbol, information, annual_financials, annual_cashflow, annual_balance_sheet)
    
else:  # Formula di Graham
    st.subheader("Formula di Benjamin Graham")
    intrinsic_value = calculate_graham_value(symbol, information, annual_financials, annual_earnings_growth)

