import numpy as np
import pandas as pd
import yfinance as yfin
import matplotlib.pyplot as plt
import mplfinance as mpf
import ta
import pandas_datareader as pdr

# Importa a função de dividir os dados entre treino e teste
from sklearn.model_selection import train_test_split

# Importa função para normalizar os dados de entrada de treino e teste
from sklearn.preprocessing import StandardScaler

# Importa função do algoritmos de ML
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC

# Importa funções para medir desempenho do algoritmo de ML
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import RocCurveDisplay

import warnings    
warnings.simplefilter("ignore")



def graficoEstrategia(df, shortMM, longMM):
    plt.figure(figsize = (20, 10))
    plt.plot(df['Close'], alpha = 0.6, label = 'Preços')

    plt.scatter(df.index, df['Compra'], color = 'green', marker = '^', label = 'Compra')
    plt.scatter(df.index, df['Venda'], color = 'crimson', marker = 'v', label = 'Venda')

    plt.title(f'Sinais de Compra e Venda, EMA {shortMM} & EMA {longMM}')
    plt.xlabel('Dia')
    plt.ylabel('Preço USD/bbl')
    plt.legend(loc = 'upper left')

    return plt


def graficoPatrimonio(ativo):
    plt.figure(figsize = (20, 10))
    plt.plot(ativo['Equity'])
    plt.title('Patrimônio - Estratégia de Médias Móveis Exponencias')
    plt.xlabel('Dia')
    plt.ylabel('Valor')
    plt.show()

def calculaParametros(df, risk_free):
    retornos = np.log(df['Equity']).diff().mean()
    retornos = retornos * 252
    #print(retornos)
    volatility = np.log(df['Equity']).diff().std()
    volatility = volatility * np.sqrt(252)
    #print(volatility)
    sharpe_ratio = (retornos - risk_free)/volatility
    saida = {
        'retornos': retornos,
        'sharpe': sharpe_ratio,
        'volatilidade': volatility
        
    }
    return saida
    
def estrategiaMediaMovel(ativo, inicio, fim, intervalo, shortMM, longMM):
    def algoritmo(dfTemp):
        n = len(dfTemp)       
        sinais = [0]

        for i in range(1, n):
            short = dfTemp['EMA_short'][i]
            long = dfTemp['EMA_long'][i]

            previous_short = dfTemp['EMA_short'][i-1]
            previous_long = dfTemp['EMA_long'][i-1]

            if previous_short < previous_long and short >= long:
                sinais.append(1)

            elif previous_short > previous_long and short <= long:
                sinais.append(-1)

            else:
                sinais.append(0)

        dfTemp['Sinais'] = sinais
        dfTemp['Compra'] = np.where(dfTemp['Sinais'] == 1, dfTemp['Close'], np.nan)
        dfTemp['Venda'] = np.where(dfTemp['Sinais'] == -1, dfTemp['Close'], np.nan)
        
        return dfTemp
        


    df = yfin.download(ativo, inicio, fim, interval = intervalo)
    df = df.dropna()
    df['EMA_short'] = df['Close'].ewm(span =shortMM, adjust = False).mean()
    df['EMA_long'] = df['Close'].ewm(span = longMM, adjust = False).mean()
    
    df = algoritmo(df)
    #resultado = grafico(df, shortMM, longMM)
    
    return df#, grafico
    
def backtestEstrategia(ativo, equity, TP, SL):
    '''Backtesting: Retorna o DataFrame com a variação de patrimônio, retorno, volatilidade e sharpe ratio da estratégia'''   
    
    N = len(ativo)
    equity = [equity]

    # Take Profit (lucro/stop gain) TP
    # Stop Loss SL

    pos = 0

    for i in range(1, N):

        equity.append(equity[i-1])

        if pos == 1:
            if ativo['Close'][i] >= preco*(1 + TP):
                pos = 0
                equity[i] *= (1 + TP)

            elif ativo['Close'][i] <= preco*(1 - SL):
                pos = 0
                equity[i] *= (1 - SL)


        elif pos == -1:
            if ativo['Close'][i] <= preco*(1 - TP):
                pos = 0
                equity[i] *= (1 + TP)

            elif ativo['Close'][i] >= preco*(1 + SL):
                pos = 0
                equity[i] *= (1 - SL)


        else:
            if ativo['Sinais'][i] != 0:
                pos = ativo['Sinais'][i]
                preco = ativo['Close'][i]
        #print(equity)   

    ativo['Equity'] = equity
    
    

    return ativo


def indicadoresTecnicos(df, listaMM, indicadores):
    '''Gera as colunas com os dados dos indicadores técnicos baseados em um dataframe e uma lista de médias móveis'''
    # Definindo as médias móveis pelo preço do ativo para identificar se está acima ou abaixo
    for mm in listaMM:
        mm = int(mm)
        df[f'EMA_{mm}'] = ta.trend.ema_indicator(close = df['Close'], window = mm, fillna = True)/df['Close']
    
    # Índice de Força Relativa (RSI)
    if 'RSI' in indicadores:
        df['RSI'] = ta.momentum.rsi(close = df['Close'], fillna = True)
    
    # Alcance Médio Real (ATR)
    if 'ATR - Average True Range' in indicadores:
        df['ATR'] =  ta.volatility.average_true_range(high = df['High'], low = df['Low'], close = df['Close'], fillna = True)
    
    # Williams Percent Range
    if 'Williams Range' in indicadores:
        df['WR'] = ta.momentum.williams_r(high = df['High'], low = df['Low'], close = df['Close'], fillna = True)
    
    # Bollinger Bands
    if 'Bollinger Bands' in indicadores:
        indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)

        # Add Bollinger Bands features
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()
                # Add Bollinger Band high indicator
        df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

        # Add Bollinger Band low indicator
        df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

#     df = ta.add_all_ta_features(
#         df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    
    return df

def tratandoDF(df):
    '''Adiciona uma coluna target com a info de compra ou venda e ajusta o DF dropando algumas linhas nas quais os indicadores não foram calculados'''
    # Criando um vetor em target para definir com base nos preços de fechamento se o preço de encerramento foi maior ou menor que o anterior
    cl = np.array(df['Close'])
    target = np.where(cl[1:] > cl[0:-1], 1, -1)
    # Como não teremos o preço da última linha, podemos dropa-la
    df.drop(df.tail(1).index, inplace = True)
    df['Target'] = target
    
    # Eliminando linhas que não possuem alguns dados para indicadores técnicos
    df.drop(df.head(29).index, inplace = True)
    
    
    return df


def matrizEntradaSaida(df):
    '''Gera as matrizes de dados de entrada e saída com base no recorte dos indicadores e do campo alvo (venda ou compra)'''
    # Precisamos construir uma matriz que contenha os dados dos indicadores técnicos que geramos
    X = np.array(df.iloc[:, 6:-1]) 

    # Também construímos um vetor que contém as variáveis de saída
    Y = np.array(df['Target'])
    
    return X, Y

def organizaTesteTreino(matrizEntrada, matrizSaida, testSize):
    '''Organiza os dados de teste e treino, também normaliza os indicadores. Retorna um dicionário'''    
    X_train, X_test, Y_train, Y_test = train_test_split(matrizEntrada, matrizSaida, test_size = testSize, shuffle = False)
    ss = StandardScaler()
    ss = ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    dados = {
        'entradaTreino': X_train,
        'entradaTeste': X_test,
        'saidaTreino': Y_train,
        'saidaTeste': Y_test
    }
    
    return dados



def classificador(algoritmo, dados):
    '''Dado um algoritmoe e os dados, a função treina o algoritmo com base nos dados e gera as métricas'''
    algoritmo = algoritmo.fit(dados['entradaTreino'], dados['saidaTreino'])
    predicao = algoritmo.predict(dados['entradaTeste'])
    
    accScore = accuracy_score(dados['saidaTeste'], predicao)
    confMatrix = confusion_matrix(dados['saidaTeste'], predicao)
    
    saida = {
        'predicao': predicao,
        'metricas':{
            'acuracia': accScore,
            'confusao': confMatrix
        }        
    }
    return saida

def ajustaSinaisDF(df, dados, clf):
    '''Ajusta a coluna de sinais de compra ou venda do DF baseado no retorno do classificador de ML'''
    N = len(dados['saidaTeste'])
    df = df.tail(N)
    df['Sinais'] = clf['predicao']
    return df


def TradingComMachineLearning(infoAtivo, infoTecnica, infoBacktest):
    # try:
    try:
        df = yfin.download(infoAtivo['ativo'], infoAtivo['inicio'], infoAtivo['fim'], interval = infoAtivo['intervalo'])
        df = indicadoresTecnicos(df, infoTecnica['indicadores']['listaMedias'], infoTecnica['indicadores']['indicadores'])
        df = tratandoDF(df)
        X, Y = matrizEntradaSaida(df)
        dados = organizaTesteTreino(X, Y, infoTecnica['pesoTeste'])
        resClassif = classificador(infoTecnica['algoritmoClassificador'], dados)
        dfComSinais = ajustaSinaisDF(df, dados, resClassif)
        backtest = backtestEstrategia(dfComSinais, infoBacktest['equity'], infoBacktest['takeProfit'], infoBacktest['stopLoss'])
        saida = {
            'Estrategia': infoTecnica['estrategia'],
            'Informacoes': df,
            'Sinais': dfComSinais,
            'Backtest': backtest,
            'DesempenhoClassificador': resClassif,
            'Dados': dados,
            'Algoritmo': infoTecnica['algoritmoClassificador']
        }
        return saida
    
    except:
        return False