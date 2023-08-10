from typing import ValuesView
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
import streamlit_nested_layout
import random
import string
import os
from datetime import date
from datetime import timedelta
import backendML as be
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


#caminho = os.getcwd()
#st.markdown(f"<link rel='stylesheet' href='{caminho}/css/fontawesome-free-6.4.0-web/css/font-awesome.min.css'>", unsafe_allow_html = True)
st.set_page_config(layout = 'wide')


# Funcionalidades
memoria = st.session_state

if 'indices' not in memoria:
    memoria.indices = []

if 'simulacaoBotao' not in memoria:
    memoria.simulacaoBotao = False

# Interativo
def processaDados(simulacao):   
    def trataAlgoritmo(simulacao):
        infoAlgoritmos = {}
        infoTecnica = {}
        #['Árvore Binária', 'Random Forest', 'SVM - Support Vector Machine', 'Rede Neural - MLPClassifier']
        for algoritmo in simulacao['algoritmos']:
            # infoTecnica = {}
            if algoritmo == 'Árvore Binária':
                motorAlg = DecisionTreeClassifier()

            if algoritmo == 'Random Forest':
                motorAlg = RandomForestClassifier()

            if algoritmo == 'SVM - Support Vector Machine':
                motorAlg = SVC()

            if algoritmo == 'Rede Neural - MLPClassifier':
                motorAlg = MLPClassifier(hidden_layer_sizes = (64, 32), max_iter = 1000)

            infoTecnica[algoritmo] = {
                'estrategia': algoritmo,
                'algoritmoClassificador': motorAlg,
                'indicadores': trataIndicadores(simulacao),
                'pesoTeste': simulacao['pesoTreino']/100
            }
            #infoAlgoritmos[algoritmo] = infoTecnica

        return infoTecnica
    
    
    infoTecnicas = trataAlgoritmo(simulacao)
    
    retornoGeral = {}

    for acao in simulacao['tickers']:
        print(acao)
        print('-')
        retornoGeral[acao] = {}
        infoAtivo = {
            'ativo': acao.upper(),
            'inicio': simulacao['inicio'],
            'fim': simulacao['fim'],
            'intervalo': '1d'
        }        

        tp = (simulacao['takeProfit'] - simulacao['equity'])/100
        sl = (simulacao['equity'] - simulacao['stopLoss'])/100
        infoBacktest ={
            'equity': simulacao['equity'],
            'takeProfit': tp,
            'stopLoss': sl
        }

        retEstrategia = {}        
        for nomeEst, dadosEst in infoTecnicas.items():
            print(f"""
                'Info ativo': {infoAtivo},
                'Info técnica': {dadosEst},
                'Info backtest': {infoBacktest} 
            """)            
            retEstrategia[nomeEst] = be.TradingComMachineLearning(infoAtivo, dadosEst, infoBacktest)
        
        retornoGeral[acao] = retEstrategia
    
    #print(retornoGeral)
    return retornoGeral


        #print(f"\nAtivo = {infoAtivo}\nBacktest = {infoBacktest},\nAlgoritmo = {infoAlgoritmos}")



# Auxilio
def espacoVertical(n):
    for i in range(n):
        st.markdown('')

def trataIndicadores(simulacao):        
    for ind in memoria.indices:
        listaMedias = memoria[f"valor_{ind}"] if 'Médias Móveis' in memoria[f"tipo_{ind}"]  else []
        indicadores = memoria[f"tipo_{ind}"]
    
    
    
    if 'Médias Móveis' in memoria[f"tipo_{ind}"] and len(listaMedias) == 0:
        st.warning("**Adicione pelo menos uma média!!**")



    return {
        'indicadores': indicadores,
        'listaMedias': listaMedias
    }

def checaErros(simulacao):
    erros = 0
    if len(simulacao['tickers']) < 1:
        erros += 1
        st.error("Escolha pelo menos um ativo!!\n\nCheque se está 'vermelho' no campo")
    
    print(simulacao['fim'])
    print(simulacao['inicio'])
    print((simulacao['fim'] - simulacao['inicio']))
    if simulacao['fim'] >= date.today():
        erros += 1
        st.error(f"Escolha no máximo até o último dia útil")
    
    if simulacao['inicio'] >= date.today():
        erros += 1
        st.error(f"Escolha no máximo até o último dia útil")

    if simulacao['inicio'] >= simulacao['fim']:
        erros += 1
        st.error(f"Data inválida")

    if (simulacao['fim'] - simulacao['inicio']).days < 30:
        erros += 1
        st.error(f"Escolha um intervalo de pelo menos um mês")
    
    try:
        indicadores = trataIndicadores(simulacao)
        if len(indicadores['indicadores']) < 1:
            erros += 1
            st.error(f"Escolha pelo menos um indicador")
        
        if 'Médias Móveis' in indicadores['indicadores'] and len(indicadores['listaMedias']) < 1:
            erros += 1
            st.error(f"Adicione pelo menos uma média")
    except:
        erros += 1
        st.error(f"Checar indicadores")
    
    return erros



# Layout
def ativosGrid():
    st.markdown('<h4>Ativos</h4>', unsafe_allow_html = True)
    simulacao['tickers'] = st_tags(
        label='_Tickers_ dos Ativos :moneybag:',
        text='Aperte enter para adicionar mais',    
        suggestions=['ITUB4.SA', 'VALE3.SA', 'PETR4.SA', 'KNRI11.SA', 'AAPL', 'GOOG'],    
        key='tickers')

    #simulacao['tickers'] = [tick.upper() for tick in simulacao['tickers']]

    colunas = st.columns(2)
    dataMax = date.today()
    dataMax = dataMax - timedelta(days=1)
    with colunas[0]:
        simulacao['inicio'] = st.date_input('Ínicio :calendar:', key = 'inicio')

    with colunas[1]:
        simulacao['fim'] = st.date_input('Fim :calendar:', key = 'fim')


def algoritmoGrid():
    st.markdown('<h4>Aprendizado</h4>', unsafe_allow_html = True)
    simulacao['algoritmos'] = st.multiselect('Algoritmos de Teste :robot_face:', ['Árvore Binária', 'Random Forest', 'SVM - Support Vector Machine', 'Rede Neural - MLPClassifier'], key = 'algoritmos')
    colunas = st.columns(2)
    with colunas[0]:
        simulacao['pesoTreino'] = st.slider('Porcentagem dos dados para treino', 1, 100, 10, key = 'pesoTreino')


def indicadoresGrid():
    st.markdown('<h4>Indicadores Técnicos</h4>', unsafe_allow_html = True)    
    indicadores = {}
    indSelect = ['Médias Móveis', 'RSI', 'Bollinger Bands', 'ATR - Averate True Range', 'Williams Range']
    
    for chave in memoria.indices: 
        indicadores[chave] = {}
        tipo = st.multiselect("Indicador :bar_chart:", indSelect, key = f"tipo_{chave}")        
        if 'Médias Móveis' in tipo:
            valor = st_tags(
                        label='Médias :part_alternation_mark:',
                        text='Aperte enter para adicionar mais',                    
                        key=f'valor_{chave}')
           
            indicadores[chave]['listaMedias'] = valor
    
        excluir = st.button("Excluir :heavy_minus_sign:", key = f"excluir_{chave}" )#, on_click = excluirChave(chave))
        if excluir:
            #print(f"vou excluir essa {chave}")        
            tipoRem, valorRem, botaoRem = memoria.pop(f"tipo_{chave}", None), memoria.pop(f"valor_{chave}", None), memoria.pop(f"excluir_{chave}", None)
            memoria.indices.remove(chave)
            st.experimental_rerun() # Força atualização da pagína para sumir com o item                

        indicadores[chave]['Tipo'] = tipo
    
    simulacao['indicadores'] = indicadores

    colAlgo = st.columns(3)
    with colAlgo[0]:
        if len(indicadores) == 0:
            if st.button("Adicionar indicador :heavy_plus_sign:"):
                chave = f"{random.choice(string.ascii_uppercase)}_{(random.randint(0,999999))}"    
                memoria.indices.append(chave)
                st.experimental_rerun()


def backtestGrid():
    st.markdown('<h4>Backtest</h4>', unsafe_allow_html = True)
    simulacao['equity'] = st.number_input("Patrimônio :briefcase:", min_value = 1.00, value = 100.00, key = 'equity')
    colunas = st.columns(2)
    # colunas = st.columns(2)
    # with colunas[0]:
    #     simulacao['pesoTreino'] = st.slider('Porcentagem dos dados para treino', 1, 100, 10, key = 'pesoTreino')
    with colunas[0]:
        simulacao['takeProfit'] = st.slider("_Take Profit_ :chart_with_upwards_trend:", 
        (1.01 * simulacao['equity']), (2 * simulacao['equity']), (1.03 * simulacao['equity']),
        key = 'takeProfit')
        st.success(f"Liquidar quando o patrimônio aumentar para $ {str(simulacao['takeProfit']).replace('.', ',')} ({round(simulacao['takeProfit']/simulacao['equity'], 2)}%)")

    with colunas[1]:
        simulacao['stopLoss'] = st.slider("_Stop Loss_ :chart_with_downwards_trend:",
        round((0.01 * simulacao['equity']), 2), round((0.99 * simulacao['equity']), 2), round((0.99 * simulacao['equity']), 2),  
        key = 'stopLoss')
        st.warning(f"Liquidar quando restar $ {str(simulacao['stopLoss']).replace('.', ',')} do patrimônio ({round(simulacao['stopLoss']/simulacao['equity'], 2)}%)")

# Resultados Grid
def mostraResultadosGrid(resultado):
    if len(resultado) > 0:
        for acao, dados in resultado.items():
            with st.expander(f"{acao} :briefcase:"):
                equityEstrategia = pd.DataFrame()                
                for estrategia, resultadoEstrategia in dados.items():                    
                    equity = resultadoEstrategia['Backtest']['Equity'].iloc[0]
                    equityEstrategia['Base'] = resultadoEstrategia['Backtest']['Close']                    
                    equityEstrategia['Base'] = (equity/equityEstrategia['Base'][0])*equityEstrategia['Base']                    
                    #st.write(equityEstrategia)
                    with st.expander(f"{estrategia} :chart_with_upwards_trend:"):                       
                        equityEstrategia[estrategia] = resultadoEstrategia['Backtest']['Equity']
                        st.line_chart(equityEstrategia[['Base', estrategia]])
                    
                st.subheader("Comparação estratégias")
                st.line_chart(equityEstrategia)
    else:
        st.warning("Realize a simulação!")






st.markdown('<h3>Trading com <i>Machine Learning</i></h3>', unsafe_allow_html = True)

guias = st.tabs(['Setup :computer:', 'Resultados :bow_and_arrow:'])
resultadoEstrategia = {}

with guias[0]:
    simulacao = {}
    with st.expander('**Etapa 1 - Tickers e Períodos...**'):
        ativosGrid()
        
    with st.expander('**Etapa 2 - O Algoritmo...**'):
        algoritmoGrid()              
        indicadoresGrid()        

    with st.expander(f"**Etapa 3 - _Backtest_**"):
        backtestGrid()
    
    if st.button("**Simular Estratégia** :rocket:"):
        if checaErros(simulacao) < 1:            
            resultadoEstrategia = processaDados(simulacao)
            if resultadoEstrategia == False:
                st.error("Não foi possível realizar a simulação\n\nCheque as informações dos ativos")

with guias[1]:
    if resultadoEstrategia != False:
        mostraResultadosGrid(resultadoEstrategia)

print(f"\n\n\nATUALIZEI A PAGINA")
#print(memoria)
#print(f"indicadores = {indicadores}")
        