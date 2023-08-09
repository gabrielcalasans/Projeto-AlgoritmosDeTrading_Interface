# Sobre
É um projeto de criação de interface via streamlit para os códigos desenvolvidos no curso de trading baseado em machine learning.

É possível acessá-lo via https://projeto-algoritmosdetradinginterface.streamlit.app/.

A ideia é basicamente selecionar alguns ativos, estratégias e indicadores e a partir daí gerar um backtest de estratégias de negociação baseada em motores de machine learning (Random Forest, SVM, Árvore Binária, Rede Neural MLPC). Dessa forma, é possível compara-las.

Para roda-lo é necessário usar via cmd o comando "streamlit run TradingComML.py", há um arquivo .bat para que a execução fique mais fácil.

É necessário também utilizar algumas bibliotecas:

- pip install yfinance==0.2.24
- pip install pandas==2.0.3
- pip install streamlit==1.24.1
- pip install streamlit-nested-layout==0.1.1
- pip install streamlit-tags==1.2.8
- pip install ta==0.10.2
- pip install mplfinance==0.12.9b7
- pip install sklearn==0.0.post5
- pip install scikit-learn==1.3.0
- pip install pandas-datareader==0.10.0

Um arquivo .bat  também está incluso nos arquivos do projeto.

## Ainda em construção...

Há planos para adição de mais algumas seções, como uma para adicionar um benchmark.
