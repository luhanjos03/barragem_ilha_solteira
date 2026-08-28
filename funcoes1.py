"""
Funções 1
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose, STL


def memory_by_df(data_dict: dict[str, pd.DataFrame]):
    """
    Calcula e exibe o uso de memória de um dicionário de DataFrames Pandas.

    A função itera sobre um dicionário contendo DataFrames, calculando o 
    uso de memória profundo (incluindo o tamanho real de objetos/strings) 
    de cada um. Ao final, imprime no console o consumo individual de cada 
    aba em megabytes (MB) e o consumo total.

    Parameters
    ----------
    data_dict : dict[str, pandas.DataFrame]
        Dicionário onde as chaves representam os nomes ou identificadores 
        dos dados (ex: nomes das abas de uma planilha) e os valores são os 
        DataFrames correspondentes.

    Returns
    -------
    None
        A função não retorna nenhum valor, apenas imprime os resultados 
        na saída padrão (stdout).

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'A': range(10000)})
    >>> df2 = pd.DataFrame({'B': ['texto'] * 10000})
    >>> data = {'Aba 1': df1, 'Aba 2': df2}
    >>> memory_by_df(data)
    Uso de memória por aba:
    ==============================
     - Aba 1: 0.08 MB
     - Aba 2: 0.60 MB
    
    ==============================
    Tamanho Total dos dados: 0.68 MB
    """
    all_bytes = 0

    print("Uso de memória por aba:")
    print("=" * 30)

    for key, value in data_dict.items():
        value_shape_bytes = value.memory_usage(deep=True).sum()
        all_bytes += value_shape_bytes

        print(f" - {key}: {value_shape_bytes / (1024*1024):.2f} MB")

    print()
    print("=" * 30)
    print(f"Tamanho Total dos dados: {all_bytes / (1024*1024):.2f} MB")

def compass_rose_map(data: pd.DataFrame, direction_column: str) -> pd.DataFrame:
    """
    Mapeia ângulos de direção em graus para os pontos da rosa dos ventos.

    A função recebe um DataFrame e o nome de uma coluna contendo dados 
    numéricos que representam direções em graus (geralmente de 0 a 360). 
    Os graus são divididos em 8 setores de 45°, mapeando-os para as 
    strings correspondentes da rosa dos ventos ("N -> S", "NE -> SW", "E -> W", 
    "SE -> NW", "S -> N", "SW -> NE", "W -> E", "NW -> SE"). 

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contendo a base de dados original.
    direction_column : str
        Nome da coluna no DataFrame que contém os ângulos numéricos 
        (em graus) que serão mapeados.

    Returns
    -------
    pandas.DataFrame
        Uma cópia do DataFrame original acrescida da nova coluna 
        'Dir-vento-map' contendo as categorias da rosa dos ventos.

    Notes
    -----
    A função realiza um arredondamento após a divisão por 45. Assim, 
    ângulos entre 337.5 e 22.5 caem no índice 0 ("N"), ângulos entre 
    22.5 e 67.5 caem no índice 1 ("NE"), e assim por diante. O uso do 
    módulo 8 (`% 8`) garante que valores próximos a 360 voltem a ser "N".

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'vento_graus': [0, 42, 95, 350]})
    >>> compass_rose_map(df, 'vento_graus')
       vento_graus Dir-vento-map
    0            0        N -> S
    1           42      NE -> SW
    2           95        E -> W
    3          350        N -> S
    """
    
    data_copy = data.copy()
    data_copy["Dir-vento-map"] = np.nan

    # Lista com as direções em ordem
    direcoes = np.array(["N -> S", "NE -> SW", "E -> W", "SE -> NW", "S -> N", "SW -> NE", "W -> E", "NW -> SE"])

    # Cria uma máscara boolean apenas para as linhas onde direction_column NÃO é nulo
    mascara_validos = data_copy[direction_column].notna()

    # Calcula os índices apenas para as linhas válidas
    indices = np.round(data_copy.loc[mascara_validos, direction_column] / 45).astype(int) 
    indices = indices % 8

    # 4. Mapeia as direções e substitui no DataFrame
    data_copy.loc[mascara_validos, "Dir-vento-map"] = direcoes[indices]

    return data_copy

def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """
    Calcula o Erro Percentual Absoluto Médio (MAPE).

    Esta função avalia a precisão de um modelo de previsão comparando os 
    valores reais com os previstos. O resultado é retornado em formato de 
    porcentagem. Casos onde o valor real é igual a zero são automaticamente 
    filtrados para evitar erros matemáticos (divisão por zero).

    A fórmula matemática utilizada é:
    $$
    \begin{aligned}
    \text{MAPE} = 100 \times \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_{true,i} - y_{pred,i}}{y_{true,i}} \right|
    \end{aligned}
    $$

    Parameters
    ----------
    y_true : array_like
        Array, lista ou série do Pandas contendo os valores reais observados.
    y_pred : array_like
        Array, lista ou série do Pandas contendo os valores previstos pelo modelo.
        Deve ter o mesmo tamanho que `y_true`.

    Returns
    -------
    float
        O valor do MAPE calculado em porcentagem (por exemplo, um retorno 
        de 5.5 significa um erro médio de 5.5%).

    Notes
    -----
    A função converte internamente as entradas para `numpy.ndarray`. 
    Uma máscara booleana (`non_zero`) é criada para isolar e calcular 
    o erro apenas nos índices onde `y_true != 0`. Isso garante 
    estabilidade numérica em bases de dados que contêm zeros reais.

    Examples
    --------
    >>> import numpy as np
    >>> y_real = [3.0, -0.5, 2.0, 0.0, 7.0]
    >>> y_previsto = [2.5, 0.0, 2.0, 1.0, 8.0]
    >>> mean_absolute_percentage_error(y_real, y_previsto)
    32.73809523809524
    """
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0

    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def criar_features_regressao(df_slice: pd.DataFrame, grau: int= 5) -> pd.DataFrame:
    """
    Cria matriz de covariáveis com polinômio de tendência e dummies de hora.

    Constrói um DataFrame (matriz X) contendo uma base polinomial de tempo 
    de grau `grau` para capturar a tendência da série temporal, concatenada 
    com variáveis indicadoras (dummies) de hora para capturar a sazonalidade. 
    Este método de ajuste de curvas via regressão é baseado na metodologia 
    proposta por Larrubia (2021) [1] para análise de séries temporais.

    A matriz gerada segue a estrutura:
    $$
    \begin{aligned}
    X &= \left[ t^1, t^2, \dots, t^n, D_2, \dots, D_H \right]
    \end{aligned}
    $$
    Onde $t$ é o vetor de tempo, $n$ é o grau do polinômio, e $D$ são as 
    variáveis dummies de hora.

    Parameters
    ----------
    df_slice : pandas.DataFrame
        O recorte de dados original. Deve conter obrigatoriamente um índice 
        temporal (ou sequencial) válido e uma coluna chamada `"hora"` 
        representando a hora correspondente a cada observação.
    grau : int, optional
        O grau máximo do polinômio de tendência a ser gerado (representa 
        o $n$ da fórmula matemática). O valor padrão é 5.

    Returns
    -------
    pandas.DataFrame
        Matriz de características (features) contendo as colunas de tendência 
        (nomeadas como 't_1', 't_2', ..., 't_n') e as variáveis dummies de 
        hora (nomeadas como 'hora_X').

    Notes
    -----
    A função utiliza o parâmetro `drop_first=True` na criação das variáveis 
    dummies por meio do `pd.get_dummies()`. Isso remove a primeira categoria 
    da coluna de horas para evitar o problema de multicolinearidade perfeita 
    (conhecido como "armadilha das dummies"), garantindo que a matriz tenha 
    posto completo para a estimação de modelos de regressão.

    References
    ----------
    .. [1] Larrubia, L. F. (2021). Detecção de anomalias, interpolação e 
       previsão em tempo real de séries temporais para operação de 
       reservatórios e distribuição de água (Dissertação de Mestrado). 
       Instituto de Matemática e Estatística, Universidade de São Paulo, 
       São Paulo. DOI: 10.11606/D.45.2021.tde-10062021-231004

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'hora': [8, 9, 10, 8]}, index=[0, 1, 2, 3])
    >>> criar_features_regressao(df, grau=2)
       t_1  t_2  hora_9  hora_10
    0    0    0   False    False
    1    1    1    True    False
    2    2    4   False     True
    3    3    9   False    False
    """
    
    X = pd.DataFrame(index= df_slice.index)
    t = np.arange(len(df_slice))
    
    # Polinômio de grau n (tendência)
    for i in range(1, grau + 1):
        X[f"t_{i}"] = t ** i
    
    # Dummies de hora (sazonalidade)
    X = pd.concat([X, pd.get_dummies(df_slice["hora"], prefix= "hora", drop_first= True)], axis=1)
    return X

def metodo_1_regressao_residuos(df_window: pd.DataFrame, col: str, grau_pol: int = 5) -> pd.Series:
    """
    Imputa valores ausentes combinando regressão linear (tendência/sazonalidade) 
    com interpolação de resíduos.

    Este é o "Método 1" descrito por Larrubia (2021) [1] para tratamento de 
    falhas em séries temporais. O método ajusta um modelo de regressão linear 
    (Mínimos Quadrados Ordinários) exclusivamente sobre as observações válidas da 
    janela. Em seguida, os resíduos (erros) da regressão são calculados e 
    interpolados linearmente nos períodos faltantes. A estimativa final é a 
    soma da previsão da regressão com o resíduo interpolado.

    A formulação matemática do preenchimento é:
    $$
    \begin{aligned}
    \hat{y}_t &= X_t \hat{\beta} \quad \text{(Previsão da regressão para todo } t \text{)} \\
    r_t &= y_t - \hat{y}_t \quad \text{(Resíduo calculado apenas onde } y_t \text{ é válido)} \\
    \tilde{r}_t &= \text{Interpolação Linear}(r_t) \quad \text{(Estimativa do resíduo nas falhas)} \\
    y^*_t &= \hat{y}_t + \tilde{r}_t \quad \text{(Estimativa final para os valores nulos)}
    \end{aligned}
    $$

    Parameters
    ----------
    df_window : pandas.DataFrame
        O recorte (janela) de dados original. Deve conter a coluna alvo com 
        os valores a serem imputados e a coluna `"hora"` (exigida pela 
        função auxiliar de features).
    col : str
        Nome da coluna alvo no `df_window` que contém os dados com valores 
        ausentes (`NaN`) a serem preenchidos.
    grau_pol : int, optional
        O grau máximo do polinômio de tendência a ser gerado para a matriz 
        de covariáveis. O valor padrão é 5.

    Returns
    -------
    pandas.Series
        Série correspondente à coluna `col` original, mas com os valores nulos 
        preenchidos pelas estimativas do método. Valores originais não-nulos 
        são preservados intactos.

    See Also
    --------
    criar_features_regressao : Função auxiliar que constrói a matriz $X$ 
        contendo o polinômio temporal e as variáveis dummies de hora.

    Notes
    -----
    - Caso a janela possua menos de 10 observações válidas (não-nulas), 
      o ajuste do modelo é considerado instável e a função retorna a série 
      original sem alterações.
    - Se não houver valores nulos na janela (tamanho de `idx_null` == 0), 
      a função realiza um retorno antecipado (*early exit*) por eficiência.
    - A função realiza uma cópia interna de `df_window` para evitar efeitos 
      colaterais no DataFrame original durante o cálculo dos resíduos.
    - O método requer que `sklearn.linear_model.LinearRegression` seja 
      importado ou esteja disponível no escopo do código.

    References
    ----------
    .. [1] Larrubia, L. F. (2021). Detecção de anomalias, interpolação e 
       previsão em tempo real de séries temporais para operação de 
       reservatórios e distribuição de água (Dissertação de Mestrado). 
       Instituto de Matemática e Estatística, Universidade de São Paulo.
       DOI: 10.11606/D.45.2021.tde-10062021-231004

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Criando uma base fictícia de exemplo
    >>> df = pd.DataFrame({
    ...     'hora': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ...     'valor': [10.5, 12.0, np.nan, np.nan, 15.0, 16.2, 17.5, 18.0, 19.5, 20.0, 21.0, 22.5]
    ... }, index=range(12))
    >>> serie_imputada = metodo_1_regressao_residuos(df, col='valor', grau_pol=2)
    >>> print(serie_imputada.isnull().sum())
    0
    """
    
    X = criar_features_regressao(df_window, grau= grau_pol)
    y = df_window[col]
    df_window = df_window.copy()
    
    # Separar dados válidos e nulos
    idx_valid = y.dropna().index
    idx_null = y[y.isnull()].index
    
    # Se houver menos de 10 dados válidos, não há graus de liberdade suficientes
    # para ajustar um modelo robusto. Retorna sem alterar.
    if len(idx_valid) < 10 or len(idx_null) == 0:
        return y
    
    # Ajustar regressão apenas nos válidos
    modelo = LinearRegression()
    modelo.fit(X.loc[idx_valid], y.loc[idx_valid])
    
    # Prever na janela toda
    y_pred_reg = modelo.predict(X)
    df_window["pred_reg"] = y_pred_reg
    
    # Calcular resíduo nos dados válidos
    df_window["residuo"] = y - df_window["pred_reg"]
    
    # Interpolar o resíduo linearmente ao longo do buraco
    df_window["residuo_interpolado"] = df_window["residuo"].interpolate(method= "linear")
    
    # Estimativa final: Regressão + Resíduo
    estimativa = df_window["pred_reg"] + df_window["residuo_interpolado"]
    
    # Preencher apenas onde estava nulo originalmente
    y_filled = y.fillna(estimativa)
    return y_filled

def metodo_2_decomposicao_classica(df_window: pd.DataFrame, col: str, period: int = 1440, grau_pol: int = 5) -> pd.Series:
    """
    Imputa valores ausentes aplicando Decomposição Clássica sobre um preenchimento 
    prévio baseado em regressão e interpolação.

    Este é o "Método 2" descrito por Larrubia (2021) [1]. Como a decomposição clássica 
    não lida diretamente com valores nulos, o método utiliza primeiramente o "Método 1" 
    como um "rascunho" para gerar uma série contínua. Em seguida, aplica uma decomposição 
    sazonal aditiva nessa série completa. A estimativa final para as falhas é a soma dos 
    componentes de tendência e sazonalidade, descartando o ruído (resíduo da decomposição), 
    o que resulta em uma curva mais suave e representativa do comportamento esperado.

    A formulação matemática da técnica é:
    $$
    \begin{aligned}
    y^{(1)}_t &= \text{Método 1}(X_t, y_t) \quad \text{(Série contínua gerada como rascunho)} \\
    y^{(1)}_t &= T_t + S_t + R_t \quad \text{(Decomposição aditiva clássica)} \\
    \hat{y}_t &= T_t + S_t \quad \text{(Estimativa baseada na tendência e sazonalidade)} \\
    y^*_t &= \begin{cases} 
      y_t, & \text{se } y_t \text{ é válido} \\ 
      \hat{y}_t, & \text{se } y_t \text{ é nulo} 
    \end{cases}
    \end{aligned}
    $$
    Onde $T_t$ é a tendência, $S_t$ é a sazonalidade e $R_t$ é o resíduo (ruído) isolado.

    Parameters
    ----------
    df_window : pandas.DataFrame
        O recorte (janela) de dados original. Deve conter a coluna alvo e as 
        variáveis necessárias para a execução do `metodo_1_regressao_residuos`.
    col : str
        Nome da coluna alvo no `df_window` que contém os valores ausentes (`NaN`).
    period : int, optional
        O período de sazonalidade da série, utilizado pela função de decomposição. 
        O valor padrão é 1440, tipicamente usado para dados com frequência de 
        1 minuto (1 dia = 1440 minutos).
    grau_pol : int, optional
        O grau máximo do polinômio de tendência a ser passado para a etapa de 
        regressão (Método 1). O valor padrão é 5.

    Returns
    -------
    pandas.Series
        Série correspondente à coluna `col` original com os valores nulos preenchidos 
        pela estimativa suavizada ($T_t + S_t$).

    See Also
    --------
    metodo_1_regressao_residuos : Função utilizada na primeira etapa para gerar o "rascunho".
    statsmodels.tsa.seasonal.seasonal_decompose : Função do pacote Statsmodels responsável 
        pela decomposição sazonal clássica.

    Notes
    -----
    A função `seasonal_decompose` necessita de, no mínimo, dados cobrindo duas vezes 
    o período da sazonalidade (`2 * period`) para calcular as médias móveis corretamente. 
    Caso a janela recebida seja menor que este limite, a função aborta a decomposição 
    e retorna o resultado do "Método 1" (`y_filled_reg`) como um mecanismo de *fallback* 
    (segurança). Além disso, como `seasonal_decompose` não aceita NaN, são aplicados 
    `.bfill().ffill()` na saída do Método 1 antes da decomposição, garantindo que 
    eventuais nulos nas bordas da janela (onde a interpolação linear não alcança) 
    sejam preenchidos.

    References
    ----------
    .. [1] Larrubia, L. F. (2021). Detecção de anomalias, interpolação e 
       previsão em tempo real de séries temporais para operação de 
       reservatórios e distribuição de água (Dissertação de Mestrado). 
       Instituto de Matemática e Estatística, Universidade de São Paulo.
       DOI: 10.11606/D.45.2021.tde-10062021-231004
    """
    
    # O "rascunho" com a regressão
    y_filled_reg = metodo_1_regressao_residuos(df_window, col, grau_pol= grau_pol)
    
    # Se a janela for menor que 2 períodos, o clássico falha, então retornamos a regressão
    if len(y_filled_reg) < 2 * period:
        return y_filled_reg
    
    # seasonal_decompose não aceita NaN. Nulos nas bordas podem sobrar da interpolação
    # do Método 1, então garantimos uma série completamente preenchida antes de decompor.
    y_filled_reg = y_filled_reg.bfill().ffill()
    
    # Decomposição clássica
    decomp = seasonal_decompose(y_filled_reg, model="additive", period=period, extrapolate_trend="freq")
    
    # Estimativa = Tendência + Sazonalidade (suaviza o ruído/interpolação)
    estimativa = decomp.trend + decomp.seasonal
    return df_window[col].fillna(estimativa)

def metodo_3_stl(df_window: pd.DataFrame, col: str, period: int = 1440) -> pd.Series:
    """
    Imputa valores ausentes aplicando a decomposição STL (Seasonal and Trend 
    decomposition using Loess).

    Este é o "Método 3" descrito por Larrubia (2021) [1]. Como o algoritmo STL 
    (do pacote statsmodels) requer uma série temporal completa, sem falhas, o 
    método realiza um preenchimento inicial simples (interpolação linear) para 
    servir de "chute" (estimativa prévia). Sobre essa série contínua, aplica-se 
    a decomposição STL em modo robusto, e a estimativa final para as falhas 
    passa a ser a soma dos componentes extraídos de tendência e sazonalidade.

    A formulação matemática do processo é:
    $$
    \begin{aligned}
    y^{(guess)}_t &= \text{Interpolação Linear}(y_t) \quad \text{(Preenchimento prévio)} \\
    y^{(guess)}_t &= T_t + S_t + R_t \quad \text{(Decomposição STL Robusta)} \\
    \hat{y}_t &= T_t + S_t \quad \text{(Estimativa suavizada)} \\
    y^*_t &= \begin{cases} 
      y_t, & \text{se } y_t \text{ é válido} \\ 
      \hat{y}_t, & \text{se } y_t \text{ é nulo} 
    \end{cases}
    \end{aligned}
    $$
    Onde $T_t$ é a tendência obtida via Loess, $S_t$ é a sazonalidade e $R_t$ 
    é o resíduo contendo o ruído e possíveis anomalias (*outliers*).

    Parameters
    ----------
    df_window : pandas.DataFrame
        O recorte (janela) de dados original contendo a série temporal.
    col : str
        Nome da coluna alvo no `df_window` que contém os dados com valores 
        ausentes (`NaN`).
    period : int, optional
        O período de sazonalidade da série (número de observações em um 
        ciclo completo). O valor padrão é 1440, apropriado para séries 
        minutais diárias (1 dia = 1440 minutos).

    Returns
    -------
    pandas.Series
        Série correspondente à coluna `col` original, com os valores nulos 
        preenchidos pela soma da tendência e sazonalidade extraídas pelo STL.

    See Also
    --------
    statsmodels.tsa.seasonal.STL : Classe do pacote Statsmodels que implementa 
        o algoritmo de decomposição baseado em Loess.

    Notes
    -----
    - **Tratamento de Bordas:** A interpolação linear inicial é acompanhada de 
      `.bfill()` (backward fill) e `.ffill()` (forward fill) para garantir que valores 
      nulos nas extremidades da janela, onde a interpolação linear falha, 
      sejam preenchidos adequadamente.
    - **Fallback de Tamanho:** O método requer pelo menos `2 * period` 
      observações para calcular adequadamente a sazonalidade. Caso a janela seja 
      menor que isso, a função retorna a estimativa inicial (`y_guess`).
    - **Fallback de Estabilidade:** O ajuste do STL (`stl.fit()`) está contido 
      em um bloco `try-except`. Se a decomposição falhar (devido a problemas 
      de convergência ou matriz singular), a função recai silenciosamente para 
      a interpolação linear simples.

    References
    ----------
    .. [1] Larrubia, L. F. (2021). Detecção de anomalias, interpolação e 
       previsão em tempo real de séries temporais para operação de 
       reservatórios e distribuição de água (Dissertação de Mestrado). 
       Instituto de Matemática e Estatística, Universidade de São Paulo.
       DOI: 10.11606/D.45.2021.tde-10062021-231004
    """
        
    y = df_window[col]
    # STL do statsmodels requer dados sem NaNs na entrada, usamos interpolação linear simples como "chute" inicial
    y_guess = y.interpolate(method="linear").bfill().ffill()
    
    if len(y_guess) < 2 * period:
        return y_guess
        
    try:
        stl = STL(y_guess, period=period, robust=True)
        res = stl.fit()
        estimativa = res.trend + res.seasonal
        return y.fillna(estimativa)
    except:
        return y.interpolate(method="linear")

def imputador_interativo_larrubia(
        df: pd.DataFrame, 
        col_alvo: str, 
        gap_teste: int = 120, 
        period: int = 1440, 
        grau_pol: int = 5
    ) -> pd.DataFrame:
    """
    Orquestra a avaliação interativa e a aplicação dos métodos de imputação.

    Esta função atua como um pipeline de testes e preenchimento baseado na 
    metodologia de Larrubia (2021) [1]. O processo ocorre em três etapas:
    1. **Benchmark**: Isola uma janela contínua de dados válidos e introduz 
       uma falha artificial (*gap*) no centro dela.
    2. **Teste e Métricas**: Aplica os três métodos de imputação disponíveis 
       (Regressão com Resíduos, Decomposição Clássica e STL) sobre o *gap* 
       artificial, calculando os erros RMSE, MAE e MAPE em relação aos dados 
       originais ocultados.
    3. **Interação e Aplicação**: Exibe o relatório de desempenho no console e 
       pausa a execução, aguardando o usuário escolher o melhor método. O 
       método escolhido é então aplicado sobre a base de dados inteira.

    A construção geométrica do teste na janela temporal obedece à lógica:
    $$
    \begin{aligned}
    \text{Tamanho da Janela} &= 3 \times \text{period} \\
    \text{Início da Falha} &= \lfloor \frac{\text{Tamanho da Janela}}{2} \rfloor \\
    \text{Tamanho da Falha} &= \text{gap\_teste}
    \end{aligned}
    $$

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame completo contendo a série temporal original. Recomenda-se 
        que o índice seja do tipo `datetime`. Caso não seja, a função tentará 
        promover uma coluna chamada `"data"` ao índice.
    col_alvo : str
        Nome da coluna que contém os valores a serem analisados e imputados.
    gap_teste : int, optional
        Tamanho da falha artificial (em número de observações) gerada para 
        testar os métodos. O padrão é 120 (ex: 2 horas em dados minutais).
    period : int, optional
        Periodicidade (sazonalidade) dos dados, essencial para os métodos de 
        decomposição. O padrão é 1440 (ex: 1 dia em dados minutais).
    grau_pol : int, optional
        Grau do polinômio utilizado para o ajuste de tendência no Método 1 
        e Método 2. O padrão é 5.

    Returns
    -------
    pandas.DataFrame
        Uma cópia do DataFrame original acrescida de uma nova coluna chamada 
        `"{col_alvo}_filled"`, que contém a série temporal preenchida pelo 
        método escolhido pelo usuário. O índice é resetado ao final.

    See Also
    --------
    metodo_1_regressao_residuos : O Método 1 testado.
    metodo_2_decomposicao_classica : O Método 2 testado.
    metodo_3_stl : O Método 3 testado.
    mean_absolute_percentage_error : Função utilizada no cálculo do MAPE.

    Notes
    -----
    - **Bloqueio de Execução:** Esta função utiliza a instrução embutida `input()`. 
      Em ambientes de automação pura (sem interface interativa), isso fará o 
      script travar aguardando resposta. É ideal para uso em Jupyter Notebooks 
      ou execução manual em terminal.
    - **Requisito de Dados:** O método de teste falhará e retornará a base intacta 
      caso a série não possua um período contínuo e sem falhas de tamanho igual a 
      `3 * period` para servir como campo de provas.

    References
    ----------
    .. [1] Larrubia, L. F. (2021). Detecção de anomalias, interpolação e 
       previsão em tempo real de séries temporais para operação de 
       reservatórios e distribuição de água (Dissertação de Mestrado). 
       Instituto de Matemática e Estatística, Universidade de São Paulo.
       DOI: 10.11606/D.45.2021.tde-10062021-231004
    """
    print(f"\n--- Iniciando Avaliação para a coluna: {col_alvo} ---")
    
    # Garantir que o index é datetime para facilitar cortes (caso não seja, usaremos a coluna "data")
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index("data")
        
    # CALCULANDO QUALIDADE DOS MÉTODOS
    # Procurar um trecho sem nulos com tamanho 3x o período (ex: 3 dias)
    window_size = 3 * period
    y_valid = df[col_alvo].dropna()
    
    if len(y_valid) < window_size:
        print("Dados insuficientes para criar um benchmark confiável.")
        return df.reset_index(drop= True)
        
    # Pega uma janela válida e faz uma cópia profunda
    df_test = df.loc[y_valid.index[:window_size]].copy()
    y_true = df_test[col_alvo].copy()
    
    # Insere o buraco no meio da janela de teste
    start_gap = window_size // 2
    df_test.iloc[start_gap:start_gap+gap_teste, df_test.columns.get_loc(col_alvo)] = np.nan
    idx_gap = df_test.index[start_gap:start_gap+gap_teste]
    
    # Guardar os valores reais do buraco
    true_gap_values = y_true.loc[idx_gap]
    
    # TESTAR OS MÉTODOS
    print("Aplicando Metodo 1 (Regressão + Resíduos)...")
    res_1 = metodo_1_regressao_residuos(df_test.copy(), col_alvo, grau_pol= grau_pol)
    
    print("Aplicando Metodo 2 (Decomposição Clássica)...")
    res_2 = metodo_2_decomposicao_classica(df_test.copy(), col_alvo, period, grau_pol= grau_pol)
    
    print("Aplicando Metodo 3 (Decomposição STL)...")
    res_3 = metodo_3_stl(df_test.copy(), col_alvo, period)
    
    # CALCULAR MÉTRICAS
    metodos = ["Regressao + Residuos", "Decomposicao Classica", "STL"]
    resultados = [res_1.loc[idx_gap], res_2.loc[idx_gap], res_3.loc[idx_gap]]
    
    print("\n=== RELATÓRIO DE MÉTRICAS ===")
    for i, res in enumerate(resultados):
        rmse = np.sqrt(mean_squared_error(true_gap_values, res))
        mae = mean_absolute_error(true_gap_values, res)
        mape = mean_absolute_percentage_error(true_gap_values, res)
        print(f"[{i+1}] {metodos[i]}:")
        print(f"    RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")
        
    # INTERAÇÃO COM USUÁRIO
    escolha = input("\nQual método você deseja aplicar nos dados reais? (1, 2 ou 3): ")
    
    # APLICAÇÃO FINAL
    if escolha == "1":
        print("\nProcessando do Método 1 imputação na base completa...")
        df[f"{col_alvo}_filled"] = metodo_1_regressao_residuos(df, col_alvo)
    
    elif escolha == "2":
        print("\nProcessando do Método 2 imputação na base completa...")
        df[f"{col_alvo}_filled"] = metodo_2_decomposicao_classica(df, col_alvo, period)
    
    elif escolha == "3":
        print("\nProcessando do Método 3 imputação na base completa...")
        df[f"{col_alvo}_filled"] = metodo_3_stl(df, col_alvo, period)
    
    else:
        print("Opção inválida. Retornando dataframe original.")
        return df.reset_index()
        
    print(f"Os dados preenchidos estão na nova coluna: '{col_alvo}_filled'.")
    return df.reset_index()

