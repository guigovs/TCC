import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

def calcular_diferenca_percentual(valor_atual, melhor_valor, tipo_metrica='max', partidas=1):
    # Caso especial: quando o melhor_valor é 0
    if melhor_valor == 0:
        # Se ambos são 0, é o melhor valor
        if valor_atual == 0:
            return None, "normal"
        # Se o melhor é 0 mas o valor atual não é 0, calculamos taxa por partida
        # Calcular taxa por partida (valor_atual / partidas)
        taxa_valor = valor_atual / partidas
        # Para o melhor valor (0), a taxa também é 0
        taxa_melhor = 0
        
        # Calcular diferença percentual baseada nas taxas
        if taxa_melhor == 0:
            # Se a taxa do melhor é 0, qualquer taxa positiva é diferença infinita
            # Mas vamos mostrar a taxa absoluta por partida
            diferenca_taxa = taxa_valor * 100  # Mostrar como porcentagem da taxa
            # SEMPRE VERMELHO quando melhor_valor = 0 e valor_atual != 0
            return f"+{diferenca_taxa:.1f}%", "inverse"  # Vermelho
    
    # Cálculo normal quando melhor_valor != 0
    diferenca = ((valor_atual - melhor_valor) / melhor_valor) * 100
    
    if tipo_metrica == 'max':
        if valor_atual == melhor_valor:
            return None, "normal"  # Melhor valor, não mostra delta
        return f"{diferenca:.1f}%", "normal" if diferenca < 0 else "inverse"
    if valor_atual == melhor_valor:
        return None, "normal"  # Melhor valor, não mostra delta
    return f"+{diferenca:.1f}%", "normal" if diferenca < 0 else "inverse"

class AnalisadorFutebol:
    def __init__(self, df):
        self.df = df

    def filtrar_dados(self, jogador=None, temporadas=None, competicoes=None):
        """
        Aplica filtros ao conjunto de dados com base no nome do jogador, 
        temporadas específicas e competições selecionadas.
        """
        # Cria uma cópia do DataFrame original para evitar modificações indesejadas
        df_filtrado = self.df.copy()
        
        # Aplica o filtro de jogador, mantendo apenas as linhas correspondentes ao nome fornecido
        if jogador:
            df_filtrado = df_filtrado[df_filtrado['Jogador'] == jogador]

        # Restringe os dados às temporadas presentes na lista fornecida
        if temporadas:
            df_filtrado = df_filtrado[df_filtrado['Temporada'].isin(temporadas)]
            
        # Restringe os dados às competições presentes na lista fornecida
        if competicoes:
            df_filtrado = df_filtrado[df_filtrado['Competição'].isin(competicoes)]

        return df_filtrado

    def calcular_jogadores_similares(self, jogador_alvo, metricas, filtros_temporada, filtros_competicao, cols_media, cols_soma, n=5):
        """
        Identifica e retorna os jogadores estatisticamente mais semelhantes ao jogador alvo,
        utilizando a Distância Euclidiana sobre dados normalizados.
        """
        # Define o universo de comparação aplicando os filtros de temporada e competição
        df_universo = self.filtrar_dados(jogador=None, temporadas=filtros_temporada, competicoes=filtros_competicao)
        
        if df_universo.empty:
            return None, "Não há dados para os filtros selecionados."

        # Prepara o dicionário de agregação para consolidar os dados (uma linha por jogador)
        # Define se a métrica deve ser calculada pela média ou pela soma total
        agg_dict = {}
        metricas_validas = []
        
        for metrica in metricas:
            if metrica in df_universo.columns:
                metricas_validas.append(metrica)
                if metrica in cols_media:
                    agg_dict[metrica] = 'mean'
                else:
                    agg_dict[metrica] = 'sum'
        
        if not metricas_validas:
            return None, "Nenhuma métrica válida selecionada."

        # Realiza o agrupamento dos dados por jogador, aplicando as agregações definidas
        df_agrupado = df_universo.groupby('Jogador')[metricas_validas].agg(agg_dict).dropna()

        # Garante que o jogador alvo esteja presente no conjunto de dados para comparação
        if jogador_alvo not in df_agrupado.index:
            # Busca os dados do jogador alvo ignorando filtros de competição, mas respeitando a temporalidade
            df_alvo_raw = self.filtrar_dados(jogador=jogador_alvo, temporadas=filtros_temporada)
            if df_alvo_raw.empty:
                return None, f"O jogador {jogador_alvo} não possui dados nas temporadas selecionadas."
            
            # Processa os dados do jogador alvo separadamente e o insere no grupo principal
            df_alvo_agg = df_alvo_raw.groupby('Jogador')[metricas_validas].agg(agg_dict)
            df_agrupado = pd.concat([df_agrupado, df_alvo_agg])
            
            # Elimina duplicatas preservando a primeira ocorrência
            df_agrupado = df_agrupado.loc[~df_agrupado.index.duplicated(keep='first')]

        # Normaliza os dados utilizando o StandardScaler para colocar todas as métricas na mesma escala
        scaler = StandardScaler()
        try:
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df_agrupado),
                index=df_agrupado.index,
                columns=df_agrupado.columns
            )
        except ValueError:
            return None, "Erro na normalização. Verifique se há dados suficientes ou variação nas métricas."

        # Calcula a distância euclidiana entre o jogador alvo e todos os outros jogadores do conjunto
        vetor_jogador = df_scaled.loc[[jogador_alvo]]
        distancias = cdist(vetor_jogador, df_scaled, metric='euclidean').flatten()
        
        df_resultado = pd.DataFrame({
            'Distância Euclidiana': distancias
        }, index=df_scaled.index)
        
        # Ordena os resultados pela menor distância (maior similaridade) e seleciona os Top N
        df_resultado = df_resultado.sort_values('Distância Euclidiana')
        df_top_n = df_resultado.iloc[1:n+1]
        
        # Constrói o DataFrame final combinando os dados originais agregados com a métrica de distância calculada
        df_final = df_agrupado.loc[df_top_n.index].copy()
        df_final.insert(0, 'Similaridade (Distância)', df_top_n['Distância Euclidiana'])
        
        return df_final, None
    
    def gerar_ranking(self, metrica, lista_media, lista_soma, min_minutos=900, temporadas=None, competicoes=None, posicoes=None):
        """
        Gera um ranking ordenado de jogadores para uma métrica específica,
        respeitando critérios de desempate e filtros de contexto.
        """
        df_filtrado = self.df.copy()
        
        # Aplica os filtros contextuais de temporada, competição e posição tática
        if temporadas:
            df_filtrado = df_filtrado[df_filtrado['Temporada'].isin(temporadas)]
        if competicoes:
            df_filtrado = df_filtrado[df_filtrado['Competição'].isin(competicoes)]
        if posicoes:
            df_filtrado = df_filtrado[df_filtrado['Posição'].isin(posicoes)]
            
        tipo_agg = 'mean' if metrica in lista_media else 'sum'
        
        # Define as regras de agregação, mantendo informações auxiliares como Equipe e Posição
        agg_funcs = {metrica: tipo_agg, 'Minutos Jogados': 'sum', 'Equipe': 'last', 'Posição': 'first'}
        
        if metrica not in df_filtrado.columns:
            return None, f"Métrica '{metrica}' não encontrada nos dados."

        # Consolida os dados por jogador
        df_agrupado = df_filtrado.groupby('Jogador').agg(agg_funcs).reset_index()
        
        # Filtra jogadores que não atingiram a minutagem mínima exigida
        df_elegiveis = df_agrupado[df_agrupado['Minutos Jogados'] >= min_minutos].copy()
        
        if df_elegiveis.empty:
            return None, "Nenhum jogador atende aos critérios de minutos jogados neste filtro."
            
        # Ordena o ranking: Primeiramente pelo valor da métrica (decrescente)
        # Em caso de empate, prioriza quem jogou menos minutos (crescente)
        df_rank = df_elegiveis.sort_values(
            by=[metrica, 'Minutos Jogados'],
            ascending=[False, True]
        ).reset_index(drop=True)
        
        # Ajusta o índice para iniciar em 1, representando a posição no ranking
        df_rank.index = df_rank.index + 1
        df_rank.index.name = 'Ranking'
        df_rank = df_rank.reset_index()
        
        return df_rank, None
    
    def calcular_evolucoes(self, metrica, temporada_anterior, temporada_atual, lista_media, min_minutos=500, n=10, competicoes=None, posicoes=None):
        """
        Calcula a evolução de performance dos jogadores entre duas temporadas distintas
        para identificar os jogadores com maior evolução.
        """
        df = self.df.copy()
        
        # Aplica filtros globais de competição e posição antes de separar por temporada
        if competicoes:
            df = df[df['Competição'].isin(competicoes)]
        if posicoes:
            df = df[df['Posição'].isin(posicoes)]

        agg_type = 'mean' if metrica in lista_media else 'sum'
        
        # Isola e agrega os dados da temporada anterior, aplicando filtro de minutagem
        df_ant = df[(df['Temporada'] == temporada_anterior)].groupby('Jogador').agg({
            metrica: agg_type,
            'Minutos Jogados': 'sum',
            'Equipe': 'last'
        }).reset_index()
        df_ant = df_ant[df_ant['Minutos Jogados'] >= min_minutos]

        # Isola e agrega os dados da temporada atual, aplicando filtro de minutagem
        df_atu = df[(df['Temporada'] == temporada_atual)].groupby('Jogador').agg({
            metrica: agg_type,
            'Minutos Jogados': 'sum',
            'Equipe': 'last'
        }).reset_index()
        df_atu = df_atu[df_atu['Minutos Jogados'] >= min_minutos]

        if df_ant.empty or df_atu.empty:
            return None, "Dados insuficientes em uma das temporadas selecionadas."

        # Realiza a junção dos dados das duas temporadas baseada no nome do jogador
        df_comp = pd.merge(df_ant, df_atu, on='Jogador', suffixes=('_ant', '_atu'))
        
        # Calcula o delta (evolução) da métrica entre os dois períodos
        df_comp['Melhora'] = df_comp[f'{metrica}_atu'] - df_comp[f'{metrica}_ant']
        
        # Ordena os jogadores pela magnitude da melhora (decrescente)
        # Utiliza a menor minutagem na temporada atual como critério de desempate
        top_evolucoes = df_comp.sort_values(
            by=['Melhora', 'Minutos Jogados_atu'],
            ascending=[False, True]
        ).head(n).reset_index(drop=True)
        
        top_evolucoes.index += 1
        
        return top_evolucoes, None

    def calcular_consistencia(self, metrica, lista_media, min_temporadas=3, min_minutos=2000, n=10, competicoes=None, posicoes=None, temporadas=None):
        """
        Avalia a regularidade dos jogadores calculando o Coeficiente de Variação (CV)
        ao longo de múltiplas temporadas.
        """
        df = self.df.copy()
        
        # Aplica filtros iniciais para definir o escopo da análise
        if temporadas:
            df = df[df['Temporada'].isin(temporadas)]
        if competicoes:
            df = df[df['Competição'].isin(competicoes)]
        if posicoes:
            df = df[df['Posição'].isin(posicoes)]

        agg_type = 'mean' if metrica in lista_media else 'sum'

        # Agrupa os dados por Jogador e Temporada para obter a performance anual
        dados_sazonais = df.groupby(['Jogador', 'Temporada']).agg(
            valor_metrica=(metrica, agg_type),
            minutos=('Minutos Jogados', 'sum'),
            equipe=('Equipe', 'last')
        ).reset_index()

        # Calcula estatísticas descritivas por jogador (média, desvio padrão, contagem de temporadas)
        stats = dados_sazonais.groupby('Jogador').agg(
            media_performance=('valor_metrica', 'mean'),
            desvio_padrao=('valor_metrica', 'std'),
            num_temporadas=('Temporada', 'nunique'),
            equipe_atual=('equipe', 'last'),
            total_minutos=('minutos', 'sum')
        ).reset_index()

        # Filtra jogadores que não atingem o número mínimo de temporadas ou minutos totais no período
        stats = stats[stats['num_temporadas'] >= min_temporadas]
        stats = stats[stats['total_minutos'] >= min_minutos]
        
        if stats.empty:
            return None, "Nenhum jogador atende aos critérios de temporadas e minutos totais selecionados."

        # Calcula o Coeficiente de Variação (CV = Desvio Padrão / Média)
        # Adiciona um epsilon (1e-9) para evitar divisão por zero
        stats['CV'] = stats['desvio_padrao'] / (stats['media_performance'].abs() + 1e-9)
        
        # Restringe a análise aos jogadores com performance média acima da mediana (elite/relevantes)
        mediana = stats['media_performance'].median()
        stats_elite = stats[stats['media_performance'] >= mediana].copy()
        
        if stats_elite.empty:
            return None, "Nenhum jogador acima da mediana de performance encontrado."

        # Ordena pela consistência (menor CV primeiro) e desempata pela média de performance (maior melhor)
        top_consistentes = stats_elite.sort_values(
            by=['CV', 'media_performance'],
            ascending=[True, False]
        ).head(n).reset_index(drop=True)
        
        top_consistentes.index += 1
        
        return top_consistentes, None

    def analise_idade_performance(self, metrica, lista_media, posicoes=None, competicoes=None, temporadas=None, min_jogadores=5, min_minutos=900):
        """
        Constrói uma curva de performance baseada na idade dos jogadores,
        agregando dados individuais e calculando médias por faixa etária.
        """
        df = self.df.copy()
        agg_type = 'mean' if metrica in lista_media else 'sum'
        
        # Aplica filtros de escopo para refinar a base de dados
        if temporadas:
            df = df[df['Temporada'].isin(temporadas)]
        if competicoes:
            df = df[df['Competição'].isin(competicoes)]
        if posicoes:
            df = df[df['Posição'].isin(posicoes)]

        # Remove registros que não atendem ao critério mínimo de minutos jogados
        df = df[df['Minutos Jogados'] >= min_minutos]

        # Agrupa primeiramente por Jogador e Idade para obter a performance individual em cada idade
        df_player_age = df.groupby(['Jogador', 'Idade']).agg({
            metrica: agg_type,
            'Minutos Jogados': 'sum'
        }).reset_index()
        
        # Agrupa os resultados por Idade para obter a média geral da métrica para aquela idade
        curva = df_player_age.groupby('Idade').agg(
            performance_media=(metrica, 'mean'),
            contagem_jogadores=('Jogador', 'nunique'),
            media_minutos=('Minutos Jogados', 'mean')
        ).reset_index()
        
        # Filtra idades que possuem pouca amostragem de jogadores para garantir relevância estatística
        curva = curva[curva['contagem_jogadores'] >= min_jogadores]
        
        if curva.empty:
            return None, "Dados insuficientes. Tente reduzir os filtros de minutos ou número mínimo de jogadores."
            
        return curva, None