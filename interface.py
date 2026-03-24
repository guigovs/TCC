import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
from analises import AnalisadorFutebol, calcular_diferenca_percentual

# Inicializa a variável como vazia para o Python parar de reclamar
CLIENT = None

# Configura a chave se ela existir
if "GEMINI_CHAVE" in st.secrets:
    CLIENT = genai.Client(api_key=st.secrets["GEMINI_CHAVE"])
else:
    st.error("Chave da API não encontrada. Verifique o arquivo .streamlit/secrets.toml")

st.set_page_config(
    layout="wide",
    page_title="Análise de Jogadores",
    page_icon="./imagens/football.png"
)

@st.cache_data
def carregar_dados():
    df = pd.read_csv('lista_jogadores_20-24.csv')
    return df

meu_dataframe = carregar_dados()
analisador = AnalisadorFutebol(meu_dataframe)

METRICAS_BLACKLIST = [
    'Jogador', 'Nacionalidade', 'Posição', 'Equipe', 'Competição', 
    'Temporada', 'Idade', 'Ano de Nascimento'
]

colunas_numericas_originais = meu_dataframe.select_dtypes(include=['number']).columns

METRICAS_PARA_MEDIA = [
    col for col in colunas_numericas_originais
    if ('%' in col or
        '/' in col or 
        'Média' in col or 
        'Distância' in col or
        'Percentual' in col or
        col in [
            'Saldo de Gols por 90 Minutos com o Jogador em Campo - Saldo de Gols por 90 Minutos sem o Jogador em Campo',
            'Saldo de Gols Esperados por 90 Minutos com o Jogador em Campo - Saldo de Gols Esperados por 90 Minutos sem o Jogador em Campo'
            ])
       and col not in METRICAS_BLACKLIST
]

METRICAS_PARA_SOMAR = [
    col for col in colunas_numericas_originais
    if col not in METRICAS_BLACKLIST and col not in METRICAS_PARA_MEDIA
]

st.title("_FutEstat_", text_alignment='center')

tab_individuais, tab_rankings_globais, tab_dados, tab_chatbot = st.tabs([
    "**Análises Individuais**", 
    "**Rankings Globais**", 
    "**Dados Brutos**",
    "**Chatbot**"
])

with tab_individuais:
    st.header("Painel de Análises Interativas")
    
    jogadores = sorted(meu_dataframe['Jogador'].unique())

    col_filtro_jogador, _ = st.columns([1, 3])
    with col_filtro_jogador:
        jogador_selecionado = st.selectbox(
            "**Filtrar por Jogador:**",
            jogadores,
            index=None,
            placeholder="Selecione o jogador",
            label_visibility="collapsed"
        )
        
    st.markdown("---")

    if jogador_selecionado:
        df_jogador_geral = meu_dataframe[meu_dataframe['Jogador'] == jogador_selecionado]
        
        sub_tab_visao_geral, sub_tab_comparativa, sub_tab_comparativa_pos, sub_tab_temporal, sub_tab_rankings = st.tabs([
            "**Visão Geral**", "**Análise Comparativa Individual**", "**Análise Comparativa por Posição**", "**Análise Temporal**", "**Ranking e Similaridade**"
        ])

        with sub_tab_visao_geral:
            temporadas_disponiveis = sorted(df_jogador_geral['Temporada'].unique())
            
            col1, col2 = st.columns(2)
            with col1:
                temporadas_selecionadas = st.multiselect(
                    "**Temporadas:**", 
                    temporadas_disponiveis,
                    default=[],
                    placeholder="Todas as temporadas"
                )

            if temporadas_selecionadas:
                df_filtrado_por_temporada = df_jogador_geral[df_jogador_geral['Temporada'].isin(temporadas_selecionadas)]
                competicoes_disponiveis = sorted(df_filtrado_por_temporada['Competição'].unique())
            else:
                competicoes_disponiveis = sorted(df_jogador_geral['Competição'].unique())

            with col2:
                competicoes_selecionadas = st.multiselect(
                    "**Competições:**", 
                    competicoes_disponiveis,
                    default=[],
                    placeholder="Todas as competições"
                )

            temporadas_para_analise = temporadas_selecionadas if temporadas_selecionadas else temporadas_disponiveis
            competicoes_para_analise = competicoes_selecionadas if competicoes_selecionadas else competicoes_disponiveis

            df_filtrado = meu_dataframe[
                (meu_dataframe['Jogador'] == jogador_selecionado) &
                (meu_dataframe['Temporada'].isin(temporadas_para_analise)) &
                (meu_dataframe['Competição'].isin(competicoes_para_analise))
            ]

            if df_filtrado.empty:
                st.warning("Nenhum dado encontrado para o jogador com os filtros selecionados.")
            else:
                col1, col2 = st.columns([1, 2.5])
                with col1:
                    st.subheader(f"{jogador_selecionado} - {int(df_filtrado['Idade'].max())} anos")
                    st.markdown(f"**Posição(ões):** {df_filtrado['Posição'].mode()[0]}")
                    st.markdown(f"**Nacionalidade:** {df_filtrado['Nacionalidade'].iloc[0]}")
                    st.markdown(f"**Equipe(s):** {', '.join(df_filtrado['Equipe'].unique())}")
                    st.markdown(f"**Competição(ões):** {', '.join(df_filtrado['Competição'].unique())}")
                    
                with col2:
                    st.subheader("**Resumo de Participação**")
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Partidas", df_filtrado['Partidas'].sum())
                    kpi2.metric("Como Titular", df_filtrado['Partidas como Titular'].sum())
                    kpi3.metric("Como Reserva", df_filtrado['Partidas como Substituto'].sum())
                    
                    kpi4, kpi5, kpi6 = st.columns(3)
                    kpi4.metric("Minutos Jogados", df_filtrado['Minutos Jogados'].sum())
                    kpi5.metric("Minutos Jogados por Partida", f"{df_filtrado['Minutos Jogados / Partidas'].mean():.0f}")
                    kpi6.metric("Minutos Jogados por Partida como Titular", f"{df_filtrado['Minutos / Partidas como Titular'].mean():.0f}")

                st.divider()

                st.subheader("**Estatísticas Chave e Indicadores de Impacto**")
                kpi_main1, kpi_main2, kpi_main3, kpi_main4 = st.columns(4)
                
                with kpi_main1: # Defensivo
                    st.markdown("<h5 style='text-align: left;'>Defensivo</h5>", unsafe_allow_html=True)
                    st.metric("Desarmes + Interceptações", df_filtrado['Desarmes + Interceptações'].sum())
                    st.metric("Bolas Recuperadas", df_filtrado['Bolas Recuperadas'].sum())
                    st.metric("Faltas Cometidas", df_filtrado['Faltas Cometidas'].sum())
                    st.metric("Cartões Amarelos / Cartões Vermelhos", f"{df_filtrado['Cartões Amarelos'].sum()} / {df_filtrado['Cartões Vermelhos'].sum()}")
                
                with kpi_main2: # Criação
                    st.markdown("<h5 style='text-align: left'>Criação</h5>", unsafe_allow_html=True)
                    st.metric("Assistências / Assistências Esperadas", f"{df_filtrado['Assistências'].sum()} / {df_filtrado['Assistências Esperadas'].sum():.1f}")
                    st.metric("Passes Chave", df_filtrado['Passes Chave'].sum())
                    st.metric("Passes Progressivos", df_filtrado['Passes Progressivos Efetuados'].sum())
                    st.metric("Cruzamentos", df_filtrado['Cruzamentos'].sum())
                
                with kpi_main3: # Ofensivo
                    st.markdown("<h5 style='text-align: left;'>Ofensivo</h5>", unsafe_allow_html=True)
                    st.metric("Gols / Gols Esperados", f"{df_filtrado['Gols'].sum()} / {df_filtrado['Gols Esperados'].sum():.1f}")
                    st.metric("Participações em Gols / 90", f"{df_filtrado['Gols+Assistências / 90'].mean():.2f}")
                    total_gols = df_filtrado['Gols'].sum()
                    total_finalizacoes = df_filtrado['Finalizações'].sum()
                    gols_por_finalizacao = (total_gols / total_finalizacoes) if total_finalizacoes > 0 else 0
                    st.metric("Gols por Finalização", f"{gols_por_finalizacao:.2f}")
                    st.metric("Finalizações no Gol / Finalizações", f"{df_filtrado['Finalizações no Gol'].sum()} / {df_filtrado['Finalizações'].sum()}")
                    
                with kpi_main4: # Impacto Coletivo
                    st.markdown("<h5 style='text-align: left;'>Impacto na Equipe</h5>", unsafe_allow_html=True)
                    st.metric("Média de Pontos por Jogo", f"{df_filtrado['Média de Pontos Ganhos pela Equipe em Jogos Participados pelo Jogador'].mean():.2f}")
                    gols_assistencias_total = df_filtrado['Gols + Assistências'].sum()
                    gols_equipe_total = df_filtrado['Gols Marcados pela Equipe com o Jogador em Campo'].sum()
                    participacao_gols_equipe = (gols_assistencias_total / gols_equipe_total) * 100 if gols_equipe_total > 0 else 0
                    st.metric(
                        label="% de Participação nos Gols da Equipe",
                        value=f"{participacao_gols_equipe:.1f}%",
                        help="Dos gols marcados pela equipe com ele em campo, em quantos % ele participou (gol ou assistência)."
                    )
                    st.metric(
                        "Saldo de Gols / Saldo de Gols Esperados",
                        f"{df_filtrado['Saldo de Gols com o Jogador em Campo'].sum()} / {df_filtrado['Saldo de Gols Esperados da Equipe com o Jogador em Campo'].sum():.1f}",
                        help="Saldo de gols da equipe com o jogador em campo."
                    )
                    impacto_saldo_gols = df_filtrado['Saldo de Gols por 90 Minutos com o Jogador em Campo - Saldo de Gols por 90 Minutos sem o Jogador em Campo'].mean()
                    st.metric(
                        label="Saldo de Gols com/sem o Jogador (90 Min)",
                        value=f"{impacto_saldo_gols:+.2f}",
                        help="Mostra quanto o saldo de gols da equipe muda por 90 minutos quando o jogador está em campo em relação a quando não está."
                    )

                st.divider()
                
                st.subheader("**Perfil de Jogo**")
                
                col_vis1, col_vis2 = st.columns(2)
                with col_vis1:
                    st.markdown("##### Toques na Bola por Zona do Campo")
                    toques_data = {
                        'Zona': ['Terço Defensivo', 'Terço Médio', 'Terço Ofensivo'],
                        'Toques': [
                            df_filtrado['Toques na Bola no Terço Defensivo do Campo'].sum(),
                            df_filtrado['Toques na Bola no Terço Médio do Campo'].sum(),
                            df_filtrado['Toques na Bola no Terço Ofensivo do Campo'].sum()
                        ]
                    }
                    df_toques = pd.DataFrame(toques_data)

                    try:
                        pitch_image = plt.imread('./imagens/campo_tercos.jpg')
                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.patch.set_alpha(0.0)
                        ax.patch.set_alpha(0.0)
                        ax.imshow(pitch_image, extent=[-0.5, 2.5, 0, df_toques['Toques'].max() * 1.1], aspect='auto', alpha=1.0)
                        bars = ax.bar(df_toques['Zona'], df_toques['Toques'], color=["#ffffff"], alpha=0.85, zorder=2, width=0.6)
                        ax.get_yaxis().set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(axis='x', colors='white', rotation=0, labelsize=10, pad=7)
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', color='black', weight='bold', zorder=3)
                        st.pyplot(fig)
                    except Exception as e:
                        st.bar_chart(df_toques.set_index('Zona'))
                
                with col_vis2:
                    st.markdown("##### Distribuição de Passes")
                    passes_data = {
                        'Tipo': ['Passes Longos\n(>27.4 metros)', 'Passes Médios\n(13.7 - 27.4 metros)', 'Passes Curtos\n(4.6 - 13.7 metros)'],
                        'Completos': [
                            df_filtrado['Passes Longos Completos'].sum(),
                            df_filtrado['Passes Médios Completos'].sum(),
                            df_filtrado['Passes Curtos Completos'].sum()
                        ],
                        'Efetuados': [
                            df_filtrado['Passes Longos Efetuados'].sum(),
                            df_filtrado['Passes Médios Efetuados'].sum(),
                            df_filtrado['Passes Curtos Efetuados'].sum()
                        ]
                    }
                    df_passes = pd.DataFrame(passes_data)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    bars_efetuados = sns.barplot(x="Efetuados", y="Tipo", data=df_passes, color="#e4f2da", ax=ax, label="Efetuados")
                    bars_completos = sns.barplot(x="Completos", y="Tipo", data=df_passes, color="#22c912", ax=ax, label="Completos")
                    ax.set_xlabel("Quantidade de Passes", color='white', fontsize=12, labelpad=10)
                    ax.set_ylabel("")
                    ax.tick_params(axis='x', colors='white', labelsize=8)
                    ax.tick_params(axis='y', colors='white', labelsize=9)
                    ax.legend(loc="upper right", frameon=False, labelcolor="white")
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig)

                st.divider()
                st.dataframe(df_filtrado, hide_index=True)

        with sub_tab_comparativa:
            col_input, _ = st.columns([1, 3])
            with col_input:
                num_jogadores = st.number_input(
                    "**Quantos Jogadores Deseja Comparar?**",
                    min_value=1,
                    max_value=4,
                    value=1,
                    key="num_jogadores_ind"
                )

            larguras = [0.25] * num_jogadores
            if num_jogadores < 4:
                larguras.extend([1.0 - sum(larguras)])
            colunas_jogadores = st.columns(larguras)
            
            jogadores_comparacao = []
            jogadores_options = sorted(meu_dataframe['Jogador'].unique())

            for i in range(num_jogadores):
                with colunas_jogadores[i]:
                    jogador = st.selectbox(
                        f"**Jogador {i+1}**",
                        jogadores_options,
                        index=None,
                        key=f"comp_jogador_{i+1}",
                        placeholder="Selecione o jogador",
                        label_visibility="collapsed"
                    )
                    if jogador:
                        jogadores_comparacao.append(jogador)

            st.markdown("---")

            if jogador_selecionado and jogadores_comparacao:
                todos_jogadores_comparacao = [jogador_selecionado] + [j for j in jogadores_comparacao if j and j != jogador_selecionado]
                todos_jogadores_comparacao = list(dict.fromkeys(todos_jogadores_comparacao))
            elif jogadores_comparacao:
                todos_jogadores_comparacao = list(dict.fromkeys(jogadores_comparacao))
            else:
                todos_jogadores_comparacao = []
                if jogador_selecionado:
                    st.info("Selecione pelo menos um jogador para comparação usando os seletores acima.")
                else:
                    st.info("Selecione um jogador na aba principal e adicione jogadores para comparação.")

            if todos_jogadores_comparacao:
                df_jogadores_selecionados = meu_dataframe[meu_dataframe['Jogador'].isin(todos_jogadores_comparacao)]
                
                colunas_filtros = st.columns(len(todos_jogadores_comparacao))
                filtros_por_jogador = {}
                
                for i, jogador in enumerate(todos_jogadores_comparacao):
                    with colunas_filtros[i]:
                        st.markdown(f"**Filtros para {jogador}**")
                        df_jogador = df_jogadores_selecionados[df_jogadores_selecionados['Jogador'] == jogador]
                        
                        temporadas_jogador = sorted(df_jogador['Temporada'].unique())
                        temporadas_selecionadas = st.multiselect(
                            "**Temporadas:**",
                            temporadas_jogador, default=[], key=f"temp_{jogador}",
                            placeholder=f"Todas temporadas de {jogador}"
                        )
                        
                        competicoes_jogador = sorted(df_jogador['Competição'].unique())
                        competicoes_selecionadas = st.multiselect(
                            "**Competições:**",
                            competicoes_jogador, default=[], key=f"comp_{jogador}",
                            placeholder=f"Todas competições de {jogador}"
                        )
                        
                        filtros_por_jogador[jogador] = {
                            'temporadas': temporadas_selecionadas,
                            'competicoes': competicoes_selecionadas
                        }

                st.divider()
                
                colunas_info = st.columns(len(todos_jogadores_comparacao))
                
                for i, jogador in enumerate(todos_jogadores_comparacao):
                    with colunas_info[i]:
                        df_jogador = df_jogadores_selecionados[df_jogadores_selecionados['Jogador'] == jogador]
                        
                        df_filtrado_temp = df_jogador.copy()
                        filtros = filtros_por_jogador[jogador]
                        
                        if filtros['temporadas']:
                            df_filtrado_temp = df_filtrado_temp[df_filtrado_temp['Temporada'].isin(filtros['temporadas'])]
                        if filtros['competicoes']:
                            df_filtrado_temp = df_filtrado_temp[df_filtrado_temp['Competição'].isin(filtros['competicoes'])]
                        
                        if not df_filtrado_temp.empty:
                            st.subheader(f"{jogador} - {int(df_filtrado_temp['Idade'].max())} anos")
                            st.markdown(f"**Posição(ões):** {df_filtrado_temp['Posição'].mode()[0]}")
                            st.markdown(f"**Nacionalidade:** {df_filtrado_temp['Nacionalidade'].iloc[0]}")
                            st.markdown(f"**Equipe(s):** {', '.join(df_filtrado_temp['Equipe'].unique())}")
                            st.markdown(f"**Competição(ões):** {', '.join(df_filtrado_temp['Competição'].unique())}")
                        else:
                            st.warning(f"Nenhum dado encontrado para {jogador} com os filtros selecionados.")

                st.markdown("---")
                
                dfs_filtrados = []
                for jogador in todos_jogadores_comparacao:
                    df_jogador = df_jogadores_selecionados[df_jogadores_selecionados['Jogador'] == jogador].copy()
                    filtros = filtros_por_jogador[jogador]
                    if filtros['temporadas']:
                        df_jogador = df_jogador[df_jogador['Temporada'].isin(filtros['temporadas'])]
                    if filtros['competicoes']:
                        df_jogador = df_jogador[df_jogador['Competição'].isin(filtros['competicoes'])]
                    dfs_filtrados.append(df_jogador)
                
                if dfs_filtrados:
                    df_comparacao = pd.concat(dfs_filtrados, ignore_index=True)
                else:
                    df_comparacao = pd.DataFrame()
                
                if df_comparacao.empty:
                    st.warning("Nenhum dado encontrado para os jogadores selecionados com os filtros aplicados.")
                else:
                    df_avg = pd.DataFrame()
                    if METRICAS_PARA_MEDIA:
                        colunas_media_existentes = [col for col in METRICAS_PARA_MEDIA if col in df_comparacao.columns]
                        if colunas_media_existentes:
                            df_avg = df_comparacao.groupby('Jogador')[colunas_media_existentes].mean().reset_index()

                    df_agg = pd.DataFrame()
                    if METRICAS_PARA_SOMAR:
                        colunas_soma_existentes = [col for col in METRICAS_PARA_SOMAR if col in df_comparacao.columns]
                        if colunas_soma_existentes:
                            df_agg = df_comparacao.groupby('Jogador')[colunas_soma_existentes].sum().reset_index()
                    
                    df_final_agg_temp = pd.DataFrame(df_comparacao['Jogador'].unique(), columns=['Jogador'])

                    if not df_agg.empty:
                        df_final_agg_temp = pd.merge(df_final_agg_temp, df_agg, on='Jogador', how='left')
                    if not df_avg.empty:
                        df_final_agg_temp = pd.merge(df_final_agg_temp, df_avg, on='Jogador', how='left')
                    
                    # Define o Index como Jogador para facilitar a leitura nas métricas
                    df_comparacao_final = df_final_agg_temp.set_index('Jogador').fillna(0)
                    
                    # Seção de Participação
                    st.subheader("Participação e Minutos")
                    melhores_participacao = {
                        'partidas': df_comparacao_final['Partidas'].max(),
                        'titular': df_comparacao_final['Partidas como Titular'].max(),
                        'reserva': df_comparacao_final['Partidas como Substituto'].max(),
                        'minutos': df_comparacao_final['Minutos Jogados'].max(),
                        'min_partida': df_comparacao_final['Minutos Jogados / Partidas'].max(),
                        'min_partida_titular': df_comparacao_final['Minutos / Partidas como Titular'].max(),
                        'perc_minutos': df_comparacao_final['Percentual de Minutos Jogados na Temporada'].max()
                    }
                    metricas_participacao = [
                        ("Partidas", 'partidas', 'max', False, 'Partidas'),
                        ("Partidas como Titular", 'titular', 'max', False, 'Partidas como Titular'),
                        ("Partidas como Reserva", 'reserva', 'max', False, 'Partidas como Substituto'),
                        ("Minutos Jogados", 'minutos', 'max', False, 'Minutos Jogados'),
                        ("Minutos Jogados por Partida", 'min_partida', 'max', False, 'Minutos Jogados / Partidas'),
                        ("Minutos Jogados por Partida (Titular)", 'min_partida_titular', 'max', False, 'Minutos / Partidas como Titular'),
                        ("% de Minutos Jogados", 'perc_minutos', 'max', True, 'Percentual de Minutos Jogados na Temporada')
                    ]
                    for label, chave, tipo, eh_porcentagem, col_name in metricas_participacao:
                        colunas = st.columns(len(todos_jogadores_comparacao))
                        for i, jogador in enumerate(todos_jogadores_comparacao):
                            with colunas[i]:
                                if label == "Partidas":
                                    st.markdown(f"**{jogador}**")
                                
                                partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                valor = df_comparacao_final.loc[jogador, col_name]
                                
                                if col_name in METRICAS_PARA_MEDIA:
                                    formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                else:
                                    formato = f"{int(valor)}"
                                
                                delta, cor = calcular_diferenca_percentual(valor, melhores_participacao[chave], tipo, partidas_jogador)
                                st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                    st.markdown("---")

                    # Seção de Impacto na Equipe
                    st.subheader("Impacto na Equipe")
                    for jogador in todos_jogadores_comparacao:
                        g_a = df_comparacao_final.loc[jogador, 'Gols + Assistências']
                        g_e = df_comparacao_final.loc[jogador, 'Gols Marcados pela Equipe com o Jogador em Campo']
                        df_comparacao_final.loc[jogador, 'part_gols_calculado'] = (g_a / max(1, g_e)) * 100

                    melhores_impacto = {
                        'pontos': df_comparacao_final['Média de Pontos Ganhos pela Equipe em Jogos Participados pelo Jogador'].max(),
                        'participacao_gols': df_comparacao_final['part_gols_calculado'].max(),
                        'saldo_gols': df_comparacao_final['Saldo de Gols com o Jogador em Campo'].max(),
                        'saldo_esperado': df_comparacao_final['Saldo de Gols Esperados da Equipe com o Jogador em Campo'].max()
                    }
                    metricas_impacto = [
                        ("Média de Pontos por Jogo", 'pontos', 'max', False, 'Média de Pontos Ganhos pela Equipe em Jogos Participados pelo Jogador'),
                        ("% de Participação nos Gols da Equipe", 'participacao_gols', 'max', True, 'part_gols_calculado'),
                        ("Saldo de Gols", 'saldo_gols', 'max', False, 'Saldo de Gols com o Jogador em Campo'),
                        ("Saldo de Gols Esperados", 'saldo_esperado', 'max', False, 'Saldo de Gols Esperados da Equipe com o Jogador em Campo')
                    ]
                    for label, chave, tipo, eh_porcentagem, col_name in metricas_impacto:
                        colunas = st.columns(len(todos_jogadores_comparacao))
                        for i, jogador in enumerate(todos_jogadores_comparacao):
                            with colunas[i]:
                                if label == "Média de Pontos por Jogo":
                                    st.markdown(f"**{jogador}**")
                                
                                partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                valor = df_comparacao_final.loc[jogador, col_name]
                                
                                if col_name in METRICAS_PARA_MEDIA or chave == 'participacao_gols':
                                    formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                else: 
                                    formato = f"{int(valor)}"

                                delta, cor = calcular_diferenca_percentual(valor, melhores_impacto[chave], tipo, partidas_jogador)
                                st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                    st.markdown("---")

                    # Seção Defensiva
                    st.subheader("Setor Defensivo")
                    melhores_defensivo = {
                        'desarmes': df_comparacao_final['Desarmes'].max(),
                        'interceptacoes': df_comparacao_final['Interceptações'].max(),
                        'bolas_recuperadas': df_comparacao_final['Bolas Recuperadas'].max(),
                        'faltas_cometidas': df_comparacao_final['Faltas Cometidas'].min(),
                        'cartoes_amarelos': df_comparacao_final['Cartões Amarelos'].min(),
                        'cartoes_vermelhos': df_comparacao_final['Cartões Vermelhos'].min(),
                        'bloqueios': df_comparacao_final['Bloqueios'].max(),
                        'perc_desarmes_drible': df_comparacao_final['Percentual de Sucesso em Desarmes em Dribles Adversários'].max(),
                        'perc_aereas': df_comparacao_final['Percentual de Disputas Aéreas Vencidas'].max(),
                        'erros_finalizacao': df_comparacao_final['Erros que Resultaram em Finalizações Adversárias'].min()
                    }
                    metricas_defensivo = [
                        ("Desarmes", 'desarmes', 'max', False, 'Desarmes'),
                        ("Interceptações", 'interceptacoes', 'max', False, 'Interceptações'),
                        ("Bloqueios", 'bloqueios', 'max', False, 'Bloqueios'),
                        ("Bolas Recuperadas", 'bolas_recuperadas', 'max', False, 'Bolas Recuperadas'),
                        ("% de Sucesso de Desarmes em Dribles", 'perc_desarmes_drible', 'max', True, 'Percentual de Sucesso em Desarmes em Dribles Adversários'),
                        ("% de Disputas Aéreas Vencidas", 'perc_aereas', 'max', True, 'Percentual de Disputas Aéreas Vencidas'),
                        ("Erros para Finalizações Adversárias", 'erros_finalizacao', 'min', False, 'Erros que Resultaram em Finalizações Adversárias'),
                        ("Faltas Cometidas", 'faltas_cometidas', 'min', False, 'Faltas Cometidas'),
                        ("Cartões Amarelos", 'cartoes_amarelos', 'min', False, 'Cartões Amarelos'),
                        ("Cartões Vermelhos", 'cartoes_vermelhos', 'min', False, 'Cartões Vermelhos')
                    ]
                    for label, chave, tipo, eh_porcentagem, col_name in metricas_defensivo:
                        colunas = st.columns(len(todos_jogadores_comparacao))
                        for i, jogador in enumerate(todos_jogadores_comparacao):
                            with colunas[i]:
                                if label == "Desarmes":
                                    st.markdown(f"**{jogador}**")
                                
                                partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                valor = df_comparacao_final.loc[jogador, col_name]

                                if col_name in METRICAS_PARA_MEDIA:
                                    formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                else:
                                    formato = f"{int(valor)}"
                                
                                delta, cor = calcular_diferenca_percentual(valor, melhores_defensivo.get(chave, valor), tipo, partidas_jogador)
                                st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                    st.markdown("---")

                    # Seção de Criação
                    st.subheader("Setor de Criação")
                    melhores_criacao = {
                        'assistencias_xa': df_comparacao_final['Assistências'].max(),
                        'xa': df_comparacao_final['Assistências Esperadas'].max(),
                        'passes_chave': df_comparacao_final['Passes Chave'].max(),
                        'passes_progressivos': df_comparacao_final['Passes Progressivos Efetuados'].max(),
                        'cruzamentos': df_comparacao_final['Cruzamentos'].max(),
                        'sca': df_comparacao_final['Ações que Resultaram em Finalização'].max(),
                        'gca': df_comparacao_final['Ações que Resultaram em Gol'].max(),
                        'perc_dribles': df_comparacao_final['Percentual de Dribles Bem-sucedidos'].max(),
                        'passes_terco_final': df_comparacao_final['Passes Completos no Terço Final do Campo'].max(),
                        'faltas_sofridas': df_comparacao_final['Faltas Sofridas'].max(),
                        'perda_posse': df_comparacao_final['Perda de Posse de Bola'].min()
                    }
                    metricas_criacao = [
                        ("Assistências", 'assistencias_xa', 'max', False, 'Assistências'),
                        ("Assistências Esperadas", 'xa', 'max', False, 'Assistências Esperadas'),
                        ("Passes Chave", 'passes_chave', 'max', False, 'Passes Chave'),
                        ("Passes Progressivos", 'passes_progressivos', 'max', False, 'Passes Progressivos Efetuados'),
                        ("Ações Resultantes em Finalização", 'sca', 'max', False, 'Ações que Resultaram em Finalização'),
                        ("Ações Resultantes em Gol", 'gca', 'max', False, 'Ações que Resultaram em Gol'),
                        ("% de Dribles Bem-sucedidos", 'perc_dribles', 'max', True, 'Percentual de Dribles Bem-sucedidos'),
                        ("Passes Certos no Terço Final", 'passes_terco_final', 'max', False, 'Passes Completos no Terço Final do Campo'),
                        ("Faltas Sofridas", 'faltas_sofridas', 'max', False, 'Faltas Sofridas'),
                        ("Cruzamentos", 'cruzamentos', 'max', False, 'Cruzamentos'),
                        ("Perdas de Posse de Bola", 'perda_posse', 'min', False, 'Perda de Posse de Bola')
                    ]
                    for label, chave, tipo, eh_porcentagem, col_name in metricas_criacao:
                        colunas = st.columns(len(todos_jogadores_comparacao))
                        for i, jogador in enumerate(todos_jogadores_comparacao):
                            with colunas[i]:
                                if label == "Assistências":
                                    st.markdown(f"**{jogador}**")
                                
                                partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                valor = df_comparacao_final.loc[jogador, col_name]

                                if col_name in METRICAS_PARA_MEDIA:
                                    formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                else:
                                    formato = f"{int(valor)}"
                                
                                delta, cor = calcular_diferenca_percentual(valor, melhores_criacao.get(chave, valor), tipo, partidas_jogador)
                                st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                    st.markdown("---")

                    # Seção Ofensiva
                    st.subheader("Setor Ofensivo")
                    for jogador in todos_jogadores_comparacao:
                        gols = df_comparacao_final.loc[jogador, 'Gols']
                        finalizacoes = df_comparacao_final.loc[jogador, 'Finalizações']
                        df_comparacao_final.loc[jogador, 'gols_por_finalizacao_calculado'] = gols / max(1, finalizacoes)

                    melhores_ofensivo = {
                        'gols_xg': df_comparacao_final['Gols'].max(),
                        'xg': df_comparacao_final['Gols Esperados'].max(),
                        'gols_sem_penalti': df_comparacao_final['Gols com Bola Rolando'].max(),
                        'gols_por_finalizacao': df_comparacao_final['gols_por_finalizacao_calculado'].max(),
                        'finalizacoes_gol': df_comparacao_final['Finalizações no Gol'].max(),
                        'finalizacoes': df_comparacao_final['Finalizações'].max(),
                        'toques_area_ofensiva': df_comparacao_final['Toques na Bola na Grande Área Ofensiva'].max(),
                        'impedimentos': df_comparacao_final['Impedimentos'].min()
                    }
                    metricas_ofensivo = [
                        ("Gols", 'gols_xg', 'max', False, 'Gols'),
                        ("Gols com Bola Rolando", 'gols_sem_penalti', 'max', False, 'Gols com Bola Rolando'),
                        ("Gols Esperados", 'xg', 'max', False, 'Gols Esperados'),
                        ("Toques na bola na Área Ofensiva", 'toques_area_ofensiva', 'max', False, 'Toques na Bola na Grande Área Ofensiva'),
                        ("Gols por Finalização", 'gols_por_finalizacao', 'max', False, 'gols_por_finalizacao_calculado'),
                        ("Finalizações no Gol", 'finalizacoes_gol', 'max', False, 'Finalizações no Gol'),
                        ("Finalizações", 'finalizacoes', 'max', False, 'Finalizações'),
                        ("Impedimentos", 'impedimentos', 'min', False, 'Impedimentos')
                    ]
                    for label, chave, tipo, eh_porcentagem, col_name in metricas_ofensivo:
                        colunas = st.columns(len(todos_jogadores_comparacao))
                        for i, jogador in enumerate(todos_jogadores_comparacao):
                            with colunas[i]:
                                if label == "Gols":
                                    st.markdown(f"**{jogador}**")
                                
                                partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                valor = df_comparacao_final.loc[jogador, col_name]
                                
                                if col_name in METRICAS_PARA_MEDIA or chave == 'gols_por_finalizacao':
                                    formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                else:
                                    formato = f"{int(valor)}"
                                
                                delta, cor = calcular_diferenca_percentual(valor, melhores_ofensivo.get(chave, valor), tipo, partidas_jogador)
                                st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)
                    
                    st.markdown("---")

                    st.header("Análise Gráfica Comparativa")
                    
                    df_final_agg_grafico = df_comparacao_final.reset_index()
                    
                    mapa_nomes_metricas = {}
                    for col_original in [col for col in METRICAS_PARA_SOMAR if col in df_final_agg_grafico.columns]:
                        mapa_nomes_metricas[col_original] = f"{col_original}"
                    for col_media in [col for col in METRICAS_PARA_MEDIA if col in df_final_agg_grafico.columns]:
                        mapa_nomes_metricas[col_media] = f"{col_media}"
                    
                    opcoes_disponiveis_map = {k: v for k, v in mapa_nomes_metricas.items() if k in df_final_agg_grafico.columns}
                    opcoes_labels = sorted(list(opcoes_disponiveis_map.values()))
                    mapa_labels_para_cols = {v: k for k, v in opcoes_disponiveis_map.items()}

                    if not opcoes_labels:
                        st.error("Não foi possível processar nenhuma métrica para os gráficos.")
                    else:
                        st.subheader("Gráfico de Radar")
                        
                        metricas_radar_selecionadas_labels = st.multiselect(
                            "**Selecione as Métricas para o Gráfico (Mínimo 3):**",
                            options=opcoes_labels,
                            default=[],
                            placeholder="Selecione as métricas",
                            key="ind_radar_metrics"
                        )
                        
                        if len(metricas_radar_selecionadas_labels) >= 3:
                            metricas_radar_raw = [mapa_labels_para_cols[label] for label in metricas_radar_selecionadas_labels]
                            labels_radar = metricas_radar_selecionadas_labels

                            df_radar = df_final_agg_grafico[['Jogador'] + metricas_radar_raw].copy()
                            
                            for metrica in metricas_radar_raw:
                                df_radar[metrica] = pd.to_numeric(df_radar[metrica], errors='coerce').fillna(0)
                                min_val = df_radar[metrica].min()
                                max_val = df_radar[metrica].max()
                                if (max_val - min_val) > 0:
                                    df_radar[metrica] = (df_radar[metrica] - min_val) / (max_val - min_val)
                                else:
                                    df_radar[metrica] = 0.5

                            fig_radar = go.Figure()

                            for i, row in df_radar.iterrows():
                                jogador = row['Jogador']
                                valores = row[metricas_radar_raw].values
                                valores = list(valores) + [valores[0]]
                                labels_fechados = labels_radar + [labels_radar[0]]
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=valores,
                                    theta=labels_fechados,
                                    fill='toself',
                                    name=jogador,
                                    hovertemplate=f"<b>{jogador}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
                                    opacity=0.7 # Leve transparência para facilitar sobreposição
                                ))

                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=True, template="plotly_dark",
                                # title="Comparação de Perfil de Jogo (Métricas Normalizadas)"
                            )
                            st.plotly_chart(fig_radar, width='stretch')
                        else:
                            st.info("Selecione pelo menos 3 métricas para gerar o gráfico de radar.")
                        
                        st.markdown("---")

                        st.subheader("Gráfico de Dispersão")
                        col_x, col_y = st.columns(2)
                        with col_x:
                            eixo_x_label = st.selectbox(
                                "**Selecione a Métrica para o Eixo X:**",
                                options=opcoes_labels, index=None, placeholder="Selecione uma métrica",
                                key="ind_scatter_x"
                            )
                        with col_y:
                            eixo_y_label = st.selectbox(
                                "**Selecione a Métrica para o Eixo Y:**",
                                options=opcoes_labels, index=None, placeholder="Selecione uma métrica",
                                key="ind_scatter_y"
                            )
                        
                        if eixo_x_label and eixo_y_label:
                            eixo_x_col = mapa_labels_para_cols[eixo_x_label]
                            eixo_y_col = mapa_labels_para_cols[eixo_y_label]

                            df_final_agg_grafico[eixo_x_col] = pd.to_numeric(df_final_agg_grafico[eixo_x_col], errors='coerce').fillna(0)
                            df_final_agg_grafico[eixo_y_col] = pd.to_numeric(df_final_agg_grafico[eixo_y_col], errors='coerce').fillna(0)
                                
                            fig_scatter = px.scatter(
                                df_final_agg_grafico,
                                x=eixo_x_col, y=eixo_y_col,
                                color='Jogador', text='Jogador', template="plotly_dark",
                                labels={eixo_x_col: eixo_x_label, eixo_y_col: eixo_y_label},
                                hover_data={'Jogador': False, eixo_x_col: ':.2f', eixo_y_col: ':.2f'}
                            )
                            
                            # --- PADRONIZAÇÃO APLICADA ---
                            fig_scatter.update_traces(
                                textposition='top center', # Texto acima da bolinha
                                textfont_size=13,          # Tamanho 13
                                cliponaxis=False           # Não cortar texto na borda
                            )
                            # -----------------------------
                            st.plotly_chart(fig_scatter, width='stretch')
                        else:
                            st.info("Selecione as métricas para os eixos X e Y para gerar o gráfico de dispersão.")
                        
                        st.markdown("---")
                        
                        st.subheader("Gráfico de Barras")
                        metricas_bar_selecionadas_labels = st.multiselect(
                            "**Selecione a(s) Métrica(s) para Comparação:**",
                            options=opcoes_labels, default=[],
                            placeholder="Selecione a(s) métrica(s)",
                            key="ind_bar_metrics"
                        )

                        normalizar_por_partidas = st.checkbox(
                            "**Normalizar métricas por número de partidas**",
                            value=False,
                            key="ind_normalizar_partidas"
                        )

                        if metricas_bar_selecionadas_labels:
                            metricas_bar_selecionadas = [mapa_labels_para_cols[label] for label in metricas_bar_selecionadas_labels]
                            
                            df_bar = df_final_agg_grafico[['Jogador'] + metricas_bar_selecionadas].copy()
                            
                            if normalizar_por_partidas:
                                # Converte as colunas numéricas para float antes de modificar os valores
                                df_bar[metricas_bar_selecionadas] = df_bar[metricas_bar_selecionadas].astype(float)
                                # Obter número de partidas de cada jogador
                                partidas_por_jogador = df_final_agg_grafico[['Jogador', 'Partidas']].set_index('Jogador')['Partidas']
                                
                                # Normalizar cada métrica pelo número de partidas
                                for metrica in metricas_bar_selecionadas:
                                    for jogador in df_bar['Jogador']:
                                        partidas = partidas_por_jogador.get(jogador, 1)
                                        if partidas > 0:
                                            valor_original = df_bar.loc[df_bar['Jogador'] == jogador, metrica].values[0]
                                            df_bar.loc[df_bar['Jogador'] == jogador, metrica] = valor_original / partidas
                            
                            df_bar_melted = df_bar.melt(
                                id_vars=['Jogador'], value_vars=metricas_bar_selecionadas,
                                var_name='Metrica_col', value_name='Valor'
                            )
                            
                            df_bar_melted['Valor'] = pd.to_numeric(df_bar_melted['Valor'], errors='coerce').fillna(0)
                            df_bar_melted['Métrica'] = df_bar_melted['Metrica_col'].map(opcoes_disponiveis_map)
                            
                            # Atualizar título do gráfico se normalizado
                            titulo_grafico = "Comparativo de Métricas Selecionadas"
                            if normalizar_por_partidas:
                                titulo_grafico += " (por partida)"
                            
                            fig_bar = px.bar(
                                df_bar_melted,
                                x='Métrica', y='Valor',
                                color='Jogador', barmode='group',
                                template="plotly_dark",
                                # text='Valor' # <--- IMPORTANTE: Adicionar para exibir os números
                            )
                            
                            # --- PADRONIZAÇÃO APLICADA ---
                            fig_bar.update_traces(
                                textposition='outside',      # Texto fora da barra
                                textfont_size=13,            # Tamanho 13
                                # texttemplate='%{text:.2f}',  # Formatação com 2 casas decimais
                                cliponaxis=False             # Não cortar texto
                            )
                            fig_bar.update_layout(xaxis_title=None)
                            # -----------------------------
                            st.plotly_chart(fig_bar, width='stretch')
                        else:
                            st.info("Selecione pelo menos uma métrica para exibir o gráfico de barras.")

                    st.markdown("---")
                    
                    # Tabela detalhada por temporada
                    temporadas_para_analise = sorted(df_comparacao['Temporada'].unique())
                    if not df_comparacao.empty:
                        df_detalhado = df_comparacao.sort_values(['Temporada', 'Jogador'])
                        
                        for temporada in temporadas_para_analise:
                            df_temporada = df_detalhado[df_detalhado['Temporada'] == temporada]
                            
                            if not df_temporada.empty:
                                st.markdown(f"**Temporada: {temporada}**")
                                df_display = df_temporada.copy()
                                for col in df_display.columns:
                                    if df_display[col].dtype in ['float64', 'float32']:
                                        df_display[col] = df_display[col].round(4)
                                    elif df_display[col].dtype in ['int64', 'int32']:
                                        df_display[col] = df_display[col].astype(int)
                                st.dataframe(df_display, hide_index=True, width='stretch')
                                st.markdown("---")
                    else:
                        st.info("Nenhum dado disponível para as temporadas selecionadas.")

        with sub_tab_comparativa_pos:
            if not jogador_selecionado:
                st.info("Por favor, selecione um jogador na aba principal para iniciar as análises.")
            else:
                col_input, _ = st.columns([1, 3])
                with col_input:
                    num_posicoes = st.number_input(
                        "**Quantas Posições Deseja Comparar?**",
                        min_value=1,
                        max_value=4,
                        value=1,
                        key="num_posicoes_pos"
                    )

                larguras = [0.25] * num_posicoes
                if num_posicoes < 4:
                    larguras.extend([1.0 - sum(larguras)])
                colunas_posicoes = st.columns(larguras)
                
                posicoes_selecionadas_input = []
                posicoes_options = sorted(meu_dataframe['Posição'].unique())

                for i in range(num_posicoes):
                    with colunas_posicoes[i]:
                        posicao = st.selectbox(
                            f"**Posição {i+1}**",
                            posicoes_options,
                            index=None,
                            key=f"comp_posicao_{i+1}",
                            placeholder="Selecione a posição",
                            label_visibility="collapsed"
                        )
                        if posicao:
                            posicoes_selecionadas_input.append(posicao)

                st.markdown("---")

                if not posicoes_selecionadas_input:
                    st.info("Selecione pelo menos uma posição para comparação usando os seletores acima.")
                else:
                    posicoes_selecionadas = list(dict.fromkeys(posicoes_selecionadas_input))
                    entidades_para_comparar = [jogador_selecionado] + posicoes_selecionadas
                    colunas_filtros = st.columns(len(entidades_para_comparar))
                    filtros_por_entidade = {}
                    
                    # Filtro do Jogador
                    with colunas_filtros[0]:
                        st.markdown(f"**Filtros para {jogador_selecionado}**")
                        df_jogador_temp = meu_dataframe[meu_dataframe['Jogador'] == jogador_selecionado]
                        temporadas_jogador = sorted(df_jogador_temp['Temporada'].unique())
                        filtros_jogador_temporadas = st.multiselect(
                            "**Temporadas:**",
                            temporadas_jogador, default=[], key="pos_temp_jogador",
                            placeholder=f"Todas temporadas de {jogador_selecionado}"
                        )
                        competicoes_jogador = sorted(df_jogador_temp['Competição'].unique())
                        filtros_jogador_competicoes = st.multiselect(
                            "**Competições:**",
                            competicoes_jogador, default=[], key="pos_comp_jogador",
                            placeholder=f"Todas competições de {jogador_selecionado}"
                        )
                        filtros_por_entidade[jogador_selecionado] = {
                            'temporadas': filtros_jogador_temporadas,
                            'competicoes': filtros_jogador_competicoes
                        }
                    
                    # Filtros para cada Posição
                    for i, posicao in enumerate(posicoes_selecionadas):
                        with colunas_filtros[i+1]:
                            st.markdown(f"**Filtros para Top Jogadores ({posicao})**")
                            df_posicao_temp = meu_dataframe[meu_dataframe['Posição'] == posicao]
                            temporadas_posicao = sorted(df_posicao_temp['Temporada'].unique())
                            filtros_posicao_temporadas = st.multiselect(
                                "**Temporadas:**",
                                temporadas_posicao, default=[], key=f"pos_temp_{posicao}",
                                placeholder=f"Todas temporadas de {posicao}"
                            )
                            competicoes_posicao = sorted(df_posicao_temp['Competição'].unique())
                            filtros_posicao_competicoes = st.multiselect(
                                "**Competições:**",
                                competicoes_posicao, default=[], key=f"pos_comp_{posicao}",
                                placeholder=f"Todas competições de {posicao}"
                            )
                            filtros_por_entidade[posicao] = {
                                'temporadas': filtros_posicao_temporadas,
                                'competicoes': filtros_posicao_competicoes
                            }

                    st.divider()
                    
                    lista_series_agg = []
                    mapa_info_cards = {}
                    
                    filtros_jogador = filtros_por_entidade[jogador_selecionado]
                    df_jogador_data = meu_dataframe[meu_dataframe['Jogador'] == jogador_selecionado].copy()
                    if filtros_jogador['temporadas']:
                        df_jogador_data = df_jogador_data[df_jogador_data['Temporada'].isin(filtros_jogador['temporadas'])]
                    if filtros_jogador['competicoes']:
                        df_jogador_data = df_jogador_data[df_jogador_data['Competição'].isin(filtros_jogador['competicoes'])]

                    if not df_jogador_data.empty:
                        s_jogador_sum = df_jogador_data[METRICAS_PARA_SOMAR].sum()
                        s_jogador_avg = df_jogador_data[METRICAS_PARA_MEDIA].mean()
                        s_jogador = pd.concat([s_jogador_sum, s_jogador_avg])
                        s_jogador['Jogador'] = jogador_selecionado
                        lista_series_agg.append(s_jogador)
                        mapa_info_cards[jogador_selecionado] = {'tipo': 'jogador', 'dados': df_jogador_data}
                    else:
                        st.warning(f"Nenhum dado encontrado para {jogador_selecionado} com os filtros selecionados.")
                        
                    for posicao in posicoes_selecionadas:
                        filtros_posicao = filtros_por_entidade[posicao]
                        nome_metrica = f"Top Jogadores ({posicao})"
                        
                        df_posicao_data = meu_dataframe[meu_dataframe['Posição'] == posicao].copy()
                        if filtros_posicao['temporadas']:
                            df_posicao_data = df_posicao_data[df_posicao_data['Temporada'].isin(filtros_posicao['temporadas'])]
                        if filtros_posicao['competicoes']:
                            df_posicao_data = df_posicao_data[df_posicao_data['Competição'].isin(filtros_posicao['competicoes'])]

                        jogadores_por_partidas = df_posicao_data.groupby('Jogador')['Partidas'].sum().reset_index()
                        top_10_jogadores_nomes = jogadores_por_partidas.sort_values('Partidas', ascending=False).head(10)['Jogador'].tolist()
                        df_top_10_filtrado = df_posicao_data[df_posicao_data['Jogador'].isin(top_10_jogadores_nomes)]

                        if df_top_10_filtrado.empty:
                            st.warning(f"Nenhum dado encontrado para a posição {posicao} com os filtros selecionados.")
                            continue
                        
                        df_pos_sum_per_player = df_top_10_filtrado.groupby('Jogador')[METRICAS_PARA_SOMAR].sum()
                        df_pos_avg_per_player = df_top_10_filtrado.groupby('Jogador')[METRICAS_PARA_MEDIA].mean()
                        s_posicao_sum = df_pos_sum_per_player.mean()
                        s_posicao_avg = df_pos_avg_per_player.mean()
                        s_posicao = pd.concat([s_posicao_sum, s_posicao_avg])
                        s_posicao['Jogador'] = nome_metrica
                        lista_series_agg.append(s_posicao)
                        mapa_info_cards[nome_metrica] = {
                            'tipo': 'posicao', 
                            'dados': df_top_10_filtrado, 
                            'nome_base': posicao,
                            'num_jogadores': len(top_10_jogadores_nomes)
                        }

                    if len(lista_series_agg) < 2:
                        st.warning("Não foi possível gerar a comparação. Verifique seus filtros.")
                    else:
                        df_comparacao_final = pd.DataFrame(lista_series_agg).set_index('Jogador').fillna(0)
                        todos_jogadores_comparacao = df_comparacao_final.index.tolist()

                        colunas_info = st.columns(len(todos_jogadores_comparacao))
                        
                        for i, nome_entidade in enumerate(todos_jogadores_comparacao):
                            with colunas_info[i]:
                                info = mapa_info_cards[nome_entidade]
                                df_info = info['dados']
                                
                                if info['tipo'] == 'jogador':
                                    st.subheader(f"{nome_entidade} - {int(df_info['Idade'].max())} anos")
                                    st.markdown(f"**Posição(ões):** {df_info['Posição'].mode()[0]}")
                                    st.markdown(f"**Nacionalidade:** {df_info['Nacionalidade'].iloc[0]}")
                                    st.markdown(f"**Equipe(s):** {', '.join(df_info['Equipe'].unique())}")
                                    st.markdown(f"**Competição(ões):** {', '.join(df_info['Competição'].unique())}")
                                
                                elif info['tipo'] == 'posicao':
                                    st.subheader(f"{nome_entidade}")
                                    st.markdown(f"**Posição Comparada:** {info['nome_base']}")
                                    st.markdown(f"**Nº de Jogadores na Amostra:** {info['num_jogadores']}")
                                    st.markdown(f"**Média de Idade:** {df_info['Idade'].mean():.1f} anos")
                                    st.markdown(f"**Competição(ões):** {', '.join(df_info['Competição'].unique())}")

                        st.markdown("---")

                        # Seção de Participação
                        st.subheader("Participação e Minutos")
                        melhores_participacao = {
                            'partidas': df_comparacao_final['Partidas'].max(),
                            'titular': df_comparacao_final['Partidas como Titular'].max(),
                            'reserva': df_comparacao_final['Partidas como Substituto'].max(),
                            'minutos': df_comparacao_final['Minutos Jogados'].max(),
                            'min_partida': df_comparacao_final['Minutos Jogados / Partidas'].max(),
                            'min_partida_titular': df_comparacao_final['Minutos / Partidas como Titular'].max(),
                            'perc_minutos': df_comparacao_final['Percentual de Minutos Jogados na Temporada'].max()
                        }
                        metricas_participacao = [
                            ("Partidas", 'partidas', 'max', False, 'Partidas'),
                            ("Partidas como Titular", 'titular', 'max', False, 'Partidas como Titular'),
                            ("Partidas como Reserva", 'reserva', 'max', False, 'Partidas como Substituto'),
                            ("Minutos Jogados", 'minutos', 'max', False, 'Minutos Jogados'),
                            ("Minutos Jogados por Partida", 'min_partida', 'max', False, 'Minutos Jogados / Partidas'),
                            ("Minutos Jogados por Partida (Titular)", 'min_partida_titular', 'max', False, 'Minutos / Partidas como Titular'),
                            ("% de Minutos Jogados", 'perc_minutos', 'max', True, 'Percentual de Minutos Jogados na Temporada')
                        ]
                        for label, chave, tipo, eh_porcentagem, col_name in metricas_participacao:
                            colunas = st.columns(len(todos_jogadores_comparacao))
                            for i, jogador in enumerate(todos_jogadores_comparacao):
                                with colunas[i]:
                                    if label == "Partidas":
                                        st.markdown(f"**{jogador}**")
                                    
                                    partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                    valor = df_comparacao_final.loc[jogador, col_name]
                                    
                                    if col_name in METRICAS_PARA_MEDIA:
                                        formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                    else:
                                        formato = f"{int(valor)}"
                                    
                                    delta, cor = calcular_diferenca_percentual(valor, melhores_participacao[chave], tipo, partidas_jogador)
                                    st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                        st.markdown("---")

                        # Seção de Impacto na Equipe
                        st.subheader("Impacto na Equipe")
                        for entidade in todos_jogadores_comparacao:
                            g_a = df_comparacao_final.loc[entidade, 'Gols + Assistências']
                            g_e = df_comparacao_final.loc[entidade, 'Gols Marcados pela Equipe com o Jogador em Campo']
                            df_comparacao_final.loc[entidade, 'part_gols_calculado'] = (g_a / max(1, g_e)) * 100

                        melhores_impacto = {
                            'pontos': df_comparacao_final['Média de Pontos Ganhos pela Equipe em Jogos Participados pelo Jogador'].max(),
                            'participacao_gols': df_comparacao_final['part_gols_calculado'].max(),
                            'saldo_gols': df_comparacao_final['Saldo de Gols com o Jogador em Campo'].max(),
                            'saldo_esperado': df_comparacao_final['Saldo de Gols Esperados da Equipe com o Jogador em Campo'].max()
                        }
                        metricas_impacto = [
                            ("Média de Pontos por Jogo", 'pontos', 'max', False, 'Média de Pontos Ganhos pela Equipe em Jogos Participados pelo Jogador'),
                            ("% de Participação nos Gols da Equipe", 'participacao_gols', 'max', True, 'part_gols_calculado'),
                            ("Saldo de Gols", 'saldo_gols', 'max', False, 'Saldo de Gols com o Jogador em Campo'),
                            ("Saldo de Gols Esperados", 'saldo_esperado', 'max', False, 'Saldo de Gols Esperados da Equipe com o Jogador em Campo')
                        ]
                        for label, chave, tipo, eh_porcentagem, col_name in metricas_impacto:
                            colunas = st.columns(len(todos_jogadores_comparacao))
                            for i, jogador in enumerate(todos_jogadores_comparacao):
                                with colunas[i]:
                                    if label == "Média de Pontos por Jogo":
                                        st.markdown(f"**{jogador}**")
                                    
                                    partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                    valor = df_comparacao_final.loc[jogador, col_name]

                                    if col_name in METRICAS_PARA_MEDIA or chave == 'participacao_gols':
                                        formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                    else:
                                        formato = f"{int(valor)}"

                                    delta, cor = calcular_diferenca_percentual(valor, melhores_impacto[chave], tipo, partidas_jogador)
                                    st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                        st.markdown("---")

                        # Seção Defensiva
                        st.subheader("Setor Defensivo")
                        melhores_defensivo = {
                            'desarmes': df_comparacao_final['Desarmes'].max(),
                            'interceptacoes': df_comparacao_final['Interceptações'].max(),
                            'bolas_recuperadas': df_comparacao_final['Bolas Recuperadas'].max(),
                            'faltas_cometidas': df_comparacao_final['Faltas Cometidas'].min(),
                            'cartoes_amarelos': df_comparacao_final['Cartões Amarelos'].min(),
                            'cartoes_vermelhos': df_comparacao_final['Cartões Vermelhos'].min(),
                            'bloqueios': df_comparacao_final['Bloqueios'].max(),
                            'perc_desarmes_drible': df_comparacao_final['Percentual de Sucesso em Desarmes em Dribles Adversários'].max(),
                            'perc_aereas': df_comparacao_final['Percentual de Disputas Aéreas Vencidas'].max(),
                            'erros_finalizacao': df_comparacao_final['Erros que Resultaram em Finalizações Adversárias'].min()
                        }
                        metricas_defensivo = [
                            ("Desarmes", 'desarmes', 'max', False, 'Desarmes'),
                            ("Interceptações", 'interceptacoes', 'max', False, 'Interceptações'),
                            ("Bloqueios", 'bloqueios', 'max', False, 'Bloqueios'),
                            ("Bolas Recuperadas", 'bolas_recuperadas', 'max', False, 'Bolas Recuperadas'),
                            ("% de Sucesso de Desarmes em Dribles", 'perc_desarmes_drible', 'max', True, 'Percentual de Sucesso em Desarmes em Dribles Adversários'),
                            ("% de Disputas Aéreas Vencidas", 'perc_aereas', 'max', True, 'Percentual de Disputas Aéreas Vencidas'),
                            ("Erros para Finalizações Adversárias", 'erros_finalizacao', 'min', False, 'Erros que Resultaram em Finalizações Adversárias'),
                            ("Faltas Cometidas", 'faltas_cometidas', 'min', False, 'Faltas Cometidas'),
                            ("Cartões Amarelos", 'cartoes_amarelos', 'min', False, 'Cartões Amarelos'),
                            ("Cartões Vermelhos", 'cartoes_vermelhos', 'min', False, 'Cartões Vermelhos')
                        ]
                        for label, chave, tipo, eh_porcentagem, col_name in metricas_defensivo:
                            colunas = st.columns(len(todos_jogadores_comparacao))
                            for i, jogador in enumerate(todos_jogadores_comparacao):
                                with colunas[i]:
                                    if label == "Desarmes":
                                        st.markdown(f"**{jogador}**")
                                    
                                    partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                    valor = df_comparacao_final.loc[jogador, col_name]

                                    if col_name in METRICAS_PARA_MEDIA:
                                        formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                    else:
                                        formato = f"{int(valor)}"
                                    
                                    delta, cor = calcular_diferenca_percentual(valor, melhores_defensivo.get(chave, valor), tipo, partidas_jogador)
                                    st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                        st.markdown("---")

                        # Seção de Criação
                        st.subheader("Setor de Criação")
                        melhores_criacao = {
                            'assistencias_xa': df_comparacao_final['Assistências'].max(),
                            'xa': df_comparacao_final['Assistências Esperadas'].max(),
                            'passes_chave': df_comparacao_final['Passes Chave'].max(),
                            'passes_progressivos': df_comparacao_final['Passes Progressivos Efetuados'].max(),
                            'cruzamentos': df_comparacao_final['Cruzamentos'].max(),
                            'sca': df_comparacao_final['Ações que Resultaram em Finalização'].max(),
                            'gca': df_comparacao_final['Ações que Resultaram em Gol'].max(),
                            'perc_dribles': df_comparacao_final['Percentual de Dribles Bem-sucedidos'].max(),
                            'passes_terco_final': df_comparacao_final['Passes Completos no Terço Final do Campo'].max(),
                            'faltas_sofridas': df_comparacao_final['Faltas Sofridas'].max(),
                            'perda_posse': df_comparacao_final['Perda de Posse de Bola'].min()
                        }
                        metricas_criacao = [
                            ("Assistências", 'assistencias_xa', 'max', False, 'Assistências'),
                            ("Assistências Esperadas", 'xa', 'max', False, 'Assistências Esperadas'),
                            ("Passes Chave", 'passes_chave', 'max', False, 'Passes Chave'),
                            ("Passes Progressivos", 'passes_progressivos', 'max', False, 'Passes Progressivos Efetuados'),
                            ("Ações Resultantes em Finalização", 'sca', 'max', False, 'Ações que Resultaram em Finalização'),
                            ("Ações Resultantes em Gol", 'gca', 'max', False, 'Ações que Resultaram em Gol'),
                            ("% de Dribles Bem-sucedidos", 'perc_dribles', 'max', True, 'Percentual de Dribles Bem-sucedidos'),
                            ("Passes Certos no Terço Final", 'passes_terco_final', 'max', False, 'Passes Completos no Terço Final do Campo'),
                            ("Faltas Sofridas", 'faltas_sofridas', 'max', False, 'Faltas Sofridas'),
                            ("Cruzamentos", 'cruzamentos', 'max', False, 'Cruzamentos'),
                            ("Perdas de Posse de Bola", 'perda_posse', 'min', False, 'Perda de Posse de Bola')
                        ]
                        for label, chave, tipo, eh_porcentagem, col_name in metricas_criacao:
                            colunas = st.columns(len(todos_jogadores_comparacao))
                            for i, jogador in enumerate(todos_jogadores_comparacao):
                                with colunas[i]:
                                    if label == "Assistências":
                                        st.markdown(f"**{jogador}**")
                                    
                                    partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                    valor = df_comparacao_final.loc[jogador, col_name]

                                    if col_name in METRICAS_PARA_MEDIA:
                                        formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                    else:
                                        formato = f"{int(valor)}"
                                    
                                    delta, cor = calcular_diferenca_percentual(valor, melhores_criacao.get(chave, valor), tipo, partidas_jogador)
                                    st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                        st.markdown("---")

                        # Seção Ofensiva
                        st.subheader("Setor Ofensivo")
                        for entidade in todos_jogadores_comparacao:
                            gols = df_comparacao_final.loc[entidade, 'Gols']
                            finalizacoes = df_comparacao_final.loc[entidade, 'Finalizações']
                            df_comparacao_final.loc[entidade, 'gols_por_finalizacao_calculado'] = gols / max(1, finalizacoes)

                        melhores_ofensivo = {
                            'gols_xg': df_comparacao_final['Gols'].max(),
                            'xg': df_comparacao_final['Gols Esperados'].max(),
                            'gols_sem_penalti': df_comparacao_final['Gols com Bola Rolando'].max(),
                            'gols_por_finalizacao': df_comparacao_final['gols_por_finalizacao_calculado'].max(),
                            'finalizacoes_gol': df_comparacao_final['Finalizações no Gol'].max(),
                            'finalizacoes': df_comparacao_final['Finalizações'].max(),
                            'toques_area_ofensiva': df_comparacao_final['Toques na Bola na Grande Área Ofensiva'].max(),
                            'impedimentos': df_comparacao_final['Impedimentos'].min()
                        }
                        metricas_ofensivo = [
                            ("Gols", 'gols_xg', 'max', False, 'Gols'),
                            ("Gols com Bola Rolando", 'gols_sem_penalti', 'max', False, 'Gols com Bola Rolando'),
                            ("Gols Esperados", 'xg', 'max', False, 'Gols Esperados'),
                            ("Toques na bola na Área Ofensiva", 'toques_area_ofensiva', 'max', False, 'Toques na Bola na Grande Área Ofensiva'),
                            ("Gols por Finalização", 'gols_por_finalizacao', 'max', False, 'gols_por_finalizacao_calculado'),
                            ("Finalizações no Gol", 'finalizacoes_gol', 'max', False, 'Finalizações no Gol'),
                            ("Finalizações", 'finalizacoes', 'max', False, 'Finalizações'),
                            ("Impedimentos", 'impedimentos', 'min', False, 'Impedimentos')
                        ]
                        for label, chave, tipo, eh_porcentagem, col_name in metricas_ofensivo:
                            colunas = st.columns(len(todos_jogadores_comparacao))
                            for i, jogador in enumerate(todos_jogadores_comparacao):
                                with colunas[i]:
                                    if label == "Gols":
                                        st.markdown(f"**{jogador}**")
                                    
                                    partidas_jogador = int(df_comparacao_final.loc[jogador, 'Partidas'])
                                    valor = df_comparacao_final.loc[jogador, col_name]

                                    if col_name in METRICAS_PARA_MEDIA or chave == 'gols_por_finalizacao':
                                        formato = f"{valor:.1f}%" if eh_porcentagem else f"{valor:.2f}"
                                    else:
                                        formato = f"{int(valor)}"
                                    
                                    delta, cor = calcular_diferenca_percentual(valor, melhores_ofensivo.get(chave, valor), tipo, partidas_jogador)
                                    st.metric(f"**{label}**", formato, delta=delta, delta_color=cor)

                        st.markdown("---")
                        
                        st.header("Análise Gráfica Comparativa")
                        
                        df_final_agg = df_comparacao_final.reset_index()
                        
                        mapa_nomes_metricas = {}
                        for col_original in [col for col in METRICAS_PARA_SOMAR if col in df_final_agg.columns]:
                            mapa_nomes_metricas[col_original] = f"{col_original}"
                        for col_media in [col for col in METRICAS_PARA_MEDIA if col in df_final_agg.columns]:
                            mapa_nomes_metricas[col_media] = f"{col_media}"
                        
                        opcoes_disponiveis_map = {k: v for k, v in mapa_nomes_metricas.items() if k in df_final_agg.columns}
                        opcoes_labels = sorted(list(opcoes_disponiveis_map.values()))
                        mapa_labels_para_cols = {v: k for k, v in opcoes_disponiveis_map.items()}

                        if not opcoes_labels:
                            st.error("Não foi possível processar nenhuma métrica para os gráficos.")
                        else:
                            st.subheader("Gráfico de Radar")
                            
                            metricas_radar_selecionadas_labels = st.multiselect(
                                "**Selecione as Métricas para o Gráfico (Mínimo 3):**",
                                options=opcoes_labels,
                                default=[],
                                placeholder="Selecione as métricas",
                                key="pos_radar_metrics"
                            )
                            
                            if len(metricas_radar_selecionadas_labels) >= 3:
                                metricas_radar_raw = [mapa_labels_para_cols[label] for label in metricas_radar_selecionadas_labels]
                                labels_radar = metricas_radar_selecionadas_labels

                                df_radar = df_final_agg[['Jogador'] + metricas_radar_raw].copy()
                                
                                for metrica in metricas_radar_raw:
                                    df_radar[metrica] = pd.to_numeric(df_radar[metrica], errors='coerce').fillna(0)
                                    min_val = df_radar[metrica].min()
                                    max_val = df_radar[metrica].max()
                                    if (max_val - min_val) > 0:
                                        df_radar[metrica] = (df_radar[metrica] - min_val) / (max_val - min_val)
                                    else:
                                        df_radar[metrica] = 0.5 

                                fig_radar = go.Figure()

                                for i, row in df_radar.iterrows():
                                    jogador = row['Jogador']
                                    valores = row[metricas_radar_raw].values
                                    valores = list(valores) + [valores[0]]
                                    labels_fechados = labels_radar + [labels_radar[0]]
                                    
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=valores,
                                        theta=labels_fechados,
                                        fill='toself',
                                        name=jogador,
                                        hovertemplate=f"<b>{jogador}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
                                        opacity=0.7 # Leve transparência para facilitar sobreposição
                                    ))

                                fig_radar.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                    showlegend=True, template="plotly_dark",
                                    # title="Comparação de Perfil de Jogo (Métricas Normalizadas 0-1)"
                                )
                                st.plotly_chart(fig_radar, width='stretch')
                            else:
                                st.info("Selecione pelo menos 3 métricas para gerar o gráfico de radar.")
                            
                            st.markdown("---")

                            st.subheader("Gráfico de Dispersão")
                            col_x, col_y = st.columns(2)
                            with col_x:
                                eixo_x_label = st.selectbox(
                                    "**Selecione a Métrica para o Eixo X:**",
                                    options=opcoes_labels, index=None, placeholder="Selecione uma métrica",
                                    key="pos_scatter_x"
                                )
                            with col_y:
                                eixo_y_label = st.selectbox(
                                    "**Selecione a Métrica para o Eixo Y:**",
                                    options=opcoes_labels, index=None, placeholder="Selecione uma métrica",
                                    key="pos_scatter_y"
                                )
                            
                            if eixo_x_label and eixo_y_label:
                                eixo_x_col = mapa_labels_para_cols[eixo_x_label]
                                eixo_y_col = mapa_labels_para_cols[eixo_y_label]

                                df_final_agg[eixo_x_col] = pd.to_numeric(df_final_agg[eixo_x_col], errors='coerce').fillna(0)
                                df_final_agg[eixo_y_col] = pd.to_numeric(df_final_agg[eixo_y_col], errors='coerce').fillna(0)
                                    
                                fig_scatter = px.scatter(
                                    df_final_agg,
                                    x=eixo_x_col, y=eixo_y_col,
                                    color='Jogador', text='Jogador', template="plotly_dark",
                                    labels={eixo_x_col: eixo_x_label, eixo_y_col: eixo_y_label},
                                    hover_data={'Jogador': False, eixo_x_col: ':.2f', eixo_y_col: ':.2f'}
                                )
                                
                                # --- PADRONIZAÇÃO APLICADA ---
                                fig_scatter.update_traces(
                                    textposition='top center',
                                    textfont_size=13, # Tamanho padronizado
                                    cliponaxis=False
                                )
                                # -----------------------------
                                st.plotly_chart(fig_scatter, width='stretch')
                            else:
                                st.info("Selecione as métricas para os eixos X e Y para gerar o gráfico de dispersão.")
                            
                            st.markdown("---")
                            
                            st.subheader("Gráfico de Barras")
                            metricas_bar_selecionadas_labels = st.multiselect(
                                "**Selecione a(s) Métrica(s) para Comparação:**",
                                options=opcoes_labels, default=[],
                                placeholder="Selecione a(s) métrica(s)",
                                key="pos_bar_metrics"
                            )

                            normalizar_por_partidas = st.checkbox(
                                "**Normalizar métricas por número de partidas**",
                                value=False,
                                key="pos_normalizar_partidas"
                            )

                            if metricas_bar_selecionadas_labels:
                                metricas_bar_selecionadas = [mapa_labels_para_cols[label] for label in metricas_bar_selecionadas_labels]
                                
                                df_bar = df_final_agg[['Jogador'] + metricas_bar_selecionadas].copy()
                                
                                if normalizar_por_partidas:
                                    # Converte as colunas numéricas para float antes de modificar os valores
                                    df_bar[metricas_bar_selecionadas] = df_bar[metricas_bar_selecionadas].astype(float)
                                    # Obter número de partidas de cada entidade
                                    partidas_por_entidade = df_final_agg[['Jogador', 'Partidas']].set_index('Jogador')['Partidas']
                                    
                                    # Normalizar cada métrica pelo número de partidas
                                    for metrica in metricas_bar_selecionadas:
                                        for entidade in df_bar['Jogador']:
                                            partidas = partidas_por_entidade.get(entidade, 1)
                                            if partidas > 0:
                                                valor_original = df_bar.loc[df_bar['Jogador'] == entidade, metrica].values[0]
                                                df_bar.loc[df_bar['Jogador'] == entidade, metrica] = valor_original / partidas
                                
                                df_bar_melted = df_bar.melt(
                                    id_vars=['Jogador'], value_vars=metricas_bar_selecionadas,
                                    var_name='Metrica_col', value_name='Valor'
                                )
                                
                                df_bar_melted['Valor'] = pd.to_numeric(df_bar_melted['Valor'], errors='coerce').fillna(0)
                                df_bar_melted['Métrica'] = df_bar_melted['Metrica_col'].map(opcoes_disponiveis_map)
                                
                                # Atualizar título do gráfico se normalizado
                                titulo_grafico = "Comparativo de Métricas Selecionadas"
                                if normalizar_por_partidas:
                                    titulo_grafico += " (por partida)"
                                
                                fig_bar = px.bar(
                                    df_bar_melted,
                                    x='Métrica', y='Valor',
                                    color='Jogador', barmode='group',
                                    template="plotly_dark",
                                    # text='Valor' # <--- ADICIONADO: Exibir valores
                                )
                                
                                # --- PADRONIZAÇÃO APLICADA ---
                                fig_bar.update_traces(
                                    textposition='outside',
                                    textfont_size=13,            # Tamanho padronizado
                                    # texttemplate='%{text:.2f}',  # Formatação limpa
                                    cliponaxis=False
                                )
                                fig_bar.update_layout(xaxis_title=None)
                                # -----------------------------
                                st.plotly_chart(fig_bar, width='stretch')
                            else:
                                st.info("Selecione pelo menos uma métrica para exibir o gráfico de barras.")

                        st.markdown("---")

                        # Identifica todas as temporadas presentes nos dados do jogador e das posições
                        todas_temporadas = set()
                        
                        # Temporadas do Jogador
                        if 'df_jogador_data' in locals() and not df_jogador_data.empty:
                            todas_temporadas.update(df_jogador_data['Temporada'].unique())
                        
                        # Temporadas das Posições Selecionadas
                        if 'posicoes_selecionadas' in locals():
                            for pos in posicoes_selecionadas:
                                if pos in mapa_info_cards:
                                    todas_temporadas.update(mapa_info_cards[pos]['dados']['Temporada'].unique())
                        
                        temporadas_ordenadas = sorted(list(todas_temporadas))
                        
                        # Define as colunas numéricas disponíveis
                        cols_soma = [c for c in METRICAS_PARA_SOMAR if c in meu_dataframe.columns]
                        cols_media = [c for c in METRICAS_PARA_MEDIA if c in meu_dataframe.columns]

                        if not temporadas_ordenadas:
                            st.info("Nenhum dado disponível para as temporadas selecionadas.")
                        else:
                            for temporada in temporadas_ordenadas:
                                linhas_tabela = []
                                
                                if 'df_jogador_data' in locals() and not df_jogador_data.empty:
                                    df_jog_temp = df_jogador_data[df_jogador_data['Temporada'] == temporada]
                                    if not df_jog_temp.empty:
                                        s_soma = df_jog_temp.groupby('Jogador')[cols_soma].sum()
                                        s_media = df_jog_temp.groupby('Jogador')[cols_media].mean()
                                        
                                        df_resumo_jog = pd.concat([s_soma, s_media], axis=1).reset_index()
                                        df_resumo_jog['Jogador'] = jogador_selecionado
                                        linhas_tabela.append(df_resumo_jog)

                                if 'posicoes_selecionadas' in locals():
                                    for pos in posicoes_selecionadas:
                                        # Garante que a chave correta foi buscada no dicionário
                                        nome_chave_posicao = f"Top Jogadores ({pos})"
                                        
                                        if nome_chave_posicao in mapa_info_cards:
                                            df_pos_raw = mapa_info_cards[nome_chave_posicao]['dados']
                                            df_pos_temp = df_pos_raw[df_pos_raw['Temporada'] == temporada]
                                            
                                            if not df_pos_temp.empty:
                                                # Agrupa por jogador
                                                grp_jogadores = df_pos_temp.groupby('Jogador')
                                                
                                                # Validação de colunas para evitar erros
                                                cols_soma_validas = [c for c in cols_soma if c in df_pos_temp.columns]
                                                cols_media_validas = [c for c in cols_media if c in df_pos_temp.columns]
                                                
                                                # Totais individuais
                                                df_players_sum = grp_jogadores[cols_soma_validas].sum()
                                                df_players_avg = grp_jogadores[cols_media_validas].mean()
                                                df_players_combined = pd.concat([df_players_sum, df_players_avg], axis=1)
                                                
                                                # Média do grupo (Posição)
                                                s_media_posicao = df_players_combined.mean().to_frame().T
                                                s_media_posicao['Jogador'] = pos
                                                
                                                linhas_tabela.append(s_media_posicao)

                                if linhas_tabela:
                                    st.markdown(f"**Temporada: {temporada}**")
                                    
                                    # Concatena as linhas criadas
                                    df_tabela_final = pd.concat(linhas_tabela, ignore_index=True)
                                    
                                    # Reordena as colunas para deixar o jogador em primeiro
                                    cols_exibicao = ['Jogador'] + [c for c in df_tabela_final.columns if c != 'Jogador']
                                    df_tabela_final = df_tabela_final[cols_exibicao]
                                    
                                    # Formatação para exibição limpa
                                    for col in df_tabela_final.columns:
                                        if pd.api.types.is_float_dtype(df_tabela_final[col]):
                                            df_tabela_final[col] = df_tabela_final[col].round(2)
                                            
                                    st.dataframe(df_tabela_final, hide_index=True, width='stretch')
                                    st.markdown("---")

        with sub_tab_temporal:
            if not jogador_selecionado:
                st.info("Por favor, selecione um jogador na aba principal para iniciar as análises.")
            else:
                df_jogador = meu_dataframe[meu_dataframe['Jogador'] == jogador_selecionado].copy()

                if df_jogador.empty:
                    st.warning("Não há dados temporais para este jogador.")
                else:
                    # Preparação dos dados
                    df_temporal_sum = df_jogador.groupby('Temporada')[METRICAS_PARA_SOMAR].sum(numeric_only=True)
                    df_temporal_avg = df_jogador.groupby('Temporada')[METRICAS_PARA_MEDIA].mean(numeric_only=True)
                    
                    df_temporal_agg = pd.concat([df_temporal_sum, df_temporal_avg], axis=1).reset_index()
                    
                    if df_temporal_agg.empty:
                        st.warning("Não foi possível agregar dados temporais para este jogador.")
                    else:
                        # --- CABEÇALHO UNIFICADO ---
                        st.header(f"Comparativo de Métrica(s) do {jogador_selecionado} por Temporada")

                        todas_metricas_disponiveis = sorted(METRICAS_PARA_SOMAR + METRICAS_PARA_MEDIA)
                        
                        # Layout de Controles
                        col_metrics, col_type = st.columns([3, 1])
                        
                        with col_metrics:
                            metricas_selecionadas = st.multiselect(
                                "**Selecione a(s) Métrica(s):**",
                                options=todas_metricas_disponiveis,
                                default=[],
                                key="temporal_metricas_unificadas",
                                placeholder="Selecione a(s) métrica(s)"
                            )
                        
                        with col_type:
                            tipo_visualizacao = st.radio(
                                "**Modo de Visualização:**",
                                options=["Linhas", "Barras"],
                                index=0,
                                key="temporal_tipo_vis",
                                horizontal=True
                            )

                        # Checkbox de Normalização (aplica para ambos)
                        normalizar_por_partidas = st.checkbox(
                            "**Normalizar métricas por número de partidas**",
                            value=False,
                            key="temporal_normalizar_global"
                        )

                        # st.markdown("---")

                        if metricas_selecionadas:
                            # Processamento dos dados (Comum para ambos os gráficos)
                            df_temporal_agg['Temporada'] = df_temporal_agg['Temporada'].astype(str)
                            
                            # Calcular partidas para normalização
                            partidas_por_temporada = df_jogador.groupby('Temporada')['Partidas'].sum().reset_index()
                            partidas_por_temporada_dict = partidas_por_temporada.set_index('Temporada')['Partidas'].to_dict()

                            df_melted = df_temporal_agg.melt(
                                id_vars='Temporada',
                                value_vars=metricas_selecionadas,
                                var_name='Métrica',
                                value_name='Valor'
                            )

                            df_melted['Valor'] = df_melted['Valor'].astype(float)

                            # Lógica de Normalização
                            if normalizar_por_partidas:
                                for index, row in df_melted.iterrows():
                                    temporada = row['Temporada']
                                    partidas = partidas_por_temporada_dict.get(temporada, 1) # Fallback para 1 para evitar div/0
                                    # Se a chave for string no dict, mas int no row, ou vice-versa, pode dar erro. 
                                    # Como convertemos df_temporal_agg['Temporada'] para str acima, garantimos aqui:
                                    if isinstance(list(partidas_por_temporada_dict.keys())[0], int):
                                        try:
                                            partidas = partidas_por_temporada_dict.get(int(temporada), 1)
                                        except:
                                            partidas = 1
                                    
                                    if partidas > 0:
                                        df_melted.at[index, 'Valor'] = row['Valor'] / partidas

                            # Definição do Título do Eixo Y
                            ylabel = "Valor"
                            if normalizar_por_partidas:
                                ylabel = "Média por Partida"

                            # --- RENDERIZAÇÃO DOS GRÁFICOS ---
                            if "Linha" in tipo_visualizacao: # Verifique se no seu código está "Linha" ou "Linhas"
                                fig = px.line(
                                    df_melted,
                                    x='Temporada',
                                    y='Valor',
                                    color='Métrica',
                                    markers=True,
                                    template="plotly_dark"
                                )
                                
                                # --- PADRONIZAÇÃO APLICADA ---
                                fig.update_traces(
                                    cliponaxis=False, # Evita cortar marcadores nas bordas
                                    marker_size=8     # Garante marcadores visíveis
                                )
                                # -----------------------------
                                
                                fig.update_layout(yaxis_title=ylabel, xaxis_title="Temporada")
                                st.plotly_chart(fig, width='stretch')

                            else: # Modo Barras
                                fig = px.bar(
                                    df_melted,
                                    x='Temporada',
                                    y='Valor',
                                    color='Métrica',
                                    barmode='group',
                                    template="plotly_dark",
                                    # text='Valor' # <--- IMPORTANTE: Adicionar para exibir os números
                                )
                                
                                # --- PADRONIZAÇÃO APLICADA ---
                                fig.update_traces(
                                    textposition='outside',
                                    textfont_size=13,
                                    # texttemplate='%{text:.2f}',
                                    cliponaxis=False
                                )
                                # -----------------------------
                                fig.update_layout(yaxis_title=ylabel, xaxis_title="Temporada")
                                st.plotly_chart(fig, width='stretch')
                        
                        else:
                            st.info("Selecione uma ou mais métricas acima para visualizar o gráfico.")

                        st.divider()

                        st.subheader("Dados Brutos")
                        st.dataframe(
                            df_jogador.sort_values(by=['Temporada', 'Competição'], ascending=False),
                            width='stretch',
                            hide_index=True
                        )
            
        with sub_tab_rankings:
            # ==========================================
            # SEÇÃO 1: RANKING DE PERFORMANCE
            # ==========================================
            st.header("Ranking de Performance")
            
            # Linha 1: Métrica e Minutagem
            col_rank_metrica, col_rank_minutos = st.columns([2, 1])
            
            todas_metricas = sorted(METRICAS_PARA_MEDIA + METRICAS_PARA_SOMAR)
            
            with col_rank_metrica:
                metrica_ranking = st.selectbox(
                    "**Selecione a Métrica para o Ranking:**",
                    options=todas_metricas,
                    index=None,
                    placeholder="Selecione uma métrica",
                    key="rank_ind_metrica"
                )
            
            with col_rank_minutos:
                minutos_minimos = st.number_input(
                    "**Mínimo de Minutos:**",
                    min_value=0,
                    value=10000,
                    step=50,
                    help="Filtra jogadores com quantidade de minutos jogados acima da quantidade informada.",
                    key="rank_ind_minutos"
                )

            # Linha 2: Filtros de Contexto
            col_rank_temp, col_rank_comp, col_rank_pos = st.columns(3)
                
            with col_rank_temp:
                ranking_temporadas = st.multiselect(
                    "**Temporadas:**",
                    sorted(meu_dataframe['Temporada'].unique()),
                    default=[],
                    placeholder="Todas as temporadas",
                    key="rank_ind_temp"
                )

            with col_rank_comp:
                ranking_competicoes = st.multiselect(
                    "**Competições:**",
                    sorted(meu_dataframe['Competição'].unique()),
                    default=[],
                    placeholder="Todas as competições",
                    key="rank_ind_comp"
                )

            with col_rank_pos:
                ranking_posicoes = st.multiselect(
                    "**Posições:**",
                    sorted(meu_dataframe['Posição'].unique()),
                    default=[],
                    placeholder="Todas as posições",
                    key="rank_ind_pos"
                )

            if metrica_ranking:
                analisador_rank = AnalisadorFutebol(meu_dataframe)
                
                df_rank, erro_rank = analisador_rank.gerar_ranking(
                    metrica=metrica_ranking,
                    lista_media=METRICAS_PARA_MEDIA,
                    lista_soma=METRICAS_PARA_SOMAR,
                    min_minutos=minutos_minimos,
                    temporadas=ranking_temporadas if ranking_temporadas else None,
                    competicoes=ranking_competicoes if ranking_competicoes else None,
                    posicoes=ranking_posicoes if ranking_posicoes else None
                )
                
                if erro_rank:
                    st.warning(erro_rank)
                else:
                    top_n = 10
                    df_top_10 = df_rank.head(top_n).copy()
                    
                    jogador_row = df_rank[df_rank['Jogador'] == jogador_selecionado]
                    
                    col_tabela, col_grafico = st.columns([1.5, 2])
                    
                    with col_tabela:
                        st.subheader(f"Top {top_n} - {metrica_ranking}")
                        st.dataframe(
                            df_top_10[['Ranking', 'Jogador', metrica_ranking, 'Minutos Jogados']],
                            hide_index=True,
                            width='stretch'
                        )
                        
                        if not jogador_row.empty:
                            posicao_jogador_num = jogador_row['Ranking'].values[0]
                            
                            if posicao_jogador_num > top_n:
                                st.info(f"ℹ️ **{jogador_selecionado}** ocupa a **{posicao_jogador_num}ª posição** neste ranking.")
                                st.dataframe(
                                    jogador_row[['Ranking', 'Jogador', metrica_ranking, 'Minutos Jogados']],
                                    hide_index=True,
                                    width='stretch'
                                )
                        else:
                            st.warning(f"{jogador_selecionado} não atende aos critérios de minutos ({minutos_minimos} min).")

                    with col_grafico:
                        df_grafico = df_top_10.copy()
                        
                        if not jogador_row.empty:
                            posicao_jogador_num = jogador_row['Ranking'].values[0]
                            if posicao_jogador_num > top_n:
                                df_grafico = pd.concat([df_grafico, jogador_row])
                        
                        cores_finais = []
                        for jogador in df_grafico['Jogador']:
                            if jogador == jogador_selecionado:
                                cores_finais.append('#22c912')
                            else:
                                cores_finais.append('#e4f2da')

                        fig_rank = px.bar(
                            df_grafico,
                            x=metrica_ranking,
                            y='Jogador',
                            orientation='h',
                            text=metrica_ranking,
                            template="plotly_dark"
                        )
                        
                        # --- PADRONIZAÇÃO APLICADA ---
                        fig_rank.update_traces(
                            marker_color=cores_finais, # Mantém sua lógica de cores
                            textposition='outside',
                            textfont_size=13, # Tamanho padronizado
                            cliponaxis=False
                        )
                        # -----------------------------
                        fig_rank.update_layout(yaxis=dict(autorange="reversed"))
                        fig_rank.update_yaxes(categoryorder='trace')
                        st.plotly_chart(fig_rank, width='stretch')
            else:
                st.info("Selecione uma métrica para gerar o ranking de performance.")

            st.markdown("---")

            # ==========================================
            # SEÇÃO 2: SIMILARIDADE (AUTOMÁTICA)
            # ==========================================
            st.header("Similaridade de Jogadores")
            # st.markdown("""
            # Esta ferramenta utiliza **Distância Euclidiana** com dados normalizados para encontrar jogadores estatisticamente semelhantes.
            # """)
            
            # st.subheader("Configuração da Busca")

            # st.markdown("---")
            # st.subheader("Métricas para Similaridade")
            
            metricas_similaridade = st.multiselect(
                "**Selecione as Métricas de Similaridade (Mínimo 3):**",
                options=todas_metricas,
                default=None,
                placeholder="Selecione a(s) métrica(s)",
                key="sim_metrics_input"
            )
            
            col_num, col_temp, col_comp, col_pos = st.columns([0.8, 2, 2, 2])
            
            with col_num:
                n_similares = st.number_input(
                    "**Top:**", 
                    min_value=1, max_value=20, value=5,
                    help="Quantos jogadores similares exibir",
                    key="sim_n_input"
                )
                
            with col_temp:
                univ_temporadas = st.multiselect(
                    "**Temporadas:**",
                    sorted(meu_dataframe['Temporada'].unique()),
                    default=[],
                    placeholder="Todas as temporadas",
                    key="sim_temp_input"
                )

            with col_comp:
                univ_competicoes = st.multiselect(
                    "**Competições:**",
                    sorted(meu_dataframe['Competição'].unique()),
                    default=[],
                    placeholder="Todas as competições",
                    key="sim_comp_input"
                )

            with col_pos:
                opcoes_posicao_simples = ['Defensor', 'Meio-campista', 'Atacante']
                posicao_jogador_raw = meu_dataframe[meu_dataframe['Jogador'] == jogador_selecionado]['Posição'].mode()[0]
                defaults_posicao = [p for p in opcoes_posicao_simples if p in posicao_jogador_raw]
                
                univ_posicoes = st.multiselect(
                    "**Posições:**",
                    options=opcoes_posicao_simples,
                    default=defaults_posicao,
                    placeholder="Todas as posições",
                    key="sim_pos_input"
                )

            # Lógica Automática: Se houver 3 ou mais métricas, executa.
            if not metricas_similaridade:
                st.info("Selecione pelo menos 3 métricas para iniciar a busca.")
            elif len(metricas_similaridade) < 3:
                st.warning(f"Selecione mais **{3 - len(metricas_similaridade)}** métrica(s).")
            else:
                # O cálculo inicia automaticamente aqui
                with st.spinner(f"Calculando similaridade para {jogador_selecionado}..."):
                    
                    df_analise = meu_dataframe.copy()
                    
                    if univ_posicoes:
                        filtro_regex = '|'.join(univ_posicoes)
                        df_analise = df_analise[df_analise['Posição'].str.contains(filtro_regex, na=False, regex=True)]
                    
                    analisador_sim = AnalisadorFutebol(df_analise)
                    
                    df_similares, erro = analisador_sim.calcular_jogadores_similares(
                        jogador_alvo=jogador_selecionado,
                        metricas=metricas_similaridade,
                        filtros_temporada=univ_temporadas if univ_temporadas else None,
                        filtros_competicao=univ_competicoes if univ_competicoes else None,
                        cols_media=METRICAS_PARA_MEDIA,
                        cols_soma=METRICAS_PARA_SOMAR,
                        n=n_similares
                    )
                    
                    if erro:
                        st.error(erro)
                    else:
                        # --- NOVO: Exibir o Jogador Selecionado (Referência) ---
                        # st.markdown(f"##### Referência: {jogador_selecionado}")
                        
                        # 1. Filtra os dados originais para o jogador selecionado
                        df_alvo_ref = meu_dataframe[meu_dataframe['Jogador'] == jogador_selecionado].copy()
                        
                        # 2. Aplica os mesmos filtros usados na busca (Temporada/Competição)
                        if univ_temporadas:
                            df_alvo_ref = df_alvo_ref[df_alvo_ref['Temporada'].isin(univ_temporadas)]
                        if univ_competicoes:
                            df_alvo_ref = df_alvo_ref[df_alvo_ref['Competição'].isin(univ_competicoes)]
                            
                        if not df_alvo_ref.empty:
                            # 3. Define se a métrica é de soma ou média
                            agg_dict_ref = {
                                m: 'mean' if m in METRICAS_PARA_MEDIA else 'sum' 
                                for m in metricas_similaridade
                            }
                            
                            # 4. Agrupa para ficar no mesmo formato (Index=Jogador, Colunas=Métricas)
                            df_alvo_agg = df_alvo_ref.groupby('Jogador')[metricas_similaridade].agg(agg_dict_ref)
                            
                            # 5. Adiciona a coluna de Distância (0.00 pois é ele mesmo) na primeira posição
                            df_alvo_agg.insert(0, 'Similaridade (Distância)', 0.0)
                            
                            # 6. Exibe o DataFrame
                            st.dataframe(
                                df_alvo_agg.style.format("{:.2f}"),
                                width='stretch'
                            )
                        else:
                            st.warning(f"Não há dados para {jogador_selecionado} nos filtros selecionados.")
                        
                        # st.markdown("---")
                        st.markdown("##### Jogadores Similares:")
                        # -------------------------------------------------------

                        st.dataframe(
                            df_similares.style.format("{:.2f}"),
                            width='stretch'
                        )
                        
                        st.caption("*Nota: Quanto menor a 'Distância', mais similar é o jogador.*")

                        st.subheader("Comparação Visual")
                        
                        if not df_similares.empty:
                            lista_opcoes_radar = df_similares.index.tolist()
                            
                            col_seletor_radar, col_grafico_radar = st.columns([1, 2])
                            
                            with col_seletor_radar:
                                # ALTERADO: De selectbox para multiselect
                                jogadores_radar_selecionados = st.multiselect(
                                    "**Comparar com:**",
                                    placeholder="Selecione o jogador",
                                    options=lista_opcoes_radar,
                                    default=[lista_opcoes_radar[0]] if lista_opcoes_radar else None,
                                    key="sim_radar_select"
                                )

                            dados_alvo_raw = analisador_sim.filtrar_dados(jogador_selecionado, temporadas=univ_temporadas)
                            
                            if not dados_alvo_raw.empty:
                                # 1. Prepara dados do Jogador Alvo
                                dados_alvo_agg = dados_alvo_raw.groupby('Jogador')[metricas_similaridade].agg(
                                    {m: 'mean' if m in METRICAS_PARA_MEDIA else 'sum' for m in metricas_similaridade}
                                ).iloc[0]
                                
                                # 2. Monta listas para o DataFrame combinado (Alvo + Selecionados)
                                indices = [jogador_selecionado]
                                dados_lista = [dados_alvo_agg]
                                
                                for jog_sim in jogadores_radar_selecionados:
                                    # Busca os dados já calculados no df_similares
                                    dados_sim = df_similares.loc[jog_sim, metricas_similaridade]
                                    indices.append(jog_sim)
                                    dados_lista.append(dados_sim)
                                
                                df_chart = pd.DataFrame(dados_lista, index=indices)
                                
                                # 3. Normalização (Baseada no máximo entre todos os selecionados)
                                df_chart_norm = df_chart.copy()
                                for col in df_chart_norm.columns:
                                    max_val = df_chart_norm[col].max()
                                    if max_val > 0:
                                        df_chart_norm[col] = df_chart_norm[col] / max_val
                                
                                fig_radar_sim = go.Figure()
                                
                                # 4. Adiciona traço do Jogador Alvo (Referência - Mantendo Azul Fixo)
                                fig_radar_sim.add_trace(go.Scatterpolar(
                                    r=df_chart_norm.loc[jogador_selecionado].values,
                                    theta=df_chart_norm.columns,
                                    fill='toself',
                                    name=f"{jogador_selecionado}",
                                    line_color='#e4f2da',
                                    opacity=0.7 # Leve transparência para facilitar sobreposição
                                ))
                                
                                # 5. Adiciona traços dos Jogadores Comparados (Cores automáticas)
                                for jog_sim in jogadores_radar_selecionados:
                                    fig_radar_sim.add_trace(go.Scatterpolar(
                                        r=df_chart_norm.loc[jog_sim].values,
                                        theta=df_chart_norm.columns,
                                        fill='toself',
                                        name=f"{jog_sim}",
                                        opacity=0.7 # Leve transparência para facilitar sobreposição
                                        # Sem line_color definido para que o Plotly varie as cores automaticamente
                                    ))
                                
                                fig_radar_sim.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                                    showlegend=True,
                                    template="plotly_dark",
                                    margin=dict(t=40, b=20, l=40, r=40)
                                )
                                
                                with col_grafico_radar:
                                    st.plotly_chart(fig_radar_sim, width='stretch')
                            else:
                                st.warning(f"Não foi possível gerar o gráfico: dados insuficientes para {jogador_selecionado} nas temporadas selecionadas.")

    else:
        st.info("Por favor, selecione um jogador para iniciar as análises.")


with tab_rankings_globais:
    st.header("Central de Rankings e Inteligência de Dados")
    
    subtab_top, subtab_revelacao, subtab_consistencia, subtab_idade = st.tabs([
        "**Melhores Jogadores**", "**Revelações da Temporada**", "**Jogadores Consistentes**", "**Curva de Idade**"
    ])

    # --------------------------------------------------------------------------
    # 1. TOP JOGADORES
    # --------------------------------------------------------------------------
    with subtab_top:
        st.subheader("Ranking de Melhores Jogadores")
        
        # Linha 1: Configurações Numéricas e Métrica
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            top_metrica = st.selectbox(
                "**Selecione a Métrica para o Ranking:**", 
                options=sorted(METRICAS_PARA_MEDIA + METRICAS_PARA_SOMAR),
                index=None,
                placeholder="Selecione uma métrica",
                key="top_met"
            )
        with c2:
            top_n = st.number_input("**Top:**", 5, 50, 10, key="top_n")
        with c3:
            top_min_min = st.number_input("**Mínimo de Minutos:**", 0, value=10000, step=50, key="top_min")

        # Linha 2: Filtros de Contexto (Temporada, Competição, Posição)
        c4, c5, c6 = st.columns(3)
        with c4:
            top_temp = st.multiselect(
                "**Temporadas:**",
                sorted(meu_dataframe['Temporada'].unique()),
                default=[],
                key="top_temp",
                placeholder="Todas as temporadas"
            )
        with c5:
            top_comp = st.multiselect(
                "**Competições:**",
                sorted(meu_dataframe['Competição'].unique()),
                default=[],
                key="top_comp",
                placeholder="Todas as competições"
            )
        with c6:
            top_pos = st.multiselect(
                "**Posições:**",
                sorted(meu_dataframe['Posição'].unique()),
                default=[],
                key="top_pos",
                placeholder="Todas as posições"
            )

        # st.markdown("---")

        if top_metrica:
            df_top, erro = analisador.gerar_ranking(
                metrica=top_metrica,
                lista_media=METRICAS_PARA_MEDIA,
                lista_soma=METRICAS_PARA_SOMAR,
                min_minutos=top_min_min,
                temporadas=top_temp if top_temp else None,
                competicoes=top_comp if top_comp else None,
                posicoes=top_pos if top_pos else None
            )
            
            if erro:
                st.warning(erro)
            else:
                st.subheader(f"Top {top_n} Jogadores - {top_metrica}")
                df_show = df_top.head(top_n)
                
                col_data, col_graph = st.columns([1.5, 2])
                with col_data:
                    st.dataframe(
                        df_show[['Ranking', 'Jogador', top_metrica, 'Minutos Jogados']],
                        hide_index=True,
                        width='stretch'
                    )

                with col_graph:
                    fig_top = px.bar(
                        df_show,
                        x=top_metrica, y='Jogador', orientation='h',
                        text=top_metrica, template="plotly_dark"
                    )
                    # PADRONIZAÇÃO AQUI: Tamanho 13 e cliponaxis=False
                    fig_top.update_traces(
                        marker_color='#e4f2da', 
                        textposition='outside', 
                        textfont_size=13, 
                        cliponaxis=False
                    )
                    # Inverte o eixo Y para que o Rank 1 (primeira linha do DF) fique no topo visualmente
                    fig_top.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_top, width='stretch')
        else:
            st.info("Selecione uma métrica acima para gerar o ranking.")

    # --------------------------------------------------------------------------
    # 2. REVELAÇÕES
    # --------------------------------------------------------------------------
    with subtab_revelacao:
        st.subheader("Ranking de Jogadores com Maior Evolução")
        st.markdown("Compara a performance de um jogador na temporada selecionada em relação à anterior.")
        
        all_seasons = sorted(meu_dataframe['Temporada'].unique())
        
        if len(all_seasons) < 2:
            st.error("É necessário ter pelo menos 2 temporadas na base de dados para calcular evolução.")
        else:
            # Linha 1: Configurações Numéricas e Métrica
            cr1, cr2, cr3 = st.columns([2, 1, 1])
            
            with cr1:
                rev_metrica = st.selectbox(
                    "**Selecione a Métrica para o Ranking:**", 
                    options=sorted(METRICAS_PARA_MEDIA + METRICAS_PARA_SOMAR),
                    index=None,
                    placeholder="Selecione uma métrica",
                    key="rev_met"
                )
            
            with cr2:
                rev_n = st.number_input("**Top:**", 5, 50, 10, key="rev_n")
                
            with cr3:
                rev_min_min = st.number_input("**Mínimo de Minutos:**", 0, value=2000, step=50, key="rev_min", help="Mínimo de minutos jogados em AMBAS as temporadas.")

            # Linha 2: Filtros de Contexto
            cr4, cr5, cr6 = st.columns(3)

            with cr4:
                opcoes_temporada = all_seasons[1:]
                rev_temp_selecionada = st.selectbox(
                    "**Temporadas:**", 
                    opcoes_temporada,
                    index=None,
                    placeholder="Todas as temporadas",
                    key="rev_season_ref",
                    help="A análise será feita comparando esta temporada com a imediatamente anterior."
                )

            with cr5:
                rev_competicao = st.multiselect(
                    "**Competições:**",
                    sorted(meu_dataframe['Competição'].unique()),
                    default=[],
                    placeholder="Todas as competições",
                    key="rev_comp"
                )

            with cr6:
                rev_posicao = st.multiselect(
                    "**Posições:**",
                    sorted(meu_dataframe['Posição'].unique()),
                    default=[],
                    placeholder="Todas as posições",
                    key="rev_pos"
                )
            
            if rev_metrica and rev_temp_selecionada:
                # Lógica da Temporada Anterior
                idx_atual = all_seasons.index(rev_temp_selecionada)
                temporada_passada = all_seasons[idx_atual - 1]
                temporada_atual = rev_temp_selecionada
                
                df_rev, erro = analisador.calcular_revelacoes(
                    metrica=rev_metrica,
                    temporada_anterior=temporada_passada,
                    temporada_atual=temporada_atual,
                    lista_media=METRICAS_PARA_MEDIA,
                    min_minutos=rev_min_min, # Passando o filtro de minutos
                    n=rev_n,                 # Passando o filtro de Top N
                    competicoes=rev_competicao if rev_competicao else None,
                    posicoes=rev_posicao if rev_posicao else None
                )
                
                if erro:
                    st.warning(erro)
                else:
                    st.subheader(f"Top {rev_n} Revelações - {rev_metrica}")
                    
                    # Prepara dados do gráfico (respeitando o N escolhido)
                    df_rev_graph = df_rev.head(rev_n).sort_values('Melhora', ascending=True)
                    
                    # Layout: Tabela (Esq) | Gráfico (Dir)
                    col_data, col_graph = st.columns([1.5, 2])
                    
                    with col_data:
                        st.dataframe(
                        df_rev.reset_index()[['index', 'Jogador', f'{rev_metrica}_ant', f'{rev_metrica}_atu', 'Melhora', "Minutos Jogados_atu"
                        ]].rename(columns={
                            'index': 'Ranking',
                            f'{rev_metrica}_ant': f'{rev_metrica} ({temporada_passada})',
                            f'{rev_metrica}_atu': f'{rev_metrica} ({temporada_atual})',
                            'Melhora': 'Evolução',
                            'Minutos Jogados_atu': 'Minutos Jogados'
                        }),
                        width='stretch',
                        hide_index=True
                    )

                    with col_graph:
                        df_rev_graph = df_rev.head(rev_n)

                        fig_rev = px.bar(
                            df_rev_graph,
                            x='Melhora', y='Jogador', orientation='h',
                            text='Melhora', template="plotly_dark",
                            # O label abaixo define o título do eixo X. 
                            # Você pode mudar para 'Evolução' se preferir, ou manter 'Diferença...'
                            labels={'Melhora': f'Evolução de {rev_metrica}'}
                        )
                        # PADRONIZAÇÃO APLICADA
                        fig_rev.update_traces(
                            marker_color='#e4f2da',
                            textposition='outside',
                            textfont_size=13, # Tamanho padronizado
                            cliponaxis=False # Evita cortar o texto
                        )

                        # Inverte o eixo Y para o primeiro do DF (Rank 1) ficar no topo
                        fig_rev.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_rev, width='stretch')

            else:
                st.info("Selecione a métrica e a temporada para visualizar as revelações.")

    # --------------------------------------------------------------------------
    # 3. CONSISTÊNCIA
    # --------------------------------------------------------------------------
    with subtab_consistencia:
        st.subheader("Ranking de Jogadores Mais Consistentes")
        st.markdown("Identifica regularidade de jogadores em um período contínuo.")
        
        # Obter todas as temporadas disponíveis ordenadas
        all_seasons = sorted(meu_dataframe['Temporada'].unique())
        total_seasons = len(all_seasons)

        # Linha 1: Configurações Numéricas e Métrica
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        
        with cc1:
            cons_metrica = st.selectbox(
                "**Selecione a Métrica para o Ranking:**", 
                options=sorted(METRICAS_PARA_MEDIA + METRICAS_PARA_SOMAR), 
                index=None,
                placeholder="Selecione uma métrica",
                key="cons_met"
            )
        with cc2:
            cons_n = st.number_input("**Top:**", 5, 50, 10, key="cons_n")
            
        with cc3:
            cons_min_min = st.number_input("**Mínimo de Minutos:**", 0, value=8000, step=50, key="cons_min_min", help="Mínimo de minutos jogados por temporada para ela ser contabilizada.")

        # Linha 2: Configuração do Período (Lógica de Validação) e Filtros Extras
        cc4, cc5, cc6, cc7 = st.columns([1.2, 1.2, 1.3, 1.3])
        
        with cc4:
            # 1. Seleciona o Tamanho da Sequência
            # O máximo não pode ser maior que o total de temporadas disponíveis
            cons_min_temp = st.number_input(
                "**Mínimo de Temporadas:**", 
                min_value=3,
                max_value=total_seasons,
                value=3,
                help="Quantas temporadas consecutivas o jogador precisa ter jogado para ser analisado.",
                key="cons_seq_len"
            )

        with cc5:
            # 2. Calcula as Temporadas de Início Válidas
            # Se min=3 e temos [2020, 21, 22, 23, 24], validos para inicio: 2020, 2021, 2022.
            # Se começar em 2023, só tem 2023 e 2024 (2 temps), o que viola o mínimo de 3.
            
            # Índice máximo permitido para o início = Total - Mínimo
            # Ex: 5 - 3 = 2. Índices 0, 1, 2 são válidos.
            max_start_index = total_seasons - cons_min_temp
            
            if max_start_index < 0:
                st.error("Configuração impossível: Mínimo maior que temporadas disponíveis.")
                valid_start_seasons = []
            else:
                # Slice da lista até o índice permitido (inclusive)
                valid_start_seasons = all_seasons[:max_start_index + 1]

            cons_start_season = st.selectbox(
                "**Início do Período:**",
                options=valid_start_seasons,
                index=None, # Começa vazio para forçar escolha consciente
                placeholder="Selecione o início",
                help=f"Com base no mínimo de {cons_min_temp} temporadas, apenas estes inícios garantem o período completo.",
                key="cons_start_season"
            )
            
        with cc6:
            cons_competicao = st.multiselect(
                "**Competições:**",
                sorted(meu_dataframe['Competição'].unique()),
                default=[],
                placeholder="Todas as competições",
                key="cons_comp"
            )

        with cc7:
            cons_posicao = st.multiselect(
                "**Posições:**",
                sorted(meu_dataframe['Posição'].unique()),
                default=[],
                placeholder="Todas as posições",
                key="cons_pos"
            )
        
        # st.markdown("---")
        
        if cons_metrica and cons_start_season:
            # Lógica para definir a lista de temporadas do período
            start_idx = all_seasons.index(cons_start_season)
            # O período vai do início selecionado até o final dos dados disponíveis
            # (Ou você poderia limitar a exatamente N temporadas, mas geralmente "a partir de" é mais útil)
            temporadas_periodo = all_seasons[start_idx:]
            
            st.caption(f"Analisando período: **{temporadas_periodo[0]}** até **{temporadas_periodo[-1]}**.")
            st.caption("Gráfico: Quanto mais à direita = Jogadores mais consistentes. Quanto mais ao topo = Jogadores com melhor média de performance.")

            df_cons, erro = analisador.calcular_consistencia(
                metrica=cons_metrica,
                lista_media=METRICAS_PARA_MEDIA,
                min_temporadas=cons_min_temp,
                min_minutos=cons_min_min,
                n=cons_n,
                competicoes=cons_competicao if cons_competicao else None,
                posicoes=cons_posicao if cons_posicao else None,
                temporadas=temporadas_periodo 
            )
            
            if erro:
                st.warning(erro)
            else:
                st.subheader(f"Top {cons_n} Mais Consistentes - {cons_metrica}")
                
                col_data, col_graph = st.columns([1.5, 1.5])
                
                with col_data:
                    st.dataframe(
                        df_cons.reset_index()[['index', 'Jogador', 'media_performance', 'CV', 'total_minutos']]
                        .rename(columns={
                            'index': 'Ranking',
                            'media_performance': f'Média de {cons_metrica}', 
                            'total_minutos': 'Minutos Jogados',
                            'CV': 'Coeficiente de Variação (CV)'
                        }),
                        hide_index=True,
                        width='stretch'
                    )
                
                with col_graph:
                    fig_cons = px.scatter(
                        df_cons,
                        x='CV', y='media_performance',
                        text='Jogador', size='num_temporadas',
                        labels={'media_performance': f'Média {cons_metrica}'},
                        template="plotly_dark", color='CV',
                        color_continuous_scale=['#e4f2da', '#22c912']
                    )
                    # PADRONIZAÇÃO APLICADA
                    fig_cons.update_traces(
                        textposition='top center',
                        textfont_size=13, # Tamanho padronizado
                        cliponaxis=False  # Evita cortar o texto
                    )
                    fig_cons.update_layout(xaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_cons, width='stretch')
        else:
            st.info("Selecione uma métrica e a temporada de início para analisar a consistência.")

    # --------------------------------------------------------------------------
    # 4. CURVA DE IDADE
    # --------------------------------------------------------------------------
    with subtab_idade:
        st.subheader("Ranking de Curva de Performance por Idade")
        st.markdown("Analisa em qual idade os jogadores costumam atingir o auge para uma determinada métrica.")
        
        # Linha 1: Configurações Numéricas (Métrica, Min. Jogadores, Min. Minutos)
        ci1, ci2, ci3 = st.columns([2, 1, 1])
        
        with ci1:
            idade_metrica = st.selectbox(
                "**Selecione a Métrica para o Ranking:**", 
                options=sorted(METRICAS_PARA_MEDIA + METRICAS_PARA_SOMAR), 
                index=None,
                placeholder="Selecione uma métrica",
                key="idade_met"
            )
        
        with ci2:
            idade_min_jog = st.number_input(
                "**Mínimo de Jogadores por Idade:**", 
                min_value=1,
                value=5,
                step=1,
                key="idade_min_jog",
                help="Quantidade mínima de jogadores distintos necessária para formar uma média de idade válida."
            )

        with ci3:
            # Novo filtro de Minutos
            idade_min_minutos = st.number_input(
                "**Média Mínima de Minutos:**", 
                min_value=0,
                value=1000,
                step=50,
                key="idade_min_min",
                help="Ignora registros de jogadores com poucos minutos jogados na temporada."
            )

        # Linha 2: Filtros de Contexto (Temporada, Competição, Posição)
        ci4, ci5, ci6 = st.columns(3)
        
        with ci4:
            idade_temp = st.multiselect(
                "**Temporadas:**",
                sorted(meu_dataframe['Temporada'].unique()),
                default=[],
                placeholder="Todas as temporadas",
                key="idade_temp"
            )

        with ci5:
            idade_comp = st.multiselect(
                "**Competições:**",
                sorted(meu_dataframe['Competição'].unique()),
                default=[],
                placeholder="Todas as competições",
                key="idade_comp"
            )

        with ci6:
            idade_posicoes = st.multiselect(
                "**Posições:**", 
                sorted(meu_dataframe['Posição'].unique()), 
                default=[],
                placeholder="Todas as posições",
                key="idade_pos"
            )
            
        # st.markdown("---")
            
        if idade_metrica:
            df_curva, erro = analisador.analise_idade_performance(
                metrica=idade_metrica,
                lista_media=METRICAS_PARA_MEDIA,
                posicoes=idade_posicoes if idade_posicoes else None,
                competicoes=idade_comp if idade_comp else None,
                temporadas=idade_temp if idade_temp else None,
                min_jogadores=idade_min_jog,
                min_minutos=idade_min_minutos # Passando o novo parâmetro
            )
            
            if erro:
                st.warning(erro)
            else:
                st.subheader(f"Performance Média por Idade - {idade_metrica}")
                
                idx_auge = df_curva['performance_media'].idxmax()
                idade_auge = df_curva.loc[idx_auge, 'Idade']
                valor_auge = df_curva.loc[idx_auge, 'performance_media']
                
                # Layout: Tabela (Esq) | Gráfico (Dir)
                col_data, col_graph = st.columns([1.5, 2])
                
                with col_data:
                    # Preparar o dataframe para exibição
                    df_display = df_curva.rename(columns={
                        'performance_media': f'Média de {idade_metrica}',
                        'contagem_jogadores': 'Quantidade de Jogadores',
                        'media_minutos': 'Média de Minutos Jogados' # <--- Nova Coluna
                    })
                    
                    # Opcional: Arredondar a média de minutos para ficar mais limpo visualmente
                    df_display['Média de Minutos Jogados'] = df_display['Média de Minutos Jogados'].round(0).astype(int)

                    st.dataframe(
                        df_display,
                        width='stretch',
                        hide_index=True
                    )

                with col_graph:
                    fig_idade = px.line(
                        df_curva,
                        x='Idade', y='performance_media',
                        markers=True,
                        template="plotly_dark",
                        labels={'performance_media': f'Média de {idade_metrica}'},
                        color_discrete_sequence=['#e4f2da']
                    )

                    fig_idade.update_traces(
                        cliponaxis=False
                    )
                    
                    fig_idade.add_annotation(
                        x=idade_auge, y=valor_auge,
                        text=f"Auge ({int(idade_auge)} anos)",
                        showarrow=True, arrowhead=1,
                        yshift=10,
                        font=dict(color="#22c912")
                    )
                    
                    fig_idade.add_shape(
                        type="line",
                        x0=idade_auge, y0=df_curva['performance_media'].min(),
                        x1=idade_auge, y1=valor_auge,
                        line=dict(color="#22c912", width=1, dash="dash"),
                    )
                    
                    st.plotly_chart(fig_idade, width='stretch')
        else:
            st.info("Selecione uma métrica para visualizar a curva de idade.")


with tab_dados:
    st.header("Explorador da Base de Dados Completa")

    df_filtrado_dados_brutos = meu_dataframe.copy()
    colunas_filtragem = sorted(meu_dataframe.columns)

    col_num_filtros, _ = st.columns([1, 3])
    with col_num_filtros:
        num_filtros = st.number_input(
            label="**Quantos Filtros Deseja Aplicar?**",
            min_value=0,
            step=1,
            key="num_filtros_dados"
        )

    st.markdown("---")

    if num_filtros > 0:
        cols = st.columns(3)
        
        for i in range(num_filtros):
            with cols[i % 3]:
                st.subheader(f"{i+1}º Filtro")
                
                coluna_selecionada = st.selectbox(
                    label="Selecione a coluna",
                    options=colunas_filtragem,
                    key=f"coluna_{i}",
                    index=None,
                    placeholder="Selecione a métrica",
                    label_visibility="collapsed"
                )

                if coluna_selecionada:
                    if pd.api.types.is_numeric_dtype(df_filtrado_dados_brutos[coluna_selecionada]) and coluna_selecionada not in ['Idade', 'Ano de Nascimento', 'Temporada']:
                        col_valor, col_operador = st.columns([2, 3])

                        with col_valor:
                            valor_numerico = st.number_input(
                                label="**Valor:**",
                                key=f"valor_num_{i}",
                                value=0.0,
                                format="%.2f"
                            )
                        
                        with col_operador:
                            operador = st.selectbox(
                                label="**Condição:**",
                                options=['Maior que', 'Menor que', 'Igual a', 'Maior ou igual a', 'Menor ou igual a'],
                                key=f"operador_{i}",
                                index=None,
                                placeholder="Selecione uma condição"
                            )
                        
                        if operador == 'Maior que':
                            df_filtrado_dados_brutos = df_filtrado_dados_brutos[df_filtrado_dados_brutos[coluna_selecionada] > valor_numerico]
                        elif operador == 'Menor que':
                            df_filtrado_dados_brutos = df_filtrado_dados_brutos[df_filtrado_dados_brutos[coluna_selecionada] < valor_numerico]
                        elif operador == 'Igual a':
                            df_filtrado_dados_brutos = df_filtrado_dados_brutos[df_filtrado_dados_brutos[coluna_selecionada] == valor_numerico]
                        elif operador == 'Maior ou igual a':
                            df_filtrado_dados_brutos = df_filtrado_dados_brutos[df_filtrado_dados_brutos[coluna_selecionada] >= valor_numerico]
                        elif operador == 'Menor ou igual a':
                            df_filtrado_dados_brutos = df_filtrado_dados_brutos[df_filtrado_dados_brutos[coluna_selecionada] <= valor_numerico]

                    else:
                        valores_unicos = sorted(df_filtrado_dados_brutos[coluna_selecionada].unique())
                        valores_selecionados = st.multiselect(
                            label="**Selecione o(s) valore(s):**",
                            options=valores_unicos,
                            key=f"valor_cat_{i}",
                            default=None,
                            placeholder="Todos os valores"
                        )

                        if valores_selecionados:
                            df_filtrado_dados_brutos = df_filtrado_dados_brutos[df_filtrado_dados_brutos[coluna_selecionada].isin(valores_selecionados)]
        
        st.markdown("---")

    if meu_dataframe.equals(df_filtrado_dados_brutos):
        st.subheader("Base de Dados")
    else:
        st.subheader("Base de Dados Filtrada")

    st.dataframe(df_filtrado_dados_brutos, hide_index=True)


with tab_chatbot:
    # Configura o cabeçalho e a descrição da aba
    st.header("Assistente de Inteligência Esportiva")
    # st.caption("Faça perguntas abertas sobre estatísticas, times ou jogadores. O assistente usará ferramentas nativas para consultar os dados com filtros avançados.")

    # ==========================================
    # 1. TOOLS
    # ==========================================

    def buscar_informacao_direta(coluna_busca: str, termo_busca: str, colunas_retorno: list[str], temporadas: list[str] = None, competicoes: list[str] = None, posicoes: list[str] = None) -> str:
        """
        Usa para buscar informações exatas sobre um jogador, time ou competição.
        Exemplo: Onde o 'Vini Jr' joga na temporada 2023?
        """
        try:
            df_alvo = meu_dataframe.copy()
            
            # Aplica filtros de contexto apenas se o usuário pedir (temporada, campeonato, posição)
            if temporadas: df_alvo = df_alvo[df_alvo['Temporada'].isin(temporadas)]
            if competicoes: df_alvo = df_alvo[df_alvo['Competição'].isin(competicoes)]
            if posicoes: df_alvo = df_alvo[df_alvo['Posição'].str.contains('|'.join(posicoes), case=False, na=False, regex=True)]

            # Procura o termo ignorando letras maiúsculas/minúsculas
            mascara = df_alvo[coluna_busca].astype(str).str.contains(termo_busca, case=False, na=False)
            resultado = df_alvo[mascara][colunas_retorno].drop_duplicates()
            
            # Retorna o dado ou avisa que não encontrou
            if resultado.empty:
                return f"Nenhum dado encontrado para '{termo_busca}' na coluna '{coluna_busca}' com os filtros aplicados."
            return resultado.to_string(index=False)
        except Exception as e:
            return f"Erro na busca: verifique se as colunas informadas existem na base de dados."

    def calcular_estatistica(metrica: str, operacao: str = 'soma', coluna_filtro: str = None, termo_filtro: str = None, temporadas: list[str] = None, competicoes: list[str] = None, posicoes: list[str] = None) -> str:
        """
        Usa para calcular totais (soma) ou médias numéricas (media) de um grupo ou jogador.
        Exemplo: Média de idade do Real Madrid na La Liga.
        """
        try:
            df_alvo = meu_dataframe.copy()
            
            # Aplica filtros de contexto (temporada, campeonato, posição)
            if temporadas: df_alvo = df_alvo[df_alvo['Temporada'].isin(temporadas)]
            if competicoes: df_alvo = df_alvo[df_alvo['Competição'].isin(competicoes)]
            if posicoes: df_alvo = df_alvo[df_alvo['Posição'].str.contains('|'.join(posicoes), case=False, na=False, regex=True)]
            
            # Filtra por um jogador ou time específico, se necessário
            if coluna_filtro and termo_filtro:
                df_alvo = df_alvo[df_alvo[coluna_filtro].astype(str).str.contains(termo_filtro, case=False, na=False)]
            
            if df_alvo.empty:
                return f"Não encontrei dados para calcular após buscar '{termo_filtro}' em '{coluna_filtro}'."

            # Executa a operação matemática escolhida pela IA
            if operacao == 'soma': valor = df_alvo[metrica].sum()
            elif operacao == 'media': valor = df_alvo[metrica].mean()
            elif operacao == 'maximo': valor = df_alvo[metrica].max()
            else: valor = df_alvo[metrica].min() 
                
            return f"O resultado da operação '{operacao}' na métrica '{metrica}' é {valor:.2f}"
        except Exception as e:
            return f"Erro ao calcular. Verifique se a métrica '{metrica}' é uma coluna numérica válida."

    def criar_ranking(coluna_agrupamento: str, metrica: str, operacao: str = 'soma', top_n: int = 5, crescente: bool = False, temporadas: list[str] = None, competicoes: list[str] = None,
                      posicoes: list[str] = None, min_minutos: int = 0) -> str:
        """
        Usa para responder perguntas comparativas como "Quem tem mais", "Qual o pior", "Top 5".
        Aceita filtros de temporada, competicao, posicao e um filtro de corte de "min_minutos" jogados.
        """
        try:
            df_alvo = meu_dataframe.copy()
            
            # Aplica filtros de contexto (temporada, campeonato, posição)
            if temporadas: df_alvo = df_alvo[df_alvo['Temporada'].isin(temporadas)]
            if competicoes: df_alvo = df_alvo[df_alvo['Competição'].isin(competicoes)]
            if posicoes: df_alvo = df_alvo[df_alvo['Posição'].str.contains('|'.join(posicoes), case=False, na=False, regex=True)]

            if df_alvo.empty:
                return "Os filtros aplicados não retornaram nenhum dado para gerar o ranking."

            # Agrupa os dados e soma os minutos jogados para validação futura
            if operacao == 'media':
                df_agg = df_alvo.groupby(coluna_agrupamento, as_index=False).agg({metrica: 'mean', 'Minutos Jogados': 'sum'})
            else:
                df_agg = df_alvo.groupby(coluna_agrupamento, as_index=False).agg({metrica: 'sum', 'Minutos Jogados': 'sum'})
            
            # Corta jogadores que não atingiram a minutagem mínima exigida
            if min_minutos > 0:
                df_agg = df_agg[df_agg['Minutos Jogados'] >= min_minutos]
                
            if df_agg.empty:
                return f"Nenhum {coluna_agrupamento} atingiu o critério de {min_minutos} minutos jogados."

            # Ordena do maior pro menor (ou vice-versa) e pega os top_n resultados
            df_ordenado = df_agg.sort_values(by=metrica, ascending=crescente).head(top_n)
            
            # Retorna a tabela limpa para a IA ler
            return df_ordenado[[coluna_agrupamento, metrica, 'Minutos Jogados']].to_string(index=False)
            
        except Exception as e:
            return f"Erro ao gerar ranking. Detalhe: {e}"

    # Lista de funções que a IA tem permissão para usar
    ferramentas_pandas = [buscar_informacao_direta, calcular_estatistica, criar_ranking]

    # ==========================================
    # 2. INTERFACE E CONFIGURAÇÃO DO CHAT
    # ==========================================
    
    # Cria a memória do chat se ela não existir
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Inicializa a IA na primeira execução
    if "chat_session" not in st.session_state and CLIENT is not None:
        # Pega as colunas do CSV para a IA saber o que pode pesquisar
        colunas_disponiveis = ", ".join([f"'{col}'" for col in meu_dataframe.columns])
        
        # Regras de comportamento e personalidade do bot
        instrucao = f"""
        Você é um assistente de inteligência de dados de futebol.
        Você DEVE USAR as ferramentas fornecidas para consultar a base de dados. Nunca invente dados.
        
        As colunas disponíveis na base são exatamente estas: {colunas_disponiveis}.
        
        Sua lógica de decisão:
        1. O usuário quer saber uma característica (time, posição, nacionalidade)? Use `buscar_informacao_direta`.
        2. O usuário quer saber a soma, média, máximo ou mínimo de uma métrica para um jogador/time? Use `calcular_estatistica`.
        3. O usuário pediu um ranking, comparação ou "quem tem mais"? Use `criar_ranking`.
        
        TRADUÇÃO DE JARGÕES E SINÔNIMOS:
        - "chutes", "finaliza", "bate pro gol" -> Mapeie para a coluna 'Finalizações' ou 'Finalizações no Gol'.
        - "passes decisivos", "criação", "deixa na cara do gol" -> Mapeie para 'Passes Chave' ou 'Assistências'.
        - "roubadas de bola", "marcação" -> Mapeie para 'Desarmes' ou 'Bolas Recuperadas'.
        - "jogos", "partidas jogadas" -> Mapeie para 'Partidas'.
        
        REGRAS PARA PERGUNTAS SUBJETIVAS E INCOMPLETAS:
        - Se perguntarem "Quem é o melhor jogador?", explique que a resposta depende da métrica e sugira algumas (ex: Gols, Assistências, Média de Pontos). Não dê opiniões pessoais.
        - Mantenha o contexto: se o usuário fizer uma pergunta curta ("E o Vini Jr?"), aplique a mesma métrica da pergunta anterior a este novo jogador.
        
        FILTROS AVANÇADOS:
        - Temporadas: "em 2023", "ano passado".
        - Competições: "na La Liga", "na Champions".
        - Posições: "entre os atacantes", "zagueiros".
        - Minutagem: "com mais de 1000 minutos" (use `min_minutos` na ferramenta).

        REGRAS SOBRE GRÁFICOS E IMAGENS:
        - Você NÃO consegue plotar gráficos. Explique isso educadamente e RECOMENDE que o usuário utilize as abas "Análises Individuais" ou "Rankings Globais" no topo do painel.
        
        FORMATAÇÃO DA RESPOSTA FINAL:
        - Sempre formate rankings e comparações utilizando Tabelas Markdown para facilitar a leitura.
        - Destaque o nome dos jogadores e times em **negrito**.
        - Não explique ao usuário qual ferramenta você usou ou o passo a passo lógico. Entregue apenas a resposta analítica.
        """
        
        # Define as regras e abaixa a temperatura para evitar "alucinações"
        configuracao = types.GenerateContentConfig(
            tools=ferramentas_pandas,
            system_instruction=instrucao,
            temperature=0.1
        )
        
        # Cria a sessão do Gemini
        st.session_state.chat_session = CLIENT.chats.create(
            model='gemini-2.5-flash',
            # model='gemini-3.1-flash-lite-preview', 
            config=configuracao
        )

    # Renderiza o histórico de mensagens na tela
    container_mensagens = st.container()
    for message in st.session_state.messages:
        with container_mensagens.chat_message(message["role"]):
            st.markdown(message["content"])

    # ==========================================
    # 3. PROCESSAMENTO DA MENSAGEM
    # ==========================================
    
    # Campo de digitação para o usuário
    if prompt := st.chat_input("Tire sua dúvida"):
        
        # Salva e exibe a pergunta
        st.session_state.messages.append({"role": "user", "content": prompt})
        with container_mensagens.chat_message("user"):
            st.markdown(prompt)

        # Processa a resposta da IA
        with container_mensagens.chat_message("assistant"):
            if CLIENT is None:
                st.error("O assistente está offline. Verifique a chave da API.")
            else:
                with st.spinner("Analisando os dados..."):
                    try:
                        # Envia a pergunta exigindo streaming (pedaço por pedaço)
                        response_stream = st.session_state.chat_session.send_message_stream(prompt)
                        
                        # Função interna para filtrar os pedaços que contêm texto legível
                        def stream_generator():
                            for chunk in response_stream:
                                if chunk.text:
                                    yield chunk.text
                        
                        # Imprime na tela com efeito de máquina de escrever
                        texto_resposta = st.write_stream(stream_generator())
                        
                        # Salva a resposta final na memória
                        st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
                        
                    except Exception as e:
                        st.error("Ops! Tive um problema ao analisar os dados. Tente simplificar a pergunta.")