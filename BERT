import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import re 

# --- Configurações de Caminho ---
diretorio_com_txts = r"C:\Users\tutor18.ead\Downloads\reddit_r\diretorio_com_txts_ipca"
arquivo_saida_resumo_sentimentos = r"C:\Users\tutor18.ead\Downloads\reddit_r\sintese_ipca.xlsx"
diretorio_saida_graficos = r"C:\Users\tutor18.ead\Downloads\reddit_r\graficos_polarizacao_ipca"

# --- Inicializar o Modelo de Análise de Sentimentos BERT ---
try:
    classificador = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    print("Modelo BERT de análise de sentimentos carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo de análise de sentimentos: {e}")
    print("Verifique sua conexão e se as bibliotecas estão instaladas (`pip install transformers`).")
    exit()

# --- Dicionário para converter estrelas em sentimentos ---
categorias_sentimento = {
    "1 star": "Negativo",
    "2 stars": "Negativo",
    "3 stars": "Neutro",
    "4 stars": "Positivo",
    "5 stars": "Positivo" 
}

# --- Dicionários para armazenar os resultados finais ---
resultados_gerais_por_aspecto = {}

# --- Processar cada arquivo TXT por aspecto ---
if not os.path.exists(diretorio_com_txts):
    print(f"Erro: O diretório '{diretorio_com_txts}' não foi encontrado.")
    exit()

os.makedirs(diretorio_saida_graficos, exist_ok=True)
arquivos_txt = [f for f in os.listdir(diretorio_com_txts) if f.endswith('.txt')]

if not arquivos_txt:
    print(f"Nenhum arquivo .txt encontrado no diretório: {diretorio_com_txts}")
    exit()

print(f"\nIniciando análise de sentimentos e geração de gráficos para {len(arquivos_txt)} arquivos...")

for nome_arquivo_txt in arquivos_txt:
    caminho_completo_txt = os.path.join(diretorio_com_txts, nome_arquivo_txt)
    nome_aspecto_limpo = os.path.splitext(nome_arquivo_txt)[0]

    # Ler os comentários do arquivo .txt
    try:
        with open(caminho_completo_txt, 'r', encoding='utf-8') as f:
            comentarios_do_aspecto = [linha.strip() for linha in f if linha.strip()]
    except Exception as e:
        print(f"Erro ao ler o arquivo '{nome_arquivo_txt}': {e}")
        continue
    
    if not comentarios_do_aspecto:
        print(f"Arquivo '{nome_arquivo_txt}' está vazio. Pulando.")
        continue

    contagem_sentimento_aspecto = {"Negativo": 0, "Neutro": 0, "Positivo": 0, "Erros": 0}

    for comentario in comentarios_do_aspecto:
        try:
            resultado = classificador(comentario[:512])[0]
            sentimento = categorias_sentimento.get(resultado['label'], "Neutro")
            contagem_sentimento_aspecto[sentimento] += 1
        except Exception as e:
            contagem_sentimento_aspecto["Erros"] += 1
    
    total_processados = sum(contagem_sentimento_aspecto[s] for s in ["Negativo", "Neutro", "Positivo"])

    # Armazenar resultados para o XLSX consolidado
    resultados_gerais_por_aspecto[nome_aspecto_limpo] = {
        "Total Comentários": len(comentarios_do_aspecto),
        "Total Analisados": total_processados,
        "Negativo": contagem_sentimento_aspecto["Negativo"],
        "Neutro": contagem_sentimento_aspecto["Neutro"],
        "Positivo": contagem_sentimento_aspecto["Positivo"],
        "Erros na Análise": contagem_sentimento_aspecto["Erros"]
    }

    # Geração do Gráfico para o Aspecto Atual
    plt.figure(figsize=(10, 6))
    cores = ["#ff6b6b", "#ffd166", "#06d6a0"]
    labels_grafico = ["Negativo", "Neutro", "Positivo"]
    valores_grafico = [contagem_sentimento_aspecto[s] for s in labels_grafico]

    barras = plt.bar(labels_grafico, valores_grafico, color=cores, edgecolor='black', width=0.7)
    plt.title(f"DISTRIBUIÇÃO DE SENTIMENTOS PARA: {nome_aspecto_limpo.upper()}", fontsize=10, pad=20)
    plt.xlabel("Categoria", fontsize=12)
    plt.ylabel("Quantidade", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if total_processados > 0:
        for barra in barras:
            altura = barra.get_height()
            porcentagem = (altura / total_processados) * 100
            plt.text(
                barra.get_x() + barra.get_width()/2,
                altura,
                f"{altura}\n({porcentagem:.1f}%)",
                ha='center', va='bottom', fontsize=10
            )

    caminho_salvar_jpeg = os.path.join(diretorio_saida_graficos, f"sentimentos_barras_{nome_aspecto_limpo}.jpeg")
    plt.savefig(caminho_salvar_jpeg, dpi=300)
    plt.close()
    print(f"Gráfico para '{nome_aspecto_limpo}' salvo com sucesso.")

# --- Salvar resultados consolidados em um arquivo XLSX ---
if resultados_gerais_por_aspecto:
    df_resultados = pd.DataFrame.from_dict(resultados_gerais_por_aspecto, orient='index')
    df_resultados.index.name = 'Aspecto'
    if 'Total Analisados' in df_resultados.columns and (df_resultados['Total Analisados'] > 0).any():
        df_resultados['Negativo (%)'] = (df_resultados['Negativo'] / df_resultados['Total Analisados']) * 100
        df_resultados['Neutro (%)'] = (df_resultados['Neutro'] / df_resultados['Total Analisados']) * 100
        df_resultados['Positivo (%)'] = (df_resultados['Positivo'] / df_resultados['Total Analisados']) * 100

    try:
        df_resultados.to_excel(arquivo_saida_resumo_sentimentos)
        print(f"\nResumo consolidado da análise de sentimentos salvo em: {arquivo_saida_resumo_sentimentos}")
    except PermissionError:
        print("\nERRO: Permissão negada ao salvar o arquivo XLSX. Verifique se ele não está aberto.")
    except Exception as e:
        print(f"\nERRO inesperado ao salvar o arquivo XLSX: {e}")
else:
    print("\nNenhum resultado de análise de sentimento para salvar no XLSX.")

print("\nProcesso de análise por aspecto e geração de gráficos concluído.")
