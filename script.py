import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from pandas import DataFrame

df = pd.read_csv('_ASSOC_VoleiStars.csv', encoding='latin1')

mapa_nomes = {
    'ágata': 'Ágata',
    'agata': 'Ágata',
    'bárbara': 'Bárbara',
    'barbara': 'Bárbara',
    'fábio': 'Fábio',
    'fabio': 'Fábio',
    'ricardo': 'Ricardo',
    'emanuel': 'Emanuel',
    'shelda': 'Shelda',
    'sheldom': 'Sheldom',
    'ana': 'Ana'
}

def limpar_jogadores(texto):

    jogadores = [j.strip() for j in texto.split(',')]
    jogadores_limpos = []
    
    for nome in jogadores:
        nome_normalizado = nome.replace('\xa0', 'á').replace('?', 'Á').lower()


        nome_correto = mapa_nomes.get(nome_normalizado, nome.capitalize())
        jogadores_limpos.append(nome_correto)
        
    return ', '.join(jogadores_limpos)

def df_one_hot_encoding(df):

    colunas = nomes_unicos(df)
    colunas += ["Vitória", "Derrota"]
    df_ohe = pd.DataFrame(columns=colunas)

    for i in range(len(df)):

        linha_ohe = {nome: 0 for nome in colunas}

        nomes = df.loc[i, 'Jogadore(a)s'].split(',')

        for nome in nomes:
            nome_limpo = nome.strip()
            if nome_limpo in linha_ohe:
                linha_ohe[nome_limpo] = 1

        resultado = df.loc[i, "Resultado"].lower()
        if resultado == "ganhou":
            linha_ohe["Vitória"] = 1
        else :
            linha_ohe["Derrota"] = 1

        df_ohe = pd.concat([df_ohe, pd.DataFrame([linha_ohe])], ignore_index=True)

    return df_ohe

def nomes_unicos(df):
    nomes_unicos = []

    for i in df['Jogadore(a)s']:
        nomes = i.split(',')

        for nome in nomes:
            nome_limpo = nome.strip()

            if nome_limpo not in nomes_unicos:
                nomes_unicos.append(nome_limpo)

    return nomes_unicos

def print_rules(df_ohe):
    rules = get_rules(df_ohe)

    print_vitoria(rules)
    print_derrota(rules)
    print_estrela(df_ohe)
    print_estrela_regras(rules)

def print_vitoria(rules):
    regras_vitoria = regras_para(rules, 'Vitória', 0.10)

    print('\n=== Combinação vitoriosa ===')
    print(regras_vitoria[['antecedents', 'support', 'confidence', 'lift']])

def print_derrota(rules):
    regras_derrota = regras_para(rules, 'Derrota', 0.10)

    print('\n=== Combinação perdedora ===')
    print(regras_derrota[['antecedents', 'support', 'confidence', 'lift']])

def get_rules(df_ohe) -> DataFrame:
    frequent_itemsets = apriori(
        df_ohe,
        min_support=0.10,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    return rules

def regras_para(rules, resultado, min_support):
    filtradas = rules[
        (rules['consequents'].apply(lambda x: x == frozenset([resultado])))
        & (rules['antecedents'].apply(lambda x: not (x & {'Vitória', 'Derrota'})))
        & (rules['support'] >= min_support)
    ]
    return filtradas.sort_values(by=['confidence', 'support'], ascending=False)

def print_estrela(df_ohe):
    jogadores = [c for c in df_ohe.columns if c not in ('Vitória', 'Derrota')]

    resumo = pd.DataFrame({
        'partidas': df_ohe[jogadores].sum(),
        'vitorias': df_ohe[jogadores].apply(lambda col: (col & df_ohe['Vitória']).sum()),
    })
    resumo['taxa_vitoria'] = resumo['vitorias'] / resumo['partidas']

    print('\n===Estrela===')
    print(resumo.sort_values(by=['vitorias', 'taxa_vitoria'], ascending=False))

def print_estrela_regras(rules):
    individuais = regras_para(rules, 'Vitória', 0.10)
    individuais = individuais[individuais['antecedents'].apply(lambda x: len(x) == 1)]
    print('\n===regras Estrela===')
    print(individuais[['antecedents', 'support', 'confidence', 'lift']])


df_limpo = df.drop(columns=['Jogadore(a)s.1'])
df_limpo['Jogadore(a)s'] = df['Jogadore(a)s'].apply(limpar_jogadores)
df_ohe = df_one_hot_encoding(df_limpo)
print(df_ohe)
print_rules(df_ohe)
