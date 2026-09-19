import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

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

df_limpo = df.drop(columns=['Jogadore(a)s.1'])
df_limpo['Jogadore(a)s'] = df['Jogadore(a)s'].apply(limpar_jogadores)
df_ohe = df_one_hot_encoding(df_limpo)
print(df_ohe)

frequent_itemsets = apriori(
    df_ohe,
    min_support=0.05,
    use_colnames=True
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)
print(rules)