import pandas as pd

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




def nomes_unicos(df):
    nomes_unicos = []

    for i in df['Jogadore(a)s']:
        nomes = i.split(',')

        for nome in nomes:
            nome_limpo = nome.strip()

            if nome_limpo not in nomes_unicos:
                nomes_unicos.append(nome_limpo)

    return nomes_unicos



def df_one_hot_encoding(df):

    colunas = nomes_unicos(df)
    df_ohe = pd.DataFrame(columns=colunas)  

    for i in df['Jogadore(a)s']:

        linha_ohe = {nome: 0 for nome in colunas}   

        nomes = i.split(',')


        for nome in nomes:
            nome_limpo = nome.strip()
            if nome_limpo in linha_ohe:
                linha_ohe[nome_limpo] = 1

        df_ohe = pd.concat([df_ohe, pd.DataFrame([linha_ohe])], ignore_index=True)
        df_ohe['Resultado']= df['Resultado']

        df_ohe['Resultado'] = df_ohe['Resultado'].map({'GANHOU': 1}).fillna(0).astype(int)


    return df_ohe    

    


df_limpo = df.drop(columns=['Jogadore(a)s.1'])
df_limpo['Jogadore(a)s'] = df['Jogadore(a)s'].apply(limpar_jogadores)
df_ohe = df_one_hot_encoding(df_limpo)
print(df_ohe)