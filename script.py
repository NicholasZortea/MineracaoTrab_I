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



df_limpo = df.drop(columns=['Jogadore(a)s.1'])
df_limpo['Jogadore(a)s'] = df['Jogadore(a)s'].apply(limpar_jogadores)
