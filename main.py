from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


app=FastAPI(debug=True)

base = pd.read_csv('maindb.csv')
reviews = pd.read_csv('clean_output_user_reviews.csv')
games_reviews=pd.read_csv('steam_games_devel.csv')
sample=pd.read_csv('sample.csv')
sample.drop('Unnamed: 0',axis=1,inplace=True)


@app.get('/')
def message():
    return 'Proyecto Individual Nicolás Hernández'


@app.get('/PlayTimeGenre/')
def PlayTimeGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_db=base[base[genre]==1]
    playtime_db=genre_db.groupby('release_year')['playtime_forever'].sum().reset_index()
    max_playtime_year= playtime_db.loc[playtime_db['playtime_forever'].idxmax(),'release_year']
    return {'The year that has the most number of hours played for the genre': genre, 'was': max_playtime_year}

@app.get('/UserForGenre/')
def UserForGenre(genre: str) -> dict:
    genre=genre.capitalize()
    genre_db=base[base[genre]==1]
    playtime_user=genre_db.groupby(['user_id'])['playtime_forever'].sum().reset_index()
    playtime_db=genre_db.groupby(['user_id','release_year'])['playtime_forever'].sum().reset_index()
    max_playtime_user=playtime_user.loc[playtime_user['playtime_forever'].idxmax(),'user_id']
    playtime_db1=playtime_db[playtime_db['user_id']==max_playtime_user]
    listtime=[]
    for i, row in playtime_db1.iterrows(): 
        listtime.append(['Year:', row['release_year'],'Time played', row['playtime_forever']])
    return {"The player with the most hours played in genre": genre, "was": max_playtime_user, "with the following hour split": listtime}



@app.get('/UsersRecommend/')
def UsersRecommend(year: int) -> dict:
    reviews_f=reviews[(reviews['year']== year)&((reviews['sentiment_analysis']==1)|(reviews['sentiment_analysis']==2))&(reviews['recommend']==True)]
    reviews_fr=reviews_f.groupby('item_id')['recommend'].sum().reset_index()
    merged_df=pd.merge(games_reviews,reviews_fr,right_on='item_id',left_on='id')
    search=merged_df.sort_values(by='recommend',ascending=False).head(3)
    most_rec_titles=search['title'].tolist()
    return {"The most recomended games for the year": year, "1st": most_rec_titles[0],"2nd": most_rec_titles[1],"3rd": most_rec_titles[2]}


@app.get('/UsersWorstDeveloper/')
def UsersWorstDeveloper(year: int) -> dict:
    reviews_f=reviews[(reviews['year']== year)&(reviews['sentiment_analysis']==0)&(reviews['recommend']==False)]
    reviews_fr=reviews_f.groupby('item_id')['recommend'].count().reset_index()
    merged_df=pd.merge(games_reviews,reviews_fr,right_on='item_id',left_on='id')
    search=merged_df.groupby('developer')['recommend'].sum().reset_index().sort_values(by='recommend',ascending=False)
    least_recom_developers=search['developer'].tolist()
    return {"The least recomended developers for the year": year, "1st": least_recom_developers[0],"2nd": least_recom_developers[1],"3rd": least_recom_developers[2]}
    
@app.get('/sentiment_analysis/')
def sentiment_analysis(developer: str) -> dict:
    merged_df=pd.merge(games_reviews,reviews,right_on='item_id',left_on='id')
    developer_reviews = merged_df[merged_df['developer'] == developer]
    positive_count = len(developer_reviews[developer_reviews['sentiment_analysis'] == 2])
    neutral_count = len(developer_reviews[developer_reviews['sentiment_analysis'] == 1])
    negative_count = len(developer_reviews[developer_reviews['sentiment_analysis'] == 0])

    results_dict = {
        developer: {
            'Negative': negative_count,
            'Neutral': neutral_count,
            'Positive': positive_count
        }
    }
    return results_dict


sample=sample.head(10000)
tfidf = TfidfVectorizer(stop_words='english')
sample=sample.fillna("")

tdfid_matrix=tfidf.fit_transform(sample['review'])
cosine_similarity=linear_kernel(tdfid_matrix,tdfid_matrix)

@app.get('/recomendation_id/{product_id}')
def game_recomendation(product_id: int):
    if product_id not in sample['id'].values:
        return {'Message': 'Game id does not exist.'}
    
    # We obtain the game genres with the product_id
    genres = sample.columns[3:24]  # We obtain the names of the genre columns
    
    # Filter the dataframe to include the games with similar genres but with different titles.
    filtered_df = sample[(sample[genres] == 1).any(axis=1) & (sample['id'] != product_id)]
    
    # Calculating the similarities of the cosine
    tdfid_matrix_filtered = tfidf.transform(filtered_df['review'])
    cosine_similarity_filtered = linear_kernel(tdfid_matrix_filtered, tdfid_matrix_filtered)
    
    idx = sample[sample['id'] == product_id].index[0]
    sim_cosine = list(enumerate(cosine_similarity_filtered[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_games = filtered_df['title'].iloc[sim_ind].values.tolist()
    
    return {'Recomended Games': list(sim_games)}



@app.get('/game_recomendation/{game_id}')
def game_recomendation(game_id: int):
    if game_id not in sample['id'].values:
        return {'Message': 'Game id does not exist.'}
    title = sample.loc[sample['id'] == game_id, 'title'].iloc[0]
    idx = sample[sample['title'] == title].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_games = sample['title'].iloc[sim_ind].values.tolist()
    return {'juegos recomendados': list(sim_games)}