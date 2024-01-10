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