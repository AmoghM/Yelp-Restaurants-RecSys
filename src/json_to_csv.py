import pandas as pd
import csv,json

usr_csv_columns = ['user_id','friends', 'name','review_count','compliment_cute', 'compliment_funny', 'funny', 'compliment_photos', 'compliment_hot', 'compliment_more', 'compliment_plain', 'average_stars', 'review_count', 'compliment_list', 'fans', 'useful', 'cool', 'yelping_since', 'compliment_note', 'elite', 'compliment_profile', 'compliment_cool', 'compliment_writer']
review_csv_columns = ['funny', 'review_id', 'date', 'useful', 'cool', 'user_id', 'business_id', 'text', 'stars']
business_csv_columns = ['city', 'business_id', 'address', 'state', 'categories', 'review_count', 'is_open', 'attributes', 'postal_code', 'name', 'latitude', 'stars', 'hours', 'longitude']
filenames=[('user',usr_csv_columns), ('review',review_csv_columns),('business',business_csv_columns)]
for file in filenames:
    count=0
    print(file)
    path="data/"+file[0]+".json"
    wpath = "data/"+file[0]+".csv"
    with open(path,'r',encoding='utf-8') as f, open(wpath, 'w',encoding='utf-8') as csvfile:
        for i in f:
            data = json.loads(i)
            writer = csv.DictWriter(csvfile, fieldnames=file[1])
            if count==0:
                writer.writeheader()
            writer.writerow(data)
            count+=1