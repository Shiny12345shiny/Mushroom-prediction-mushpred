from sklearn.preprocessing import LabelEncoder
def encode(df):
    enccoder=LabelEncoder()
    for column in range(len(df.columns)):
        df[df.columns[column]]=enccoder.fit_transform(df[df.columns[column]])
    return df