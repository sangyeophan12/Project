from sklearn import preprocessing
import pandas as pd
USvideos = pd.read_csv('USvidoes_modified.csv',
                       encoding="shift-jis", error_bad_lines=False)

le = preprocessing.LabelEncoder()
USvideos = USvideos.apply(le.fit_transform)
mean = USvideos['subscriber'].mean()
USvideos['subscriber'].fillna(mean, inplace=True)
x = USvideos[['subscriber', 'likes']]
y = USvideos['views']
clf = RandomForestClassifier(n_estimators=82, min_samples_split=2)
clf.fit(x, y)
print(clf.score(x, y))
print(clf.feature_importances_)
p = clf.predict(x)
t = np.arange(0.0, 31.0)
plt.plot(t, data['views'], '--b')
plt.plot(t, p, '-b')
plt.legend(('real', 'randomF'))
plt.show()
