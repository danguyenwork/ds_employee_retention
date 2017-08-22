import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence

df = pd.read_csv('employee_retention_data.csv',parse_dates=['join_date','quit_date']).set_index('employee_id').sort_index()
df.head()

df.info()

df.seniority.describe()

df.boxplot(column='seniority')
plt.title('seniority')
plt.savefig('seniority_boxplot')

df = df[df.seniority < 60]
df['quit'] = 0
df.loc[~df.quit_date.isnull(), 'quit'] = 1

company_group = df.groupby(['company_id'])
company_group.mean().reset_index()[['company_id','salary','quit']]


df.groupby(['company_id', 'dept']).count()
le = preprocessing.LabelEncoder()
df.dept = le.fit(df.dept).transform(df.dept)
df.salary = df.salary / 10000.
df['tenure'] = (df.quit_date - df.join_date).dt.days
mask = df.quit_date.isnull()
df.loc[mask, 'tenure'] = (datetime.strptime('2015-12-13','%Y-%m-%d') - (df[mask].join_date)).dt.days

df.reset_index().head()

bins = [0, 365*.5, 365 * 1, 365 * 1.5, 365 * 2, 365 * 2.5, 365 * 3, 365 * 3.5, 365 * 4, 365 * 4.5, 365 * 20]
group_names = np.arange(10) * 0.5

categories = pd.cut(df['tenure'], bins, labels=group_names)
df['tenure_bin'] = pd.cut(df['tenure'], bins, labels=group_names)
categories
df.head()
tenure_vs_quit = df.groupby('tenure_bin').mean().reset_index()[['tenure_bin','quit']]
tenure_vs_quit.quit.unique()
tenure_vs_quit
plt.show()
plt.scatter(tenure_vs_quit['tenure_bin'], tenure_vs_quit['quit'])
plt.show()
plt.xlabel('Tenure')
plt.ylabel('Attrition Rate')
plt.savefig('tenure_bin.png')
X = df[['company_id','dept','seniority','salary','tenure']]
y = df.quit
gdbr = GradientBoostingClassifier(n_estimators=500, max_depth=5).fit(X,y)
y_pred = gdbr.predict(X)
print (gdbr.feature_importances_)

features = [0,1,2,3,4]
names = X.columns
fig, axs = plot_partial_dependence(gdbr, X, features,feature_names=names,n_jobs=3, grid_resolution=100)
plt.savefig('feature_importances_1.png')
