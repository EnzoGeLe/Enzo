import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path = r"C:\Users\Enzob\Desktop\Stage M2 Genève\Bizarreness\ElceRaw.xlsx"
print('loading', path)
df = pd.read_excel(path)
print('shape', df.shape)
print('columns', list(df.columns)[:20])
X = df.select_dtypes(include=[float, int]).copy()
print('numeric columns', X.shape[1])
if X.shape[1] == 0:
    raise ValueError('No numeric columns found for PCA')

before = X.shape[0]
X = X.dropna()
print('rows dropped due to NaNs', before - X.shape[0], '->', X.shape[0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cum_explained = explained.cumsum()

print('\nPCA results:')
for i,(e,c) in enumerate(zip(explained, cum_explained), 1):
    print(f'PC{i}: {e:.4f} (cum {c:.4f})')

out_df = pd.DataFrame(X_pca[:, :5], columns=[f'PC{i}' for i in range(1,6)])
out_df.to_csv('pca_scores.csv', index=False)
print('\nWrote PCA scores first 5 PCs to pca_scores.csv')

loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f'PC{i}' for i in range(1, X.shape[1]+1)])
loadings.iloc[:, :5].to_csv('pca_loadings.csv')
print('Wrote PCA loadings first 5 PCs to pca_loadings.csv')
