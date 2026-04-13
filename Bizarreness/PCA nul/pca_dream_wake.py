import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

raw = pd.read_excel(r'C:\Users\Enzob\Desktop\Stage M2 Genève\Bizarreness\ElceRaw.xlsx')
meta = pd.read_excel(r'C:\Users\Enzob\Desktop\Stage M2 Genève\Bizarreness\ElceData_Enzo.xlsx')

if len(raw) != len(meta):
    raise ValueError(f'Row counts differ: raw {len(raw)} vs meta {len(meta)}')

# use row-wise alignment from ElceData_Enzo to get the dream/wake labels
# CodeRef in meta has DL (dream) and XDL (wakeful) entries in the same order as raw
if 'CodeRef' not in meta.columns:
    raise ValueError('CodeRef is missing from metadata.')

# preserve raw features and add label data from meta
x_df = raw.copy()
x_df['CodeRef'] = meta['CodeRef'].astype(str)
df = x_df


def label_from_coderef(x):
    if isinstance(x, str):
        if x.startswith('X'):
            return 'wake'
        if x.startswith('D'):
            return 'dream'
    return 'unknown'

# encode target
code_ref = df['CodeRef'].astype(str)
label = code_ref.apply(label_from_coderef)

print('CodeRef values count:')
print(code_ref.value_counts(dropna=False).head(20))
print('Label counts:')
print(label.value_counts())

keep = label.isin(['dream', 'wake'])
if keep.sum() == 0:
    raise ValueError('No dream/wake labeled rows found')
df = df.loc[keep].copy()
label = label.loc[keep]

df['target'] = (label == 'wake').astype(int)

# Select numeric predictors, remove id-like columns if present and non-informative
X = df.select_dtypes(include=['number']).copy()
for c in ['id_dream']:
    if c in X.columns:
        X.drop(columns=[c], inplace=True)

# Drop all-NaN or constant columns
X = X.loc[:, X.notna().any(axis=0)]

# Drop rows with NaN
before = X.shape[0]
X = X.dropna(axis=0, how='any')
label = label.loc[X.index]
print(f'data rows {before} -> {X.shape[0]} after dropna')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_

# Save scores and loadings
scores = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=X.index)
out_scores = scores.iloc[:, :10].copy()
out_scores['target'] = label.values
out_scores.to_csv('pca_dream_wake_scores.csv', index=False)

loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f'PC{i+1}' for i in range(X.shape[1])])
loadings.iloc[:, :10].to_csv('pca_dream_wake_loadings.csv')

print('PCA variance by component (first 15):')
for i, v in enumerate(explained[:15], 1):
    print(f'PC{i}: {v:.4f} cum {explained[:i].sum():.4f}')

# Estimate PC relevance using logistic regression
X_pc = scores.iloc[:, :15]
y = label.map({'dream': 0, 'wake': 1}).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_pc, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=10000, solver='liblinear').fit(X_train, y_train)
acc = clf.score(X_test, y_test)
coef = pd.Series(clf.coef_[0], index=X_pc.columns)
relevance = pd.DataFrame({'PC': X_pc.columns, 'coef': coef.values, 'abs_coef': coef.abs().values, 'explained_variance': explained[:15]})
relevance = relevance.sort_values('abs_coef', ascending=False)

relevance.to_csv('pca_dream_wake_pc_relevance.csv', index=False)

print(f'Logistic regression test accuracy (dream vs wake): {acc:.4f}')
print(relevance.head(10))

# Also save final dataframe with label
out_scores_full = scores.iloc[:, :15].copy()
out_scores_full['target'] = y.values
out_scores_full['CodeRef'] = df['CodeRef'].loc[out_scores_full.index].values
out_scores_full.to_csv('pca_dream_wake_scores_with_label.csv', index=False)

print('Created pca_dream_wake scores, loadings, relevance files.')
