from tqdm.notebook import tqdm
from sklearn.model_selection import KFold


def cross_val_score(estimator, X, Xy, y, scoring, cv=10, random_state=0):
    
    scores = []
    
    for (tr_ids, te_ids) in tqdm(list(KFold(n_splits=cv, shuffle=True, random_state=random_state).split(X))):

        x_tr, x_te = X[tr_ids], X[te_ids]
        xy_tr, xy_te = Xy[tr_ids], Xy[te_ids]
        y_tr, y_te = y[tr_ids], y[te_ids]

        y_pr = estimator.fit(x_tr, xy_tr, y_tr).predict(x_te, xy_te)
        scores.append(scoring(y_te, y_pr))
        
    return scores
