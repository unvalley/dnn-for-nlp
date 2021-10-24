from sklearn import model_selection
import pandas as pd

"""
サンプルデータが，90%の正のサンプル，10%の負のサンプルといった偏った分布である場合，
k-fold交差検証をしてもデータが偏る場合がある

stratified_kfoldは，各分割における目的変数の比率を一定に保つことが出来る
"""

if __name__ == "main":
    train_path = 'cross-validation/wine-quality.tsv'
    df = pd.read_csv(train_path, sep='\t')

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.quality

    kf = model_selection.StratifiedKFold(n_splits=5)

    # kf.splitには，割合を均一に保ちたい列を指定する
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv("train_stratified_folds.tsv", index=False, sep='\t')
