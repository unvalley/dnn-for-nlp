import pandas as pd
from sklearn import model_selection

# k-fold交差検証
if __name__ == "__main__":
    train_path = './wine-quality.tsv'
    df = pd.read_csv(train_path, sep='\t')

    # kfoldという列を作り，-1で初期化
    df["kfold"] = -1

    # ランダマイズ
    df = df.sample(frac=1).reset_index(drop=True)

    # KFoldクラスの初期化(n_splitsに分割数を与える)
    kf = model_selection.KFold(n_splits=5)

    # kfold列を埋めていく
    for fold, [trn_, val_] in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # データセットを新しい列とともに保存
    df.to_csv("train_folds.tsv", index=False, sep='\t')
