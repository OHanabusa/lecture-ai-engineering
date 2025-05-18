import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mlflow.models.signature import infer_signature


# データ準備
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = "data/Titanic.csv"
    data = pd.read_csv(path)
    
    # 特徴量エンジニアリング
    # 欠損値の多いCabinは除外し、より多くの特徴量を活用
    data_processed = data.copy()
    
    # 家族サイズの特徴量を作成
    data_processed['FamilySize'] = data_processed['SibSp'] + data_processed['Parch'] + 1
    
    # 一人かどうかの特徴量
    data_processed['IsAlone'] = (data_processed['FamilySize'] == 1).astype(int)
    
    # 敬称から特徴量を抽出
    data_processed['Title'] = data_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # タイトルをグループ化
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Mme': 'Mrs',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Sir': 'Rare',
        'Capt': 'Rare',
        'Countess': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare'
    }
    data_processed['Title'] = data_processed['Title'].map(title_mapping)
    
    # 運賃の区分を作成
    data_processed['FareBand'] = pd.qcut(data_processed['Fare'].fillna(data_processed['Fare'].median()), 4, labels=[0, 1, 2, 3])
    
    # 年齢の欠損値を中央値で補完
    data_processed['Age'].fillna(data_processed['Age'].median(), inplace=True)
    
    # 年齢の区分を作成
    data_processed['AgeBand'] = pd.cut(data_processed['Age'], 5, labels=[0, 1, 2, 3, 4])
    
    # 乗船港の欠損値を最頻値で補完
    data_processed['Embarked'].fillna(data_processed['Embarked'].mode()[0], inplace=True)
    
    # 使用する特徴量を選択
    features = [
        'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
        'FamilySize', 'IsAlone', 'Title', 'AgeBand', 'FareBand'
    ]
    
    # カテゴリ変数とカテゴリ以外の変数を分ける
    categorical_features = ['Sex', 'Embarked', 'Title']
    numerical_features = ['Age', 'Fare', 'Pclass', 'FamilySize', 'IsAlone']
    ordinal_features = ['AgeBand', 'FareBand']
    
    # 前処理パイプラインを作成
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 前処理パイプラインを組み合わせる
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', 'passthrough', ordinal_features)
        ])
    
    # 目的変数
    y = data_processed['Survived']
    
    # 特徴量
    X = data_processed[features]
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor


# 学習と評価
def train_and_evaluate(
    X_train, X_test, y_train, y_test, preprocessor, model_type='ensemble', 
    n_estimators=100, max_depth=None, random_state=42, cv=5
):
    # 前処理を適用
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # モデル選択
    if model_type == 'rf':
        # ランダムフォレストのハイパーパラメータ最適化
        param_grid = {
            'n_estimators': [n_estimators],
            'max_depth': [max_depth] if max_depth is not None else [None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        model = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
    elif model_type == 'gb':
        # 勾配ブースティングのハイパーパラメータ最適化
        param_grid = {
            'n_estimators': [n_estimators],
            'max_depth': [3, 5] if max_depth is None else [max_depth],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        model = GridSearchCV(
            GradientBoostingClassifier(random_state=random_state),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
    else:  # ensemble - 複数モデルのアンサンブル
        # ロジスティック回帰
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # ランダムフォレスト
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth if max_depth is not None else None,
            random_state=random_state
        )
        
        # 勾配ブースティング
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5 if max_depth is None else max_depth,
            random_state=random_state
        )
        
        # アンサンブルモデル
        model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            voting='soft'
        )
    
    # モデル学習
    model.fit(X_train_processed, y_train)
    
    # 予測
    if hasattr(model, 'best_estimator_'):
        predictions = model.best_estimator_.predict(X_test_processed)
        best_params = model.best_params_
        print(f"最適パラメータ: {best_params}")
    else:
        predictions = model.predict(X_test_processed)
        best_params = {}
    
    # 評価指標の計算
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    # クロスバリデーションスコア
    if hasattr(model, 'best_estimator_'):
        cv_scores = cross_val_score(model.best_estimator_, X_train_processed, y_train, cv=cv, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring='accuracy')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model, metrics, best_params


# モデル保存
def log_model(model, metrics, params, X_train_processed, X_test_processed):
    with mlflow.start_run():
        # パラメータをログ
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # メトリクスをログ
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # モデルのシグネチャを推論
        if hasattr(model, 'best_estimator_'):
            signature = infer_signature(X_train_processed, model.best_estimator_.predict(X_train_processed))
            actual_model = model.best_estimator_
        else:
            signature = infer_signature(X_train_processed, model.predict(X_train_processed))
            actual_model = model

        # モデルを保存
        mlflow.sklearn.log_model(
            actual_model,
            "model",
            signature=signature,
            input_example=X_test_processed[:5] if isinstance(X_test_processed, np.ndarray) else X_test_processed.iloc[:5],
        )
        
        # メトリクスを表示
        print(f"モデルのログ記録値")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")
        print(f"params: {params}")


# メイン処理
if __name__ == "__main__":
    # 最適化のための設定
    test_size = 0.2  # 一貫性のためにテストサイズを固定
    data_random_state = 42  # 再現性のために固定
    model_random_state = 42  # 再現性のために固定
    n_estimators = 100  # 十分な数のツリー
    max_depth = 10  # 適切な深さ
    cv = 5  # クロスバリデーションの分割数
    
    # モデルタイプを選択（'rf', 'gb', 'ensemble'のいずれか）
    model_type = 'ensemble'  # アンサンブル手法が一般的に最も高い精度

    # パラメータ辞書の作成
    params = {
        "test_size": test_size,
        "data_random_state": data_random_state,
        "model_random_state": model_random_state,
        "n_estimators": n_estimators,
        "max_depth": "None" if max_depth is None else max_depth,
        "model_type": model_type,
        "cv": cv
    }

    # データ準備（前処理パイプラインも取得）
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        test_size=test_size, random_state=data_random_state
    )

    # 学習と評価
    model, metrics, best_params = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=model_random_state,
        cv=cv
    )
    
    # best_paramsがある場合はparamsに追加
    if best_params:
        params.update(best_params)

    # 前処理を適用したデータを取得
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # モデル保存
    log_model(model, metrics, params, X_train_processed, X_test_processed)

    # モデルをファイルに保存
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"titanic_model.pkl")
    
    # 前処理パイプラインとモデルを一緒に保存
    pipeline_model = {
        'preprocessor': preprocessor,
        'model': model
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(pipeline_model, f)
    print(f"モデルを {model_path} に保存しました")
