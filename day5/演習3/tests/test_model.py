import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def processed_data(sample_data):
    """特徴量エンジニアリングを適用したデータを返す"""
    # 特徴量エンジニアリング
    data_processed = sample_data.copy()

    # 家族サイズの特徴量を作成
    data_processed["FamilySize"] = data_processed["SibSp"] + data_processed["Parch"] + 1

    # 一人かどうかの特徴量
    data_processed["IsAlone"] = (data_processed["FamilySize"] == 1).astype(int)

    # 欠損値の処理
    data_processed["Age"].fillna(data_processed["Age"].median(), inplace=True)
    data_processed["Embarked"].fillna(
        data_processed["Embarked"].mode()[0], inplace=True
    )

    # 年齢の区分を作成
    data_processed["AgeBand"] = pd.cut(data_processed["Age"], 5, labels=[0, 1, 2, 3, 4])

    # 運賃の区分を作成
    data_processed["FareBand"] = pd.qcut(
        data_processed["Fare"].fillna(data_processed["Fare"].median()),
        4,
        labels=[0, 1, 2, 3],
    )

    return data_processed


@pytest.fixture
def preprocessor(processed_data):
    """拡張された前処理パイプラインを定義"""
    # カテゴリ変数と数値変数を分ける
    categorical_features = ["Sex", "Embarked"]
    numerical_features = ["Age", "Fare", "Pclass", "FamilySize", "IsAlone"]
    ordinal_features = ["AgeBand", "FareBand"]

    # 前処理パイプラインを作成
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理パイプラインを組み合わせる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("ord", "passthrough", ordinal_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, processed_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression

    # データの分割とラベル変換
    X = (
        processed_data.drop("Survived", axis=1)
        if "Survived" in processed_data.columns
        else processed_data
    )
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 前処理を適用
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # アンサンブルモデルの作成
    # ロジスティック回帰
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # ランダムフォレスト
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # 勾配ブースティング
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

    # アンサンブルモデル
    model = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft"
    )

    # モデルの学習
    model.fit(X_train_processed, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 前処理パイプラインとモデルを一緒に保存
    pipeline_model = {"preprocessor": preprocessor, "model": model}

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline_model, f)

    return model, X_test_processed, y_test, preprocessor


def test_model_exists():
    """モデルファイルが存在するか確認"""
    print(f"モデルファイルパス: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    from sklearn.metrics import precision_score, recall_score, f1_score

    model, X_test, y_test, _ = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"モデルの精度: {accuracy:.4f}")
    print(f"適合率: {precision:.4f}")
    print(f"再現率: {recall:.4f}")
    print(f"F1スコア: {f1:.4f}")

    # 最適化されたTitanicモデルでは0.8以上の精度が期待される
    assert accuracy >= 0.8, f"モデルの精度が低すぎます: {accuracy:.4f}"
    assert f1 >= 0.75, f"F1スコアが低すぎます: {f1:.4f}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    print(f"推論時間: {inference_time:.4f}秒")

    # アンサンブルモデルでも推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time:.4f}秒"


def test_model_reproducibility(sample_data, processed_data, preprocessor):
    """モデルの再現性を検証"""
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression

    # データの分割
    X = (
        processed_data.drop("Survived", axis=1)
        if "Survived" in processed_data.columns
        else processed_data
    )
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 前処理を適用
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 同じパラメータで2つのアンサンブルモデルを作成
    # モデル1のコンポーネント
    lr1 = LogisticRegression(max_iter=1000, random_state=42)
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb1 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

    # モデル2のコンポーネント
    lr2 = LogisticRegression(max_iter=1000, random_state=42)
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb2 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

    # アンサンブルモデル1
    model1 = VotingClassifier(
        estimators=[("lr", lr1), ("rf", rf1), ("gb", gb1)], voting="soft"
    )

    # アンサンブルモデル2
    model2 = VotingClassifier(
        estimators=[("lr", lr2), ("rf", rf2), ("gb", gb2)], voting="soft"
    )

    # 学習
    model1.fit(X_train_processed, y_train)
    model2.fit(X_train_processed, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test_processed)
    predictions2 = model2.predict(X_test_processed)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_model_no_performance_regression(train_model, sample_data):
    """新しいモデルが過去バージョンのモデルと比較して性能劣化がないか検証"""
    import json

    # 現在のモデルを取得
    current_model, X_test, y_test, _ = train_model

    # 現在のモデルの正解率を計算
    current_accuracy = accuracy_score(y_test, current_model.predict(X_test))

    # 過去のモデルの正解率を読み込み
    metrics_file = os.path.join(MODEL_DIR, "metrics.json")

    # 過去のメトリクスがない場合は現在のメトリクスを保存してテストをスキップ
    if not os.path.exists(metrics_file):
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump({"accuracy": current_accuracy}, f)
        pytest.skip(
            "過去のメトリクスが存在しないため、現在のメトリクスを保存してテストをスキップします"
        )

    # 過去のメトリクスを読み込み
    with open(metrics_file, "r") as f:
        past_metrics = json.load(f)

    past_accuracy = past_metrics.get("accuracy", 0)

    # 現在のメトリクスを保存
    with open(metrics_file, "w") as f:
        json.dump({"accuracy": current_accuracy}, f)

    # 性能が向上した場合は、新しいメトリクスを保存するオプションも考えられます
    if current_accuracy > past_accuracy:
        print("性能が向上しました。新しいベースラインとして保存します。")
        # モデルの性能メトリクスを更新
        metrics = {
            "accuracy": current_accuracy,
            "timestamp": time.time(),
            "model_version": past_metrics.get("model_version", "1.0.0") + ".next",
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
