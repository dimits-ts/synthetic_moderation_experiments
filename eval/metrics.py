import joblib
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import util


def load_classifier(model_path: str):
    """Load a pickled DecisionTree classifier."""
    return joblib.load(model_path)


def tokenize_texts(texts):
    """Tokenize a list of raw text strings using your Tokenizer class."""
    tok = util.Tokenizer(preserve_case=True)
    return [" ".join(tok.tokenize(t.lower())) for t in texts]


def featurize_texts(tokenized_texts, vectorizer: CountVectorizer):
    """Convert tokenized texts to vectorizer feature space."""
    return vectorizer.transform(tokenized_texts)


def classify_texts(features, model):
    """Run the DecisionTree classifier and return predictions."""
    return model.predict(features)


def run_classifier(texts: list[str], model, vectorizer: CountVectorizer):
    """
    Full pipeline:
    - tokenize
    - featurize
    - classify
    - return labels
    """
    # Process input
    tok_texts = tokenize_texts(texts)
    X = featurize_texts(tok_texts, vectorizer)

    # Predict
    labels = classify_texts(X, model)

    return labels


def get_texts(df: pd.DataFrame) -> list[str]:
    texts = df.message.to_list()
    texts = [text for text in texts if text != ""]
    return texts


def get_classifiers(classifier_dir: Path) -> dict:
    cls_dict = {}
    for classifier_path in classifier_dir.iterdir():
        name = classifier_path.name.split("_")[0]
        model = load_classifier(classifier_path)
        cls_dict[name] = model
    return cls_dict


def train_vectorizer(texts: list[str], max_features=22113, ngram_range=(1, 2)):
    """Train a TF-IDF and CountVectorizer on the text data."""
    print(
        f"Training with max_features={max_features}, ngram_range={ngram_range}"
    )

    bow = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words="english",
    )

    bow.fit(texts)

    return bow


def main(
    input_csv_path: Path,
    train_vectorizer_csv_path: Path,
    classifier_dir: Path,
    output_path: Path,
):
    vectorizer_train_df = pd.read_csv(train_vectorizer_csv_path)
    vectorizer = train_vectorizer(vectorizer_train_df.message.to_list())

    df = pd.read_csv(input_csv_path)
    texts = get_texts(df)
    classifier_dict = get_classifiers(classifier_dir)

    task_label_dict = {}
    for cls_name, cls_model in classifier_dict.items():
        labels = run_classifier(
            texts,
            model=cls_model,
            vectorizer=vectorizer,
        )
        task_label_dict[cls_name] = labels
    res = pd.DataFrame(task_label_dict)
    res.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Perspective API scoring and save results to CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--vectorizer-train-path",
        type=str,
        required=True,
        help=(
            "Path to file containing the original dataset "
            "used to train the models"
        ),
    )
    parser.add_argument(
        "--classifier-dir",
        type=str,
        required=True,
        help="Directory holding the .pickle files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="perspective_results.csv",
        help="Output path for CSV file",
    )

    args = parser.parse_args()
    main(
        input_csv_path=Path(args.input_csv),
        classifier_dir=Path(args.classifier_dir),
        output_path=Path(args.output_path),
        train_vectorizer_csv_path=Path(args.vectorizer_train_path),
    )
