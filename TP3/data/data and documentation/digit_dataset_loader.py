import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load a digits CSV and deserialise the image column to numpy arrays."""
    df = pd.read_csv(path)
    df["image"] = df["image"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    )
    return df


def get_image(row: pd.Series, size: tuple[int, int] = (28, 28)) -> np.ndarray:
    """Reshape the flat image vector back to a 2-D array."""
    return row["image"].reshape(size)


def plot_sample(row: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(get_image(row), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Label: {int(row['label'])}", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_dataset("digits_test.csv")
    plot_sample(df.iloc[0])
