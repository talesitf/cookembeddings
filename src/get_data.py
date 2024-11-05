import pandas as pd

def get_data(path):
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    data = get_data("https://raw.githubusercontent.com/josephrmartinez/recipe-dataset/refs/heads/main/13k-recipes.csv")
    data = data["Instructions"]
    data.to_csv("data/13k-recipes.csv", index=False)