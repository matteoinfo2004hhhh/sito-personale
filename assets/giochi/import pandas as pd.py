import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset Iris.

iris = sns.load_dataset('iris')

# Esplora il dataset.

def explore_dataset():
    print("Informazioni sul dataset:\n")
    print(iris.info())

    print("\nStatistiche descrittive del dataset:")
    print(iris.describe())

    print("\nPrime 5 righe del dataset:")
    print(iris.head())

# Visualizza grafici più elaborati.

def visualize_data():
    # Creazione di un pairplot per confrontare tutte le combinazioni di features
    sns.pairplot(iris, hue='species', markers=["o", "s", "D"])

    # Boxplot delle misure dei sepali e dei petali per ogni specie
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.boxplot(x='species', y='sepal_length', data=iris, ax=axes[0, 0])
    axes[0, 0].set_title('Lunghezza Sepalo')
    sns.boxplot(x='species', y='sepal_width', data=iris, ax=axes[0, 1])
    axes[0, 1].set_title('Larghezza Sepalo')
    sns.boxplot(x='species', y='petal_length', data=iris, ax=axes[1, 0])
    axes[1, 0].set_title('Lunghezza Petalo')
    sns.boxplot(x='species', y='petal_width', data=iris, ax=axes[1, 1])
    axes[1, 1].set_title('Larghezza Petalo')

    plt.tight_layout()

    # Heatmap per visualizzare la correlazione tra le features.

    plt.figure(figsize=(8, 6))
    sns.heatmap(iris.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlazione tra le features')

    plt.show()

# Funzione principale.

def main():
    print("Esplorazione del dataset Iris:")
    explore_dataset()
    
    print("\nVisualizzazione dei dati:")
    visualize_data()

if __name__ == "__main__":
    main()
    