import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/kqbg611/Documents/astrazeneca/az-internship/moa/gsm8k_alpaca_results.csv'

def plot_gsm8k_aplaca_results(file_path, save_path=None):
    data = pd.read_csv(file_path, index_col=0)
    data['GSM8K (%)'] = pd.to_numeric(data['GSM8K (%)'], errors='coerce')
    data['AlpacaEval 2.0 (LC win %)'] = pd.to_numeric(data['AlpacaEval 2.0 (LC win %)'], errors='coerce')

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x='GSM8K (%)', y='AlpacaEval 2.0 (LC win %)', s=100)
    sns.regplot(data=data, x='GSM8K (%)', y='AlpacaEval 2.0 (LC win %)', scatter=False, color='red', label="Regression Line")

    for label, row in data.iterrows():
        plt.annotate(label, (row['GSM8K (%)'], row['AlpacaEval 2.0 (LC win %)']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.title('Correlation between AlpacaEval 2.0 and GSM8K')
    plt.xlabel('GSM8K (%)')
    plt.ylabel('AlpacaEval 2.0 (LC win %)')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path) if save_path else plt.show()
    if save_path:
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":  
    plot_gsm8k_aplaca_results(file_path=file_path, save_path='outputs/gsm8k_alpaca_results.pdf')
