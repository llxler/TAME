import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file paths
real_data_paths = [
    'synthetic/adult/real.csv',
    'synthetic/default/real.csv',
    'synthetic/shoppers/real.csv',
    'synthetic/cardio_train/real.csv'
]

synthetic_data_paths = [
    # adult
    ['sample_end_csv/tabsyn_adult_ori.csv', 'sample_end_csv/tabsyn_adult_new.csv', 
     'sample_end_csv/tabddpm_adult_ori.csv', 'sample_end_csv/tabddpm_adult_new.csv'],
    # default
    ['sample_end_csv/tabsyn_default_ori.csv', 'sample_end_csv/tabsyn_default_-1_90w.csv',
     'sample_end_csv/tabddpm_default_ori.csv', 'sample_end_csv/tabddpm_default_new.csv'],
    # shoppers
    ['sample_end_csv/tabsyn_shoppers_ori.csv', 'sample_end_csv/tabsyn_shoppers_900000.csv',
     'sample_end_csv/tabddpm_shoppers_ori.csv', 'sample_end_csv/tabddpm_shoppers_new.csv'],
    # cardio_train
    ['sample_end_csv/tabsyn_cardio_ori.csv', 'sample_end_csv/tabsyn_cardio_-5_90w.csv',
     'sample_end_csv/tabddpm_cardio_ori.csv', 'sample_end_csv/tabddpm_cardio_-5_90w.csv']
]

def plot_comparison(datasets):
    sns.set_context("notebook", font_scale=1)  # Adjust font scale to increase text size

    # Set all font sizes to 18
    plt.rc('font', size=24)  # controls default text sizes
    plt.rc('axes', titlesize=24)  # fontsize of the axes title
    plt.rc('axes', labelsize=24)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=24)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
    plt.rc('legend', fontsize=16)  # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title

    fig, axes = plt.subplots(2, len(datasets), figsize=(24, 12))

    plt.tight_layout(rect=[0.02, 0.32, 0.98, 0.95])  # Adjust the rect to reduce left and right margins
    plt.subplots_adjust(wspace=0.45, hspace=0.45)

    for i, dataname in enumerate(datasets):
        print(f"\n--- Processing dataset: {dataname} ---\n")

        # File paths
        real_data_path = real_data_paths[i]
        generated_data_path_tabsyn = synthetic_data_paths[i][0]
        generated_data_path_tabcutmix = synthetic_data_paths[i][1]

        # Load the data
        real_data = pd.read_csv(real_data_path)[:50]
        tabsyn_data = pd.read_csv(generated_data_path_tabsyn)[:50]
        tabcutmix_data = pd.read_csv(generated_data_path_tabcutmix)[:50]

        # Select a numerical feature and a categorical feature for each dataset
        if dataname == 'adult':
            num_feature = 'fnlwgt'
            cat_feature = 'relationship'
        elif dataname == 'default':
            num_feature = 'BILL_AMT4'
            cat_feature = 'PAY_0'
        elif dataname == 'shoppers':
            num_feature = 'ExitRates'
            cat_feature = 'VisitorType'
        elif dataname == 'magic':
            num_feature = 'Asym'  # Example feature, adjust as needed
            cat_feature = 'class'  # Example feature, adjust as needed
        elif dataname == 'cardio_train':
            num_feature = 'height'
            cat_feature = 'cholesterol'

        # Plot numerical feature (Density Plot)
        ax = axes[0, i]
        ax.grid()
        sns.kdeplot(real_data[num_feature], ax=ax, label='Real', color='blue', fill=True)
        sns.kdeplot(tabsyn_data[num_feature], ax=ax, label='TabSyn', color='orange', fill=True)
        sns.kdeplot(tabcutmix_data[num_feature], ax=ax, label='TabSyn+TameSyn', color='green', fill=True)
        ax.set_title(f'{dataname.capitalize()}')
        ax.set_ylabel('Density')
        if i == len(datasets) - 1:  # Only show legend for the last plot in the row
            ax.legend()
        else:
            ax.legend().remove()

        # Plot categorical feature (Bar Plot)
        ax = axes[1, i]
        ax.grid()
        real_counts = real_data[cat_feature].value_counts(normalize=True)
        tabsyn_counts = tabsyn_data[cat_feature].value_counts(normalize=True)
        tabcutmix_counts = tabcutmix_data[cat_feature].value_counts(normalize=True)

        df_bar = pd.DataFrame({
            'Category': real_counts.index,
            'Real': real_counts.values,
            'TabSyn': tabsyn_counts.reindex(real_counts.index, fill_value=0).values,
            'Tabsyn+TameSyn': tabcutmix_counts.reindex(real_counts.index, fill_value=0).values
        })

        df_bar_melted = df_bar.melt(id_vars='Category', var_name='Model', value_name='Proportion')
        sns.barplot(x='Category', y='Proportion', hue='Model', data=df_bar_melted, ax=ax)

        ax.set_xlabel(f'{cat_feature.capitalize()}')  # Set x-axis label to feature name
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if i == len(datasets) - 1:  # Only show legend for the last plot in the row
            ax.legend()
        else:
            ax.legend().remove()
    plt.savefig('quality_bar.pdf', format='pdf')

    plt.show()



if __name__ == "__main__":
    datasets = ['adult', 'default', 'shoppers', 'cardio_train']
    plot_comparison(datasets)
