import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

def get_cat_and_cont_cols(df, num_unique=10):
    '''
    Identifies columns from a df as continuous or categorical
    based on the number of unique values for each column.
    Returns a list of categorical columns and a list of continuous columns
    '''
    # store column in categorical list if there are `num_unique` or less unique values
    cat_cols = [col for col in df.columns if len(df[col].unique()) <= 10]
    # store column in categorical list if there are more than `num_unique` unique values 
    cont_cols = [col for col in df.columns if len(df[col].unique()) > 10]
    
    # print continous columns that are objects
    for col in cont_cols:
        if df[col].dtype == 'O':
            print(f'{col} is continuous but not numeric. Check if column needs to be cleaned')
    
    return cat_cols, cont_cols


def plot_heatmap(df):
    '''
    Plots heatmap of DataFrame correlations
    '''
    plt.figure(figsize=(len(df.columns), len(df.columns) * .6))
    
    mask = np.triu(np.ones_like(df.corr().iloc[1:,:-1]),k=1)
    sns.heatmap(df.corr().iloc[1:,:-1], mask=mask, linewidths=.5, annot=True,
                         cmap='RdYlGn', vmin=-1, vmax=1, square=True)
    plt.show()

    
def explore_univariate_categorical_cols(df, cat_cols = None):
    '''
    Explores categorical features
    Plots bar charts of each categorical features
    '''
    
    # set default categorical columns
    if cat_cols == None:
        cat_cols = get_cat_and_cont_cols(df)[0]
    
    for col in cat_cols:
        print(col.upper())
        # Combine count and normalized frequency into a single DataFrame
        frequency_table = pd.concat([df[col].value_counts(), 
                                     df[col].value_counts(normalize=True)], axis=1).reset_index()
        frequency_table.columns = [col,'Count', 'Frequency']
        display(frequency_table)
        # bar plot
        plt.figure(figsize=(3, 2))
        sns.countplot(x=col, data=df)
        # Annotate the bars
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom')
        plt.show()
        print()
        
        
def explore_univariate_continuous_cols(df, cont_cols = None):
    '''
    Explores continuous features
    Plots histograms and boxplots of each continuous features
    '''
    
    # set default categorical columns
    if cont_cols == None:
        cont_cols = get_cat_and_cont_cols(df)[1]
    
    # descriptive stats
    print('Descriptive Stats:\n')
    display(df[cont_cols].describe())

    for col in cont_cols:
        print('-'*60, '\n', col.upper(), '\n')
        # most frequent values
        print('Most Frequent Values:')
        print(df[col].value_counts().head(3))
        # set figure
        fig, axes = plt.subplots(1, 2, figsize=(6, 2))
        # histogram
        sns.histplot(x=col, data=df, ax=axes[0])
        # boxplot
        sns.boxplot(x=col, data=df, ax=axes[1])

        plt.show()
        print()

        
def explore_bivariate_cont_to_cat_target(df, target, cont_cols=None):
    '''
    Explores continuous feature relationships to categorical target
    Provides descriptive stats for each target category
    Shows continuous feature correlations for two-category target
    Plots bar chart of feature averages for each target category
    '''

    # set default categorical columns
    if cont_cols == None:
        cont_cols = get_cat_and_cont_cols(df)[1]
        
    # display descriptive stats for each target category
    display(df.groupby(target)[cont_cols].describe().T)
    
#     # display, in order, pearson R correlations to the target if target is binary
#     if len(df[target].unique()) == 2:
#         print(f'Continuous feature correlations (Pearson R) to {target}:')
#         display(df[cont_cols+[target]].corr()[target]\
#                                       .sort_values(ascending=False))

    for col in cont_cols:
        plot_bivariate_cont_to_cat_target_charts(df, target, col)
#         plt.figure(figsize=(3, 3)) 
#         sns.barplot(x=target, y=col, data=df, estimator='mean')
#         # Annotate the bars
#         ax = plt.gca()
#         for p in ax.patches:
#             ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()),
#                         ha='center', va='bottom')
#         plt.title(f'{col} averages')
#         # add line indicating estimate of all targets
#         plt.axhline(df[col].mean(), label=f'Total {col} mean', color='red')
#         # plt.legend()
#         plt.show()
#         print()
        
#     plot_heatmap(df[cont_cols+[target]])
    





def explore_bivariate_cat_to_cat_target(df, target, cat_cols=None):
    '''
    Explores categorical feature relationships to categorical target
    Shows continuous feature "correlations" for two-category target
    Plots bar chart of target frequencies for each categorical feature category
    '''
    # set default categorical columns
    if cat_cols == None:
        cat_cols = get_cat_and_cont_cols(df)[0]
        
#     # display, in order, pearson R "correlations" to the target if target is binary
#     if len(df[target].unique()) == 2:
#         print(f'Categorical feature (integer-type) "correlations" (Pearson R) to {target}:')
#         display(df[cat_cols].corr(numeric_only=True)[target]\
#                                   .sort_values(ascending=False))

    for col in cat_cols:
        plot_bivariate_cat_to_cat_target_charts(df, target, col)
#         plt.figure(figsize=(3, 3)) 
#         sns.barplot(x=col, y=target, data=df, estimator='mean')
#         # Annotate the bars
#         ax = plt.gca()
#         for p in ax.patches:
#             ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()),
#                         ha='center', va='bottom')
#         # plt.title(f'{target} averages')
#         # add line indicating estimate of all targets
#         plt.axhline(df[target].mean(), label=f'Total {target} mean', color='red')
#         # plt.legend()
#         plt.show()
#         print()
        
#     plot_heatmap(df[cat_cols+[target]])


def plot_bivariate_cont_to_cat_target_charts(df, target, col):
    sns.set_style('darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    plt.subplots_adjust(hspace=.3, wspace=0.4)
    
    col_label = ' '.join([word.capitalize() for word in col.split('_')])

    # first plot
    sns.barplot(x=target, y=col, data=df, errorbar=None, ax=axes[0,0])
    # Annotate the bars
    for p in axes[0,0].patches:
        axes[0,0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='bottom')
    axes[0,0].set_title(f'''{col_label} Averages''')
    axes[0,0].set_xlabel(target.capitalize())
    axes[0,0].set_xticks([0, 1], ['No', 'Yes'])
    axes[0,0].set_ylabel(col_label)
#     axes[0,0].set_yticks(range(0,
#                                int(max(df[df[target]==1][col].mean(), df[df[target]==0][col].mean()))+25,
#                                5))
#     # add line indicating total tenure average
#     axes[0,0].axhline(df[col].mean(), label=f'Total {col_label} mean', color='red')
#     axes[0,0].annotate(f'''{df[col].mean():.2f}''', xy=(.5, df[col].mean()),
#                         ha='center', va='bottom')
    # Display the legend
#     axes[0,0].legend(loc='upper right', edgecolor='black')

    # second plot
    sns.stripplot(data=df, x=target, y=col, hue=target, jitter=.3,
                  size=1.5, ax=axes[0,1], legend=False)
    axes[0,1].set_title(f'''{col_label} by {target.capitalize()}''')
    axes[0,1].set_xlabel(target.capitalize())
    axes[0,1].set_xticks([0, 1], ['No', 'Yes'])
    axes[0,1].set_ylabel(col_label)

    plt.xticks()


    # third plot
    sns.histplot(hue=target, x=col, data=df[df[target]==0], ax=axes[1,0], legend=False)
    axes[1,0].set_title(f'''{col_label} Distribution of Non-{target.capitalize()}''')
    axes[1,0].set_xlabel(target.capitalize())

    # fourth plot
    sns.histplot(hue=target, x=col, data=df[df[target]==1], ax=axes[1,1],
                 palette=['orange'], legend=False)
    axes[1,1].set_title(f'''{col_label} Distribution of {target.capitalize()}''')
    axes[1,1].set_xlabel(target.capitalize())

    plt.show()



def plot_bivariate_cat_to_cat_target_charts(df, target, col):
    sns.set_style('darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))

    col_label = ' '.join([word.capitalize() for word in col.split('_')])

    # first plot
    order = df.groupby(col)[target].mean().sort_values(ascending=False).index
    sns.barplot(x=col, y=target, data=df, errorbar=None,  order=order,
                ax=axes[0])
    # Annotate the bars
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    axes[0].set_title(f'''{target.capitalize()} Rates by {col_label}''')
    axes[0].set_xlabel(col_label)
    axes[0].set_ylabel(f'''{target.capitalize()} Rate''')
#     axes[0].set_yticks(np.arange(0, .46, 0.05))
    # add line indicating estimate of all targets
#     axes[0].axhline(telco[target].mean(), label=f'''Total Average {target.capitalize()} Rate''', color='red')
#     axes[0].annotate(f'''{telco[target].mean():.2f}''', xy=(1, telco[target].mean()),
#                          ha='center', va='bottom')
    # Display the legend
#     axes[0].legend(loc='upper right', edgecolor='black')

    # second plot
    max_count = max([df[(df[col]==val)&(df[target]==0)][target].count() for val in df[col].unique()] +\
            [df[(df[col]==val)&(df[target]==1)][target].count() for val in df[col].unique()]) + 50
    sns.heatmap(pd.crosstab(df[target], df[col])[order],
                cmap='Reds', linewidths=.5, annot=True, square=True, fmt='.0f', cbar=False,
                ax=axes[1], vmin=0, vmax=max_count)
    axes[1].set_title(f'''{target.capitalize()} By {col_label}''')
    axes[1].set_xlabel(col_label)
    axes[1].set_ylabel(target.capitalize())
#     axes[1].set_yticklabels(['No', 'Yes'], ha='center')
    plt.show()

def explore_bivariate_cont_to_cont_target(df, target, cont_cols=None):
    '''
    Explores categorical feature relationships to continuous target
    '''
    if cont_cols == None:
        cont_cols = get_cat_and_cont_cols(df)[1]
    
    sns.pairplot(data=df[cont_cols], kind='reg', corner=True,
                 plot_kws={'scatter_kws':{'s':1, 'alpha':.5},
                           'line_kws':{'linewidth':1, 'alpha':.5, 'color':'red'}})
    plt.show()
    
    plt.figure(figsize=(len(train.columns), len(train.columns) * .6))

    mask = np.triu(np.ones_like(df.corr().iloc[1:,:-1]),k=1)
    sns.heatmap(df.corr().iloc[1:,:-1], mask=mask, linewidths=.5, annot=True,
                         cmap='RdYlGn', vmin=-1, vmax=1, square=True)
    plt.show()
    
    sns.heatmap(df.corr()[target].sort_values(ascending=False).to_frame(),
            linewidths=.5, annot=True, cmap='RdYlGn',
            vmin=-1, vmax=1, square=True)
    plt.show()


def explore_bivariate_cat_to_cont_target(df, target, cat_cols=None):
    '''
    Explores categorical feature relationships to continuous target
    Provides descriptive stats for each feature category
    '''
#     if cat_cols == None:
#         cat_cols = get_cat_and_cont_cols(df)[0]
        
    for col in cat_cols:
        print(f'{col} group {target} stats')
        display(df.groupby(col)[target].describe().T)
        
        plot_bivariate_cat_to_cont_target_charts(df, target, col)
        

def plot_bivariate_cat_to_cont_target_charts(df, target, col):
    print(f'{col} group {target} averages')
    sns.barplot(data=df, x=col, y=target, errorbar=None)
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{round(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    plt.show()

    print(f'{col} group {target} distributions')
    sns.displot(data=df, x=target, col=col, hue=col)
    plt.show()
    sns.stripplot(data=df, x=col, y=target, hue=col, jitter=.3,
                  size=1.5, legend=False)
    plt.show()

    print(f'{col} group {target} boxplots')
    sns.boxplot(data=df, x=col, y=target)
    plt.show()