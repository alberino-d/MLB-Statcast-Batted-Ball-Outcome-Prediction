# Perform EDA

# import libraries
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# create DataFrame from batted ball data
bbdf = pd.read_csv('batted_balls.csv')

features = [
    'launch_speed',
    'launch_angle',
    'spray_angle',
    'bb_type'
]

hits = [
    'single',
    'double',
    'triple',
    'home_run'
]


# describe numerical features
# print(bbdf[features].describe(include=[np.number]).round(2))


# describe categorical features
# print(bbdf[features + ['events']].describe(include=[object]))


# Plot Feature Distributions

# Numeric Features
def plot_feature_dist(feature):
    # fix column name
    feature_name = feature.replace('_', ' ').title()

    fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=2)

    # create histogram
    sns.histplot(
        x=bbdf[feature],
        hue=bbdf['is_hit'],
        bins=30,
        palette={0:'steelblue', 1:'indianred'},
        ax=axes[0]
    )

    # create violin plot
    sns.violinplot(
        x=bbdf[feature],
        hue=bbdf['is_hit'],
        split=True,
        palette={0:'steelblue', 1:'indianred'},
        ax=axes[1]
    )

    # create legend
    handles, _ = plt.gca().get_legend_handles_labels()
    axes[0].legend(handles, ['Out', 'Hit'], title='Outcome')
    axes[1].legend(handles, ['Out', 'Hit'], title='Outcome')

    # create axis
    axes[0].set_xlabel(feature_name)
    axes[1].set_xlabel(feature_name)
    fig.suptitle(f'{feature_name} Distribution by Hit Outcome')

    plt.show()

# Categorical Features
fig, ax = plt.subplots(figsize=(5,3))

sns.barplot(
    bbdf.groupby('bb_type')['events'].apply(lambda x: x.isin(hits).mean()),
    linewidth=1,
    edgecolor='black',
    color='lightblue'
)

for p in ax.patches:
    h = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            h+0.02,
            f'{h*100:.1f}%',
            ha='center'
    )

plt.xlabel('Batted Ball Type')
ax.set_xticks(['fly_ball', 'ground_ball', 'line_drive', 'popup'])
ax.set_xticklabels(['Fly Ball', 'Ground Ball', 'Line Drive', 'Popup'])

plt.ylim(0, 0.7)
plt.ylabel('Hit Percentage')
plt.yticks([])

fig.suptitle('Pct. of Batted Balls Resulting in Hits by Type')

plt.show()

# print(plot_feature_dist('launch_speed'))
# print(plot_feature_dist('launch_angle'))
# print(plot_feature_dist('spray_angle'))
