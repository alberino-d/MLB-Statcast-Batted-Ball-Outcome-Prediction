# Perform EDA

# import libraries
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
    'sprint_speed',
    'bb_type'
]

hits = [
    'single',
    'double',
    'triple',
    'home_run'
]

# create outcomes column
outcomes = ['out', 'single', 'double', 'triple', 'home_run']

bbdf['outcome'] = np.where(
    bbdf['events'].isin(outcomes),
    bbdf['events'],
    'out'
)


# describe numerical features
# print(bbdf[features].describe(include=[np.number]).round(2))


# describe categorical features
# print(bbdf[features + ['events']].describe(include=[object]))




# Hit v. Out EDA

# plot feature distributions

# numeric features
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

# categorical features
# fig, ax = plt.subplots(figsize=(5,3))

# sns.barplot(
#     bbdf.groupby('bb_type')['events'].apply(lambda x: x.isin(hits).mean()),
#     linewidth=1,
#     edgecolor='black',
#     color='lightblue'
# )

# for p in ax.patches:
#     h = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2,
#             h+0.02,
#             f'{h*100:.1f}%',
#             ha='center'
#     )

# plt.xlabel('Batted Ball Type')
# ax.set_xticks(['fly_ball', 'ground_ball', 'line_drive', 'popup'])
# ax.set_xticklabels(['Fly Ball', 'Ground Ball', 'Line Drive', 'Popup'])

# plt.ylim(0, 0.7)
# plt.ylabel('Hit Percentage')
# plt.yticks([])

# fig.suptitle('Pct. of Batted Balls Resulting in Hits by Type')

# plt.show()

# print(plot_feature_dist('launch_speed'))
# print(plot_feature_dist('launch_angle'))
# print(plot_feature_dist('spray_angle'))
# print(plot_feature_dist('sprint_speed'))


# create feature relationship graphs
def plot_feature_interaction(feature_A, feature_B):
    # fix column name
    featureA_name = feature_A.replace('_', ' ').title()
    featureB_name = feature_B.replace('_', ' ').title()

    fig, ax = plt.subplots(figsize=(10, 10))

    # batting average by feature bucket
    pivot = bbdf.pivot_table(
        values='is_hit',
        index=pd.cut(bbdf[feature_A], bins=20),
        columns=pd.cut(bbdf[feature_B], bins=20),
        aggfunc='mean',
        observed=True
    )

    # rename columns/rows
    pivot.columns = [np.round(i.mid, 1) for i in pivot.columns]
    pivot.index = [np.round(i.mid, 1) for i in pivot.index]

    # heatmap
    sns.heatmap(pivot.iloc[::-1], cmap='coolwarm', annot=True, fmt='.2f', cbar=False)

    # customize title/axes/ticks
    plt.title(f'Batting Average by {featureA_name}/{featureB_name} Bin', fontsize=18)
    plt.xlabel(f'{featureB_name}')
    plt.ylabel(f'{featureA_name}')
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.show()

# plot_feature_interaction('launch_speed', 'launch_angle')
# plot_feature_interaction('launch_speed', 'spray_angle')
# plot_feature_interaction('launch_angle', 'spray_angle')
# plot_feature_interaction('launch_speed', 'sprint_speed')
# plot_feature_interaction('launch_angle', 'sprint_speed')
# plot_feature_interaction('spray_angle', 'sprint_speed')









# Hit Outcome EDA

# plot outcome frequencies

# sns.countplot(
#     data=bbdf,
#     x='outcome',
#     order=[
#         'out',
#         'single',
#         'double',
#         'triple',
#         'home_run'
#     ]
# )

# plt.title('Distribution of Outcomes')
# plt.xlabel('BB Outcome')
# plt.ylabel('Number of Batted Balls')

# plt.show()


# plot features by outcome
def plot_feature_by_outcome(feature):
    feature_name = feature.replace('_', ' ').title()

    sns.boxplot(
        data=bbdf,
        x="outcome",
        y=feature,
        order=[
            "out",
            "single",
            "double",
            "triple",
            "home_run"
        ]
    )

    plt.title(f"{feature_name} by BB Outcome")

    plt.xlabel("BB Outcome")
    plt.ylabel(f"{feature_name}")

    plt.show()

# plot_feature_by_outcome('launch_speed')
# plot_feature_by_outcome('launch_angle')
# plot_feature_by_outcome('spray_angle')
# plot_feature_by_outcome('sprint_speed')


# launch speed/angle per outcome
# sns.scatterplot(
#     data=bbdf,
#     x="launch_speed",
#     y="launch_angle",
#     hue="outcome",
#     alpha=0.5
# )

# plt.title(
#     "Exit Velocity vs. Launch Angle by BB Outcome"
# )

# plt.xlabel("Exit Velocity")
# plt.ylabel("Launch Angle")

# plt.show()
