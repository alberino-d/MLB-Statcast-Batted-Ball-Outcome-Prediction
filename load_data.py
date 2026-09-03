# Get/Create Statcast Datasets

# import libraries
import pandas as pd
from pybaseball.datahelpers.statcast_utils import add_spray_angle
from pybaseball import statcast


# get statcast data
df = statcast(
    start_dt='2025-03-27',
    end_dt='2025-09-28'
)

# add sprint speed
sprint_df = pd.read_csv('sprint_speed.csv')
sprint_data = sprint_df[["player_id", "sprint_speed"]]

df = df.merge(
    sprint_data,
    how="left",
    left_on="batter",
    right_on="player_id"
)
df.drop(columns="player_id", inplace=True)
df.dropna(subset=['sprint_speed'])


# create batted balls dataset
batted_balls = df[df['bb_type'].notna()]
batted_balls = add_spray_angle(batted_balls)

hit_outcomes = ['single', 'double', 'triple', 'home_run']
batted_balls['is_hit'] = (
    batted_balls['events'].isin(hit_outcomes)
).astype(int)

batted_balls.to_csv('batted_balls.csv', index=False)


# create hits dataset
hits = df[df['events'].isin(hit_outcomes)].copy()
hits = add_spray_angle(hits)

hits.to_csv('hits.csv', index=False)
