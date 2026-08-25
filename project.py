from pybaseball.datahelpers.statcast_utils import add_spray_angle
from pybaseball import statcast

df = statcast(
    start_dt="2025-03-27",
    end_dt="2025-09-28"
)

batted_balls = df[~df["bb_type"].isna()]
batted_balls = add_spray_angle(batted_balls)

hit_outcomes = ["single", "double", "triple", "home_run"]
batted_balls["hit"] = (
    batted_balls["events"].isin(hit_outcomes)
).astype(int)

batted_balls.to_csv("batted_balls.csv", index=False)
