# %% [markdown]
# # Imports & Constants

# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import numpy as np
import shutil
import time
import json
from time import sleep
from datetime import datetime
import re
import seaborn as sns
from loguru import logger


def extract_time_format_from_datetime(formatted_time):
    if("T" in formatted_time):
      formatted_time = formatted_time.split("+")[0]
      formatted_time = formatted_time.replace("T", " ")
      return "%Y-%m-%dT%H:%M:%S"
    return "%Y-%m-%d %H:%M:%S.%f"
def convert_datetime_to_time_since_last_epoch(formatted_time):
    time_format = extract_time_format_from_datetime(formatted_time)
    time_struct = time.strptime(formatted_time, time_format)
    datetime_obj = datetime.fromtimestamp(time.mktime(time_struct))
    return datetime_obj.timestamp()

def find_most_recent_directory(dir):
    """
        This will do this by seeing the most recent directory in the directory, looking at the directory date
    """
    dirs = os.listdir(dir)
    for i in range(len(dirs)):
        if not os.path.isdir(dir + dirs[i]):
            dirs.pop(i)

    most_recent_dir = dirs[0]
    for _dir in dirs:
        if os.path.getmtime(dir + _dir) > os.path.getmtime(dir + most_recent_dir):
            most_recent_dir = _dir

    return dir + most_recent_dir + "/"

# Most recent directory
galaxy_watch_folder_dirs = os.listdir("./data/galaxywatch/")
galaxy_watch_folder_dirs.sort()
galaxy_watch_folder_dir = "./data/galaxywatch/"+galaxy_watch_folder_dirs[-1] + "/"
exercise_logs_folder_dir = "./data/exerciselogs/"
computer_usage_folder_dir = "./data/computerusage/"
calendar_events_folder_dir = find_most_recent_directory("./data/calendar/")


# No data filter
# data_start_time = 0

# First day of college 
data_end_time = 9e50
data_start_time = 0

# 3 weeks out to a good time
data_start_time = round(convert_datetime_to_time_since_last_epoch("2023-8-5 00:00:00.0"))
data_end_time = round(convert_datetime_to_time_since_last_epoch("2024-5-17 0:0:00.0"))

# testing celander
# data_start_time = round(convert_datetime_to_time_since_last_epoch("2023-10-23 00:00:00.0"))
# data_end_time = round(convert_datetime_to_time_since_last_epoch("2023-10-30 00:00:00.0"))


if data_end_time >= 9e20:
  data_end_time = time.time()

current_date = time.time()

# Create a common dataframe that holds information about stuff I do every day, it will be indexed by a date and it will hold all of the dates from data_start_time to dtata_end_time
daily_df = pd.DataFrame()
daily_df["date"] = pd.date_range(start=datetime.fromtimestamp(data_start_time), end=datetime.fromtimestamp(data_end_time), freq="D")
daily_df["date"] = daily_df["date"].dt.date
daily_df = daily_df.set_index("date")
daily_df.index = pd.to_datetime(daily_df.index)

# Excluse some dates from the daily range (the ones that we don't want to care about)
dates_to_exclude = []
for date in dates_to_exclude:
  daily_df = daily_df.drop(pd.to_datetime(date).date())

# %% [markdown]
# # Data Management

# %%
# Check to see if there is any new data (from my android phone)
# Define the source and destination directories

src_dir = "/run/user/1000/gvfs/mtp:host=motorola_motorola_one_5G_ace_ZY22DDHW4G/Internal shared storage/Download/Samsung Health"
dest_dir = "./data/galaxywatch"
if os.path.exists(src_dir):
    # Loop through all the directories in the source directory
    for dir_name in os.listdir(src_dir):
        # Check if the item is a directory
        if os.path.isdir(os.path.join(src_dir, dir_name)):
            # Check if the directory already exists in the destination directory
            if not os.path.exists(os.path.join(dest_dir, dir_name)):
                # If it doesn't exist, copy it over
                shutil.copytree(os.path.join(src_dir, dir_name), os.path.join(dest_dir, dir_name))

    dest_dir = "./data/galaxywatch/"
    for dir_name in os.listdir(dest_dir):
        folder_dir = f"{dest_dir}/{dir_name}/"

        # Do some cleaning on the folder data
        date_time_of_upload = folder_dir.split("_")[-1][:-1]

        # Rename all of the files so that they are easier to deal with
        for file_name in os.listdir(folder_dir):
            if os.path.isdir(folder_dir + file_name):
                continue
            if "com.samsung" not in file_name:
                continue
            new_file_name = file_name.replace(".".join(file_name.split(
                ".")[:3]) + ".", "").replace(date_time_of_upload + ".", "")
            os.rename(folder_dir + file_name, folder_dir + new_file_name)

        # Clean csv's so that they dont have that first line with junk
        for file_name in os.listdir(folder_dir):
            if os.path.isdir(folder_dir + file_name):
                continue
            file_text = ""
            with open(folder_dir + file_name, "r") as f:
                file_text = f.read()

                if "com.samsung" in (file_text.split("\n")[0]):
                    file_text = "\n".join(file_text.split("\n")[1:])
                    with open(folder_dir + file_name, "w") as f:
                        f.write(file_text)

else:
    # See if its been more than 2 months since health data has been updated to the computer
    is_older_than_2_months = lambda dir: (time.time() - os.path.getmtime(dir) > 60 * 60 * 24 * 30 * 2)
    if all(os.listdir(dest_dir)):
        logger.warning("It's been a long time since you uploaded data, you should consider uploading again!")

    


# %%
# Clean up the data a but

# %% [markdown]
# # Helper functions

# %%

def list_intersection(list_1, list_2) -> list:
    return list(set(list_1) & set(list_2))
def union_dataframes(dataframe_1, dataframe_2, on_column="date"):
    for column in dataframe_2.columns:
        if column not in dataframe_1.columns:
            dataframe_1[column] = dataframe_2[column]
        else:
           print(f"Warning: {column} already exists in dataframe_1")
    return dataframe_1

def compute_correlation(df_1, df_2, x_axis, y_axies):
    # Convert the "date" column to a datetime type
    if x_axis == "date":
        df_1[x_axis] = pd.to_datetime(df_1[x_axis])
        df_2[x_axis] = pd.to_datetime(df_2[x_axis])

    # Merge the sleep and computer usage DataFrames on the date column
    if x_axis != "index":
        merged_df = pd.merge(df_1, df_2, on=x_axis)
    else:
        merged_df = pd.concat([df_1, df_2])

    # Compute the correlation between "last_time_used" and "sleep_amount"
    correlation = merged_df[y_axies[0]].corr(merged_df[y_axies[1]])
    return correlation


def average_series(series_1, series_2):
    new_series = pd.Series(dtype=pd.Float64Dtype)
    for column in series_1.index:
        if type(series_1[column]) == str:
            continue
        new_series[column] = (series_1[column] + series_2[column]) / 2
    return new_series

# Define a function to find the closest weight entry in fitdays_data for a given row in shealth_data

def average_measurements_that_are_close_in_time(dataframe: pd.DataFrame, time_column_name, group_1_function, group_2_function, time_difference_threshold, mode="mean"):
    # Calculate the time difference between the current row and all rows in fitdays_data

    group_1_dataframe = group_1_function(dataframe)
    group_2_dataframe = group_2_function(dataframe)

    group_1_dataframe = group_1_dataframe.sort_values(by=time_column_name)
    group_2_dataframe = group_2_dataframe.sort_values(by=time_column_name)

    new_dataframe = pd.DataFrame(columns=group_1_dataframe.columns)

    # for each entry in the group_1_dataframe, find the neastest entry in time
    for (index, row) in group_1_dataframe.iterrows():
        candidates = (group_2_dataframe[abs(
            group_2_dataframe[time_column_name] - row[time_column_name]) < time_difference_threshold])
        if candidates.shape[0] == 0:
            continue

        match mode:
            case "mean":
                another_row = average_series(candidates.iloc[0], row)
                new_dataframe = pd.concat(
                    [new_dataframe, another_row.to_frame().T], ignore_index=True)

    return new_dataframe


def plot_variables(df, x_axis_column_name, y_axis_column_names, y_axis_visible, title, image_name="plot.png", convert_to_date=False):

  if (len(y_axis_column_names) < 2):
     print("Warning: only the you have only passed in 1 variable to plot")

  color_hex_codes = ['#FF5733', '#40E0D0', '#4169E1', '#32CD32',
      '#FFD700', '#DA70D6', '#FF7F50', '#708090', '#FF1493', '#008B8B']

  axes = []
  fig, ax1 = plt.subplots(figsize=(16, 9))
  ax2 = ax1.twinx()

  axes = [ax1, ax2]

  if len(y_axis_column_names) > 2:
    for i in range(2, len(y_axis_column_names)):
      axes.append(axes[i % 2].twinx())
  elif len(y_axis_column_names) == 2:
    # Label the axes with the two different column names
    axes[0].set_ylabel(
        convert_snake_case_to_pascal_case(y_axis_column_names[0]))
    axes[1].set_ylabel(
        convert_snake_case_to_pascal_case(y_axis_column_names[1]))
  else:
     axes[0].set_ylabel(
         convert_snake_case_to_pascal_case(y_axis_column_names[0]))
  for ax, column_name in zip(axes, y_axis_column_names):
    random.seed(69)
    color = random.choice(color_hex_codes)
    color_hex_codes.remove(color)
    print(column_name, " is ", color)
    ax.plot(df[x_axis_column_name], df[column_name],
            label=column_name, color=color)

    ax.get_yaxis().set_visible(y_axis_visible)
    plt.legend()

  if convert_to_date or "date" in x_axis_column_name:
    convert_time_since_last_epoch_x_axis_to_date(ax)
  plt.title(title)
  savefig(title)


def savefig(title="", image_name="plot.png", type_="plot"):
    if title == "" and image_name == "plot.png":
      raise Exception("You must pass in a title or an image name")
    if title != "":
      image_name = type_ + "_for_" + to_snake_case(title)+ ".png"
    plt.savefig(f"./figs/{image_name}")
    

def convert_time_since_last_epoch_to_date(time_since_last_epoch):
    time_struct = time.localtime(time_since_last_epoch)
    datetime_obj = datetime.fromtimestamp(time.mktime(time_struct))
    return datetime_obj.strftime("%Y/%m/%d")

def convert_snake_case_to_pascal_case(snake_case_string):
    return "".join([word.capitalize() for word in snake_case_string.split("_")])

def extract_time_format_from_datetime(formatted_time:str):
    if formatted_time.count(":") < 1:
      return "%Y-%m-%d"
    elif "T" in formatted_time:
      formatted_time = formatted_time.split("+")[0]
      formatted_time = formatted_time.replace("T", " ")
      return "%Y-%m-%dT%H:%M:%S"
    return "%Y-%m-%d %H:%M:%S.%f"
def convert_datetime_to_time_since_last_epoch(formatted_time):
    formatted_time = formatted_time.replace("/", "-")
    time_format = extract_time_format_from_datetime(formatted_time)
    time_struct = time.strptime(formatted_time, time_format)
    datetime_obj = datetime.fromtimestamp(time.mktime(time_struct))
    return datetime_obj.timestamp()
def apply_conversion_to_muliple_columns(dataframe, column_names, conversion: float):
    if type(conversion) == list:
        raise Exception("muliptle conversions not implemented yet")

    for column in column_names:
        dataframe[column] = dataframe[column].apply(lambda x: float(x) * conversion)
    return dataframe

def convert_time_since_last_epoch_x_axis_to_date(ax):
  labels = ax.get_xticklabels()
  # Define the new x-axis tick labels
  new_labels = [convert_time_since_last_epoch_to_date(float(label.get_text()) * 1e9) for label in labels]
  
  # Set the new x-axis tick labels
  ax.set_xticklabels(new_labels)

def convert_utc_offset_to_hours(utc_offset):
  return int(utc_offset.split("-")[1][:2])

def remove_columns(dataframe, columns):
  return dataframe.drop(columns=columns)

weekday_int_to_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def convert_datetime_to_weekday(dataframe, column_name, weekday_int_to_day = False):
  if(weekday_int_to_day):
    dataframe["day_of_the_week"] = dataframe[column_name].apply(lambda x: weekday_int_to_day[datetime.fromtimestamp(x).weekday()])
  else:
    dataframe["day_of_the_week"] = dataframe[column_name].apply(lambda x: datetime.fromtimestamp(x).weekday())
  return dataframe

def to_snake_case(string):
  return "_".join(string.split(" ")).lower()

def clamp_dataframe_time(dataframe, time_name, start_time, end_time):
  return dataframe[(dataframe[time_name] >= start_time) & (dataframe[time_name] <= end_time)]

def plot_box_and_whisker(dataframe: pd.DataFrame, x_axis_column_name, y_axis_column_name, title, x_axis_label, y_axis_label, image_name="box_and_whisker.png"):
  if(image_name == "box_and_whisker.png"):
    image_name = image_name.replace(".png", "_for_") + to_snake_case(title) + ".png" 
  axes = dataframe.boxplot(column=y_axis_column_name, by=x_axis_column_name, return_type='axes', figsize=(16, 9))
  axes = (axes[y_axis_column_name])
  
  axes.set_xlabel(x_axis_label)
  axes.set_ylabel(y_axis_label)
  axes.set_title(title)

  # Get the current x-axis tick labels
  # labels = axes.get_xticklabels()
  if x_axis_column_name == "day_of_the_week":
    new_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Set the new x-axis tick labels
    axes.set_xticklabels(new_labels)

  plt.savefig(f"./figs/{image_name}")



def compute_r_squared(x, y, m, b):
  # Compute the residuals
  y_pred = m*x + b
  residuals = y - y_pred

  # Compute the mean of the residuals
  mean_residuals = np.mean(residuals)

  # Compute the total sum of squares
  total_sum_squares = np.sum((y - mean_residuals)**2)

  # Compute the residual sum of squares
  residual_sum_squares = np.sum(residuals**2)

  # Compute the R-squared
  r_squared = 1 - (residual_sum_squares / total_sum_squares)
  return r_squared

# Plot the line of best fit for the data
def plot_line_of_best_fit(dataframe, x_axis_column_name, y_axis_column_name, title, x_axis_label, y_axis_label, image_name = "line_of_best_fit.png"):
  if(image_name == "line_of_best_fit.png"):
    image_name = image_name.replace(".png", "_for_") + to_snake_case(title) + ".png" 
  x = dataframe[x_axis_column_name]
  y = dataframe[y_axis_column_name]
  m, b = np.polyfit(x, y, 1)
  fig, ax = plt.subplots(figsize=(16, 9))
  ax.set_title(title)
  ax.set_xlabel(x_axis_label)
  ax.set_ylabel(y_axis_label)
  ax.plot(x, y, '.', label="data")
  ax.plot(x, m*x + b, '-', label="line of best fit")
  print(m)
  # label the slope of the line in minutes/day
  # Calculate the position of the annotation
  x_pos = 0.5  # Adjust this value to move the annotation along the x-axis
  y_pos = m * x_pos + b  # This will place the annotation on the line of best fit

  # Create the annotation
  ax.annotate(f"{round(m * 3600 * 24 * 60, 5)} minutes/day", 
              xy=(x_pos, y_pos), 
              xytext=(20, 20), 
              textcoords='offset points', 
              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
              fontsize=12, 
              color='red', 
              ha='center')


  if "date" not in x_axis_column_name:
    convert_time_since_last_epoch_x_axis_to_date(ax)
  ax.legend()
  plt.savefig(f"./figs/{image_name}")


def do_the_rudamentary_time_changes(dataframe: pd.DataFrame, use_end_time_for_date = False):
    columns_to_convert_to_time_since_last_epoch = ["start_time", "end_time"]

    if "time_offset" in dataframe.columns:
      dataframe["time_offset"] = convert_utc_offset_to_hours(dataframe["time_offset"].iloc[0])
    else:
      dataframe["time_offset"] = 0
    for column in columns_to_convert_to_time_since_last_epoch:
        if column in dataframe.columns and type(dataframe[column].iloc[0]) == str:
          dataframe[column] = dataframe[column].apply(convert_datetime_to_time_since_last_epoch) - dataframe["time_offset"] * 3600

    dataframe = dataframe.sort_values(by="start_time")
    if "end_time" in dataframe.columns:
      dataframe = convert_datetime_to_weekday(dataframe=dataframe, column_name="end_time")
    elif "start_time" in dataframe.columns:
      dataframe = convert_datetime_to_weekday(dataframe=dataframe, column_name="start_time")
      
    print(dataframe.head(5))
    dataframe = clamp_dataframe_time(dataframe, "start_time", data_start_time, data_end_time)
    if not use_end_time_for_date:
      dataframe["date"] = dataframe["start_time"].apply(convert_time_since_last_epoch_to_date)
    else:
      dataframe["date"] = dataframe["end_time"].apply(convert_time_since_last_epoch_to_date)
    return dataframe

def sum_df_by_variable(df: pd.DataFrame, x_axis, y_axies, modes):
    # Group computer usage events by date
    grouped = df.groupby(df[x_axis])

    if type(y_axies) == str:
        y_axies = [y_axies]

    if type(modes) == str:
        modes = [modes]
    if len(y_axies) > len(modes) and len(modes) == 1:
        modes = modes * len(y_axies)
    elif len(y_axies) < len(modes) and len(y_axies) == 1:
        y_axies = y_axies * len(modes)
    elif len(y_axies) != len(modes):
        raise Exception("y_axies and modes must be the same length")

    # Create new DataFrame with the total durations
    result_df = pd.DataFrame({
        # x_axis: grouped.groups,
    })
    for axis, mode in zip(y_axies, modes):
        match mode:
            case "mean":
                result_df[axis] = grouped[axis].mean()
            case "sum":
                result_df[axis] = grouped[axis].sum()
            case "max":
                result_df[axis] = grouped[axis].max()
            case "min":
                result_df[axis] = grouped[axis].min()
            case "first":
                result_df[axis] = grouped[axis].first()
            case _:
                raise Exception("Invalid mode")
    # if ""
    #     "last_time_used" : grouped["start_time"].max()
    # result_df["start_of_the_day"] = result_df["date"].apply(lambda x: str(x) + " 00:00:00.00").apply(convert_datetime_to_time_since_last_epoch)
    # result_df["last_time_used"] = (result_df["last_time_used"] - result_df["start_of_the_day"])  / 3600
    return result_df


# %% [markdown]
# # Weight

# %%
# Steps
steps = pd.read_csv(galaxy_watch_folder_dir + "tracker.pedometer_day_summary.csv", index_col=False)
steps = steps.rename(columns={"create_time" : "start_time"})
steps["start_time"] = steps["start_time"].apply(convert_datetime_to_time_since_last_epoch)


steps = do_the_rudamentary_time_changes(steps)
steps = pd.DataFrame(steps.groupby("date")["step_count"].max()).reset_index()

print(daily_df.head(10))

steps = steps.set_index("date")
steps.index = pd.to_datetime(steps.index)
steps = steps.rename({"step_count" : "num_steps"})
daily_df = union_dataframes(daily_df, steps, "date")
print(steps.head(10))
print(daily_df.head(10))
exit()



# %%
weight = pd.read_csv(galaxy_watch_folder_dir + "weight.csv", index_col=False)
# filter weight

# weight = weight[weight["pkg_name"].isin(["cn.fitdays.fitdays"])]
weight = weight[pd.notna(weight["body_fat_mass"])]
weight["start_time"] = weight["start_time"].apply(convert_datetime_to_time_since_last_epoch)
weight = weight.rename(columns = {"total_body_water" : "water_mass"})
weight = apply_conversion_to_muliple_columns(weight, ["body_fat_mass", "weight", "muscle_mass", "skeletal_muscle_mass", "fat_free_mass", "water_mass"], 2.2)
# weight = clamp_dataframe_time(weight, "start_time", data_start_time, data_end_time)

weight = do_the_rudamentary_time_changes(weight)


# For every weight entry that has a package name of "com.sec.android.app.shealth" find the closest weight entry from cn.fitdays.fitdays, make sure they are less than 7200 seconds apart


weight.drop(["deviceuuid", "vfa_level", "deviceuuid", ], axis=1, inplace=True)
group_1_function = lambda df: df[df["pkg_name"] == "com.sec.android.app.shealth"]
group_2_function = lambda df: df[df["pkg_name"] == "cn.fitdays.fitdays"]
    
weight = average_measurements_that_are_close_in_time(weight, "start_time", group_1_function, group_2_function, 3600 * 12)

weight = weight.sort_values(by="start_time")
weight.sort_index(inplace=True)

weight["date"] = weight["start_time"].apply(convert_time_since_last_epoch_to_date)
weight = weight.set_index("date")
weight.index = pd.to_datetime(weight.index)

# add weight data to daily_df
daily_df = union_dataframes(daily_df, weight, "date")

# for the daily_df, figure out what the daily change in weight is per day, when there are long stretchs without data then use the next days change in weight to compute on average how much weight was lost
# ex. 11/10 - NaN
#     11/11 - NaN
#     11/12 - -3
# TURNS INTO:
# ex. 11/10 - -1
#     11/11 - -1
#     11/12 - -1
#

def imputate_missing_data(df, column_name):
  i = 0
  while i < df.shape[0]:
    # count number of instances of NaN
    j = i
    while j < df.shape[0] and pd.isna(df[column_name].iloc[j]):
      j += 1
    
    # if there are more than 0 NaN's in a row, then fill them in with the average of the next non-NaN value
    if j == i:
      i += 1
    else:
      for b in range(i ,j+1):
        try:
          df[column_name].iloc[b] = df[column_name].iloc[j] / (j - i+1)
        except:
          print("ERROR", b)
          pass
      i = j
  return df

daily_df["delta_weight"] = daily_df["weight"].diff()
daily_df["delta_muscle_mass"] = daily_df["muscle_mass"].diff()
daily_df["delta_fat_free_mass"] = daily_df["fat_free_mass"].diff()
daily_df["delta_water_mass"] = daily_df["water_mass"].diff()
daily_df["delta_body_fat_mass"] = daily_df["body_fat_mass"].diff()
daily_df["delta_skeletal_muscle_mass"] = daily_df["skeletal_muscle_mass"].diff()

daily_df = imputate_missing_data(daily_df, "delta_weight")
daily_df = imputate_missing_data(daily_df, "delta_muscle_mass")
daily_df = imputate_missing_data(daily_df, "delta_fat_free_mass")
daily_df = imputate_missing_data(daily_df, "delta_water_mass")
daily_df = imputate_missing_data(daily_df, "delta_body_fat_mass")
daily_df = imputate_missing_data(daily_df, "delta_skeletal_muscle_mass")

# %%
# Create a rolling average calculation using the previous 7 days worth of data. If there is not any data within those 7 days
# then interpolate the data from other points

def create_rolling_average_dataframe(df: pd.DataFrame):
  '''
  The purpose of this function is that it will make a dataframe that is a rolling average of the input dataframe, but it will do that using the 
  df["start_time"] as the time since last epoch to make the rolling average calculations. This is useful because the data is not evenly spaced out
  '''
  rolling_average_df = pd.DataFrame()
  rolling_average_df["start_time"] = df["start_time"]
  rolling_average_df["rolling_average"] = df["weight"]
  # Create the date column from the starttime column
  rolling_average_df["date"] = rolling_average_df["start_time"].apply(convert_time_since_last_epoch_to_date)
  rolling_average_df.set_index("date", inplace=True)
  rolling_average_df.index = pd.to_datetime(rolling_average_df.index)
  rolling_average_df = rolling_average_df.resample('D').mean()

  # Calculate the rolling average before interpolating
  rolling_average_df["rolling_average"] = rolling_average_df["rolling_average"].rolling(7, min_periods=1).median()

  # Then interpolate the missing values
  rolling_average_df["rolling_average"] = rolling_average_df["rolling_average"].interpolate()
  
  # fill in the start_times for the new rolling averages
  rolling_average_df["start_time"] = rolling_average_df.index.map(lambda x: convert_datetime_to_time_since_last_epoch(str(x) + ".00"))

  # Find the slope of the rolling average to figure out weight loss per day
  rolling_average_df["rolling_average_slope"] = rolling_average_df["rolling_average"].diff()
  rolling_average_df["rolling_average_slope"] = rolling_average_df["rolling_average_slope"].rolling(7, min_periods=1).mean()


  return rolling_average_df

rolling_average_df = create_rolling_average_dataframe(weight)
# rolling_average_df = rolling_average_df.dropna()
rolling_average_df = rolling_average_df.sort_values(by="start_time")
rolling_average_df = rolling_average_df.reset_index(drop=True)
rolling_average_df.head(5)
rolling_average_df.shape[0]

# %%
rolling_average_df
plot_line_of_best_fit(rolling_average_df, "start_time", "rolling_average", "Weight Loss", "Date", "Weight (lbs)")

# %%
plot_variables(rolling_average_df, "start_time", ["rolling_average", "rolling_average_slope"], True, title="Weight vs. Time", convert_to_date=True)

# %% [markdown]
# ## Sleepla

# %%
sleep_column_names = pd.read_csv(galaxy_watch_folder_dir + "sleep.csv", index_col=False, header=None, nrows=1).iloc[0].to_list  ()
# sleep_column_names = ["idk", "mental_recovery", *[f"factor{i}" for i in range(9)], "idk", "idk", "idk", "idk", "idk", "efficiency", "idk","idk", "idk", "idk", "physical_recovery", "idk", "idk", "idk", "start_time", "idk","idk","idk","idk", "sleep_cycle", "idk", "restfulness", "sleep_score", "sleep_duration", "idk", "idk", "idk","idk", "idk", "idk", "idk",  "time_offset", "idk", "idk", "idk", "end_time", "idk", "idk", "idk", "idk", "idk"]
# sleep_column_names = [x + ("{}".format(i) if "idk" in x else "") for i,x in enumerate(sleep_column_names)]

print(sleep_column_names)

sleep = pd.read_csv(galaxy_watch_folder_dir + "sleep.csv", index_col=False, header=None, skiprows=1,  names = sleep_column_names)
print(sleep.iloc[0])
print()
print()
print()
print(sleep.iloc[-3])
print()
print()
print()
print(sleep.iloc[-2])
print()
print()
print()
print(sleep.iloc[-1])
# add sleep column


columns_to_remove = ["comment", "datauuid", "custom","deviceuuid","pkg_name", "original_efficiency", "extra_data", "quality", "original_bed_time", "create_time", "update_time", "combined_id", "has_sleep_data", "sleep_type", "data_version", *["idk" + str(i) for i in range(1, 60)]]
remove_prefix = lambda x: x.replace("com.samsung.health.sleep.", "")

# Remove columns from sleep

# Rename the columns using the lambda function
sleep = sleep.rename(columns=remove_prefix)
sleep = sleep.drop(columns=list_intersection(columns_to_remove, sleep.columns))
print(sleep.head(5))
sleep["sleep_duration"] = sleep["sleep_duration"].apply(lambda x: float(x) / 60)
# Change starttime so its local timzone
sleep = do_the_rudamentary_time_changes(sleep, use_end_time_for_date=True)

daily_sleep = sum_df_by_variable(sleep, "date", ["sleep_duration", "day_of_the_week", "start_time", "sleep_score", "efficiency", "mental_recovery", "physical_recovery"], ["sum", "first", "mean", "mean", "mean", "mean", "mean"])
daily_sleep.index = pd.to_datetime(daily_sleep.index)

temp_daily_sleep = daily_sleep.copy()
# rename all columns of sleep to have sleep_ in the column name
for column in daily_sleep.columns:
  if column == "date" or "sleep" in column:
    continue
  temp_daily_sleep = temp_daily_sleep.rename(columns={column: "sleep_" + column})

# Now add every column in daily_sleep to daily_df if it doesn't already exist
union_dataframes(daily_df, temp_daily_sleep, "date")

# %%
# Plot box and whisker chart for day of the week and amount slept

plot_box_and_whisker(daily_sleep, "day_of_the_week", "sleep_duration", "Sleep Duration vs Day of the Week", "Day of the Week", "Sleep Duration (hours)")
plot_box_and_whisker(daily_sleep, "day_of_the_week", "sleep_score", "Sleep Score vs Day of the Week", "Day of the Week", "Sleep Score (0-100)")
plot_box_and_whisker(daily_sleep, "day_of_the_week", "efficiency", "Sleep Efficiency vs Day of the Week", "Day of the Week", "Sleep Efficiency (0-100)")
plot_box_and_whisker(daily_sleep, "day_of_the_week", "mental_recovery", "Mental Recovery vs Day of the Week", "Day of the Week", "Mental Recovery (0-100)")
plot_box_and_whisker(daily_sleep, "day_of_the_week", "physical_recovery", "Physical Recovery vs Day of the Week", "Day of the Week", "Physical Recovery (0-100)")

# %%
  
# plot_scatter(sleep, "com.samsung.health.sleep.start_time","sleep_duration")
plot_line_of_best_fit(daily_sleep, "start_time","sleep_duration", "Sleep Duration (hours) vs. Time", "Time", "Sleep Duration (hours)")


# %% [markdown]
# # Stress

# %%
stress = pd.read_csv(galaxy_watch_folder_dir + "stress.csv", index_col=False)
stress = stress.rename(columns=remove_prefix)
columns_to_remove = ["custom",	"binning_data",	"tag_id",	"create_time",	"algorithm",	"deviceuuid",	"comment",	"pkg_name",	"datauuid"]
stress = stress.drop(columns=list_intersection(columns_to_remove, stress.columns))

stress = do_the_rudamentary_time_changes(stress)
stress["hour"] = stress["start_time"].apply(lambda x: datetime.fromtimestamp(x).hour)
stress.head(10)
daily_stress = sum_df_by_variable(stress, "date", ["score", "max", "min", "score", "day_of_the_week"], ["mean"])
daily_stress.rename(columns={"score" : "stress_score", "max" : "stress_max", "min" : "stress_min", "score" : "stress_score"}, inplace=True)
daily_stress.index = pd.to_datetime(daily_stress.index)
daily_df.drop(columns=["score"], inplace=True, errors="ignore")
union_dataframes(daily_df, daily_stress, "date")

# %%
plot_box_and_whisker(daily_stress, "day_of_the_week", "stress_score", "Stress Level vs Day of the Week", "Day of the Week", "Stress Level")
plot_box_and_whisker(daily_stress, "day_of_the_week", "stress_max", "Peak Stress Level vs Day of the Week", "Day of the Week", "Stress Level")

# %%
plot_box_and_whisker(stress, "hour", "score", "Stress Level vs Hour of the Day", "Hour of the Day", "Stress Level")

# %%
def plot_matrix(dataframe: pd.DataFrame, x_axies, y_axis, title):
  # This function will be used to see the average stress given a specific hour and given a specific day
  assert len(x_axies) == 2
  fig, ax = plt.subplots(figsize=(16, 9))

  avg_stress = dataframe.groupby([x_axies[0], x_axies[1]])[y_axis].mean()

  # convert the resulting series to a matrix with days as rows and hours as columns
  matrix = avg_stress.unstack(level=1)
  
  im = ax.imshow(matrix, cmap="hot", interpolation="nearest")
  
  # convert the x axis into days of the week

  new_labels = ["","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  ax.set_yticklabels(new_labels)
  ax.set_xticklabels([""] + [str(z) + ":00" for z in range(6, 24, 2)])
  ax.set_title(title)
  
  colorbar = ax.figure.colorbar(im, ax=None)
  colorbar.set_label("Stress Level (0-100)")
  savefig(title, type_="matrix")

plot_matrix(stress[stress["hour"] > 6], ["day_of_the_week", "hour"], "score",  "Stress Correlation Matrix For Days of the Week vs. Hour of the Day")

# %% [markdown]
# # Computer Usage

# %%
data = json.loads(open(computer_usage_folder_dir + "aw-bucket-export_aw-watcher-window_matts-computer.json").read())

# %%
computer_usage_raw = pd.DataFrame(data["buckets"]["aw-watcher-window_matts-computer"]["events"])
computer_usage_raw["start_time"] = computer_usage_raw["timestamp"].apply(lambda string: string.split("+")[0].split(".")[0]).apply(convert_datetime_to_time_since_last_epoch) - 5 * 3600

computer_usage_raw["timestamp"] = computer_usage_raw["start_time"].apply(convert_time_since_last_epoch_to_date)
computer_usage_raw.sort_values(by="start_time",ascending=False )

computer_usage_raw = do_the_rudamentary_time_changes(computer_usage_raw)
computer_usage = sum_df_by_variable(computer_usage_raw, "date", ["duration", "day_of_the_week", "start_time"], ["mean", "first", "max"])
computer_usage["last_time_used"] = computer_usage["start_time"]
computer_usage["start_time"] = (computer_usage["start_time"]).apply(convert_time_since_last_epoch_to_date).apply(convert_datetime_to_time_since_last_epoch)
computer_usage["last_time_used"] = (abs(computer_usage["start_time"] - computer_usage["last_time_used"])) / 3600


# %% [markdown]
# ## Computer Usage vs. Sleep

# %%
def pretty(item, name):
    if name == "correlation":
        print(f"The correlation between computer usage and sleep is {item}")
pretty(compute_correlation(computer_usage, daily_sleep, "index", ["last_time_used", "sleep_score"]), "correlation")

# %%
computer_usage["datetime"] = computer_usage["start_time"].apply(convert_time_since_last_epoch_to_date)
daily_sleep["datetime"] = daily_sleep["start_time"].apply(convert_time_since_last_epoch_to_date)

# %%
merged = pd.merge(computer_usage, daily_sleep, on="datetime")
print(merged.head(5))
# plot_variables(merged, "datetime", ["last_time_used", "sleep_score"], True, "Last Time Used vs. Sleep Score")

# %% [markdown]
# # Exercise

# %%
exercise = pd.read_csv(galaxy_watch_folder_dir + "exercise.csv", index_col=False)
columns_to_remove = ["subset_data", "routine_datauuid", "activity_type", "title", "tracking_status", "source_type", "reward_status", "mission_extra_value", "program_schedule_id", "program_id", "mean_caloricburn_rate", "heart_rate_deviceuuid", "live_data_internal", "mission_value", "pace_info_id", "pace_live_data", "mission_type", "location_data_internal", "additional_internal", "min_altitude", "max_altitude", "deviceuuid","completion_status" , "comment", "location_data", "sensing_status", "incline_distance", "decline_distance", "live_data", "datauuid", "max_cadence", "altitude_gain", "update_time", "create_time"]
remove_prefix = lambda x: x.replace("com.samsung.health.exercise.", "")

load_json_from_file = lambda file_name: json.loads(open(file_name).read())
save_json_to_file = lambda file_name, data: open(file_name, "w").write(json.dumps(data, indent=2))

exercise_type_number_to_string_map = load_json_from_file("exercise_label_map.json")



exercise = exercise.rename(columns=remove_prefix)

exercise = exercise.drop(columns = columns_to_remove)
exercise = do_the_rudamentary_time_changes(exercise)
exercise["exercise_type"] = exercise["exercise_type"].apply(lambda x: str(x)) # so it works with the json label map
print(exercise.head(10))
exercise["duration"] /= 1000

# use the exercise_type_number_to_string_map
# find any exerices not in the map 
grouped = exercise.groupby("exercise_type").groups
if any([key not in exercise_type_number_to_string_map for key in list(grouped.keys())]):
    logger.error("There are some exercises that I haven't labeled yet!")
    unknown = (grouped.keys() - exercise_type_number_to_string_map.keys())
    for key in unknown:
        logger.error(f"Unknown exercise type: {key}")
        logger.info("You did this exercise on the following dates: " + ", ".join(exercise.loc[grouped[key], "date"]))
        new_label = input("What should the label be?\n")
        exercise_type_number_to_string_map[key] = new_label
    save_json_to_file("exercise_label_map.json", exercise_type_number_to_string_map)


exercise["exercise_type"] = exercise["exercise_type"].apply(lambda x: exercise_type_number_to_string_map[x])


# %% [markdown]
# ## Exercise & Sleep

# %%
# Find days when we go to the gym (weight lifting)

days_worked_out = exercise[exercise["exercise_type"] == "Weight Lifting"]["date"].unique()

print(daily_sleep.head(5))
daily_sleep["datetime"] = daily_sleep["start_time"].apply(convert_time_since_last_epoch_to_date)
daily_sleep["worked_out"] = daily_sleep["datetime"].apply(lambda x: float(x in days_worked_out))

daily_sleep["worked_out"].corr(daily_sleep["sleep_score"])

# %% [markdown]
# # Weight Lifting

# %%
data_file_path = exercise_logs_folder_dir + "gym_data_2022.txt"
raw = ""
with open(data_file_path, "r") as f:
  raw = f.read()

# Split by paragraph
raw = raw.split("\n\n")

# Clean data
raw = [r.lower() for r in raw]
print(raw)

import json

def get_first_non_alpha_character_index(string):
  for i, c in enumerate(string):
    if not c.isalpha() and c != " ":
      return i
  raise Exception("No non-alpha characters found in " + repr(string))

exercises = {}
with open(exercise_logs_folder_dir + "exercise_aliases.json", "r") as f:
  exercises = json.loads(f.read())

raw_exercises = [line.split("\n")[1:] for line in raw]

# flatten all_exercises array

raw_exercises = [item for sublist in raw_exercises for item in sublist]

# remove empty strings
raw_exercises = [x for x in raw_exercises if x and x != " "]

all_exercise_names = []
for exercise in raw_exercises:
  print(exercise)
  all_exercise_names.append(exercise[:get_first_non_alpha_character_index(exercise)])
all_exercise_names = [z.strip() for z in all_exercise_names]

# %% [markdown]
# ## Run this whenever importing new data to see if there are any new exercises

# %%

for exercise in all_exercise_names:
  exists = False
  for exercise_type in exercises:
    if exercise in exercises[exercise_type]:
      exists = True
      break
  if not exists:
    answer = input("Ayo, we couldn't find an alias for " + exercise + ". What should its alias be?\nOptions:" + str(list(exercises.keys())) + "\n")
    exercise = exercise.strip()
    answer = answer.strip()
    if answer in exercises:
      exercises[answer].append(exercise)
    else:
      ohOh = input("We couldn't find " + answer + " in the list of exercise types. Should we add it?\n")
      if "y" in ohOh:
        exercises[answer] = [exercise]
      else:
        print("We didn't add the exercise:\t" + exercise)
#
# response = input("Do you want to save this data? (y/n)\n")
#   save_json_to_file(exercise_logs_folder_dir + "exercise_aliases.json", exercises)

# %%

# Turn date into a time object
def convert_date_to_days(date):
  if type(date) == list:
      if len(date) == 1:
        date = date[0]
      else:
        raise Exception("Homie wrong data type", date)
  try:
    if "-" in date:
      date = date.split('-')
    elif "/" in date:
      date = date.split('/')
    else:
      raise Exception("Homie wrong date", date)
    year = int(date[2])
    month = int(date[1])
    day = int(date[0])
    return (year - 22) * 365 + month * 30.5 + day
  except:
     print(date)
     raise Exception("Error: " + date)

def get_first_non_number_or_space_index(string):
    for i in range(len(string)):
        if not string[i].isdigit() and string[i] != ' ' and string[i] != "/":
            return i
    return -1

def get_first_number(string):
    if len(string) == 0:
       raise Exception("Not a proper string, length is 0")
    
    for i in range(len(string)):
        if string[i].isdigit():
            start_index = i
            break
    else:
       raise ValueError(string + " does not have a number!")
    for i in range(start_index, len(string)):
        if not string[i].isdigit() and string[i] != ".":
            return string[start_index:i]
    return string[start_index:]

class Set:
   def __init__(self):
      self.weight = 0
      self.reps = 0

class Workout:
   def __init__(self, day, sets):
      self.day = day
      self.sets = sets

def get_set_info_from_string(string):
  string = string.strip()
  sets = [Set() for i in range(len(string.split(",")))]
  weight = get_first_number(string)
  remainding_string = string[string.index(weight) + len(weight):]
  for i in range(len(sets)):
    sets[i].weight = float(weight)
    reps = get_first_number(remainding_string)
    sets[i].reps = float(reps)
    remainding_string = remainding_string[remainding_string.index(reps) + len(reps):]
    # print("Weight: ", sets[i].weight, "Reps: ", reps)
  return sets
 
workouts = [] 
previous_day_of_workout = -999999
for i in range(len(raw)):
  day_of_workout = (convert_date_to_days(raw[i][:get_first_non_number_or_space_index(raw[i])]))
  if day_of_workout < previous_day_of_workout:
    print("Error: ", raw[i])
  exercises_on_that_day = raw[i].split("\n")[1:]
  exercises_on_that_day = [exercise for exercise in exercises_on_that_day if exercise != '']
  sets = []
  for exercise in exercises_on_that_day:
    exercise_type = exercise[:get_first_non_alpha_character_index(exercise)]
    exercise_type = exercise_type.strip()

    for possible_exercise_type in exercises:
      if exercise_type in exercises[possible_exercise_type]:
        exercise_type = possible_exercise_type
        break
    
    # if possible_exercise_type == "bench press":
    if exercise_type == "lat pulldown":
      print("For day of", raw[i][:get_first_non_number_or_space_index(raw[i])], "exercise type is", exercise_type)
      exercise_set = exercise[get_first_non_alpha_character_index(exercise):].split(";")
      exercise_set = [set.strip() for set in exercise_set]
      exercise_set = [set for set in exercise_set if set != '']
      for set in exercise_set:
        # print(set)
        if set[0] == "-":
          set = set[1:]
        # sets.extend(get_set_info_from_string(set))
        for set in get_set_info_from_string(set):
          sets.append(set)
      print(len(sets))
      workouts.append(Workout(day_of_workout, sets))
    
    previous_day_of_workout = day_of_workout

  # print(exercises_on_that_day)


# %%
def reps_to_one_rep_max(rep):
    return 1 - rep / 40

days = []
metrics = []

for workout in workouts:
    if any([set.reps <= 12 for set in workout.sets]):
        days.append(workout.day)
        metrics.append(max([set.weight / reps_to_one_rep_max(set.reps) for set in workout.sets if set.reps <= 12]))


fit_of_x = np.polyfit(days, metrics, 2)

### Use the fit of x to extract any outliers
outliers = []
for day, metric in zip(days, metrics):
    if metric > np.polyval(fit_of_x, day) * 1.25:
        outliers.append((day, metric)) 
    if metric < np.polyval(fit_of_x, day) * 0.8:
        outliers.append((day, metric)) 

## remove the outlier from the dataset
for outlier in outliers:
    days.remove(outlier[0])
    metrics.remove(outlier[1])

fit_of_x = np.polyfit(days, metrics, 2)
plt.plot(days, metrics, 'ro')
plt.plot(days, np.polyval(fit_of_x, days), 'b-')
plt.savefig("lat pulldown")

# %% [markdown]
# # All Data

# %% [markdown]
# ## Calendar

# %%
from datetime import datetime
import ics
from ics import Calendar
from math import ceil
# Open the .ics file and read its contents into a string variable
with open(calendar_events_folder_dir + 'handzelmatthew@gmail.com.ics', 'r') as f:
    calendar_str = f.read()

# Create a new Calendar object and parse the contents of the .ics file into it
calendar = Calendar()

# Concatenate all of the calendars in the calendar_events_folder_dir

for file in os.listdir(calendar_events_folder_dir):
  if file.endswith(".ics"):
    with open(calendar_events_folder_dir + "/" + file, 'r') as f:
      calendar_str = f.read()
    c = Calendar(calendar_str)
    calendar.events.update(c.events)

days_spent_at_home = 0


# %%
events = pd.DataFrame()

# load the previous map if it exists
previous_map = {}

# make file if not exist
if not os.path.exists(calendar_events_folder_dir + "event_classification.json"):
    with open(calendar_events_folder_dir + "event_classification.json", "w") as f:
        f.write(json.dumps({}))
with open(calendar_events_folder_dir + "event_classification.json", "r") as f:
    previous_map = json.loads(f.read())


# Filer calendar events
to_remove = []
for event in calendar.events:
  if event.all_day:
    to_remove.append(event)
    if "home" in event.name.lower():
      days_spent_at_home += 1
for event in to_remove:
  calendar.events.remove(event)

def parse_event_name(name):
  '''makes it lower case and also removes any punctiation and multiple spaces in a row'''
  name = name.lower().replace("  ", " ").replace("  ", " ").replace("!", "").replace("?", "").replace(".", "").replace(",", "")
  if " " == name[0]:
      name = name[1:]

  return name
  
events["event"] = [event for event in calendar.events]
events["name"] = events["event"].apply(lambda event: parse_event_name(event.name))
events["duration"] = events["event"].apply(lambda event: (event.end - event.begin).seconds / 3600)
events["start_time"] = events["event"].apply(lambda event: event.begin.timestamp())
events["end_time"] = events["event"].apply(lambda event: event.end.timestamp())

repeating_events = pd.DataFrame()

def is_repeating_event(event):
  return "RRULE" in event.serialize()
# get all of the repeating events out of the events[] before we do the rudamentary time changes
for index, row in events.iterrows():
    if is_repeating_event(row["event"]):
      repeating_events = pd.concat([repeating_events, pd.DataFrame(row).T])
      events.drop(index, inplace=True)

events = do_the_rudamentary_time_changes(events)

# add the rows of repeating events to events
events = pd.concat([events, repeating_events])

def get_time_until(event):
  if is_repeating_event(event):
    return event.end.timestamp() - event.begin.timestamp()
  raise Exception("whoops")

def get_frequency_from_calendar_event(event: ics.Event):
  string = event.serialize().split("\r\n")[1]
  return string.split("=")[1].split(";")[0]
def return_number_of_times_event_was_cancled(event: ics.Event, start_time = data_start_time, end_time = data_end_time):
  splitted = (event.serialize()).split("\r\n")
  exdates = []
  count = 0
  for s in splitted:
    if "EXDATE" in s:
      exdates.append(s.split(":")[1])
  for i in range(len(exdates)-1, 0, -1):
    t = (datetime.strptime(exdates[i], '%Y%m%dT%H%M%S').timestamp())
    if t > start_time and t < end_time:
      count += 1
  return count
def get_until_from_calendar_event(event: ics.Event):
  string = event.serialize().split("\r\n")[1]
  if "until" in string.lower():
    return datetime.strptime(string.split("UNTIL=")[1].split(";")[0], '%Y%m%dT%H%M%SZ').timestamp()
  return 9e90
def get_byday_from_calendar_event(event: ics.Event):
  string = event.serialize().split("\r\n")[1]
  if "BYDAY" in string:
    return string.split("BYDAY=")[1].split(";")[0]
  return "N/A"

def get_count_from_calendar_event(event: ics.Event, start_time = data_start_time, end_time = data_end_time):
  ''' ex: 
  DTSTART;TZID=America/Chicago:20230913T114500
  DTEND;TZID=America/Chicago:20230913T170000
  RRULE:FREQ=DAILY;COUNT=1
  DTSTAMP:20231027T143038Z
  UID:3n2hdiui9dr3fn02bj4p8bep9t@google.com
  CREATED:20230817T022401Z
  LAST-MODIFIED:20230827T002836Z
  SEQUENCE:0
  STATUS:CONFIRMED
  SUMMARY:Career Fair
  TRANSP:OPAQUE
  '''
  string = event.serialize().split("\r\n")[1]

  # If event.end() + count * 24 * 60 * 60 < start_time, then return 0
  if "COUNT" not in string:
    return -1
  count = int(string.split("COUNT=")[1].split(";")[0])
  count_end_time = count * 24 * 60 * 60 + event.end.timestamp()
  if count_end_time < start_time:
    return 0
  return count

events_uid_map = {}

for index, row in events.iterrows():
  uid = row["event"].uid
  if uid not in events_uid_map:
    events_uid_map[uid] = [row]
  else:
    events_uid_map[uid].append(row)

# Check to see if the event is repeating event, if it is a repeating event then compute the difference between the (start date and data_end_time) in weeks and (start date and event date until) in weeks, then assign whicher is lower to a variable called "week_repeated",
for index, row in events.iterrows():
  if is_repeating_event(row["event"]):
    # get the uid from the uid map, get the length of the resulting array and then subtract by 1
    offset = len(events_uid_map[row["event"].uid]) - 1
    max_start_time = max(row["start_time"], data_start_time)
    if((data_end_time - max_start_time) < 0):
      events.drop(index, inplace=True)
      continue
    current_event_frequency = get_frequency_from_calendar_event(row["event"])
    if current_event_frequency == "WEEKLY":
      if(get_until_from_calendar_event(row["event"]) - max_start_time) < 0:
        events.drop(index, inplace=True)
      weeks_repeated = min((data_end_time - max_start_time) / 604800, (get_until_from_calendar_event(row["event"]) - max_start_time) / 604800)
      # weeks_repeated is a variable that will figure out how many weeks the event has repeated for, we will then multiply weeks_repeated by the number of days in get_by_day
      if get_byday_from_calendar_event(row["event"]) != "N/A":
        days_repeated = ceil(len(get_byday_from_calendar_event(row["event"]).split(",")) * weeks_repeated)
      else:
        days_repeated = ceil(weeks_repeated)
        
    elif current_event_frequency == "DAILY":
      days_repeated = get_count_from_calendar_event(row["event"], max_start_time, data_end_time)
      if days_repeated < 0:
        if(get_until_from_calendar_event(row["event"]) - max_start_time) < 0:
          events.drop(index, inplace=True)
        days_repeated = min((data_end_time - max_start_time) / 604800, (get_until_from_calendar_event(row["event"]) - max_start_time) / 604800)
    elif current_event_frequency == "MONTHLY":
        
        months_repeated = min((data_end_time - max_start_time) / 604800, (get_until_from_calendar_event(row["event"]) - max_start_time) / 604800)
        if get_byday_from_calendar_event(row["event"]) != "N/A":
          days_repeated = ceil(len(get_byday_from_calendar_event(row["event"]).split(",")) * months_repeated)


        
    else:
        raise Exception(f"Invalid frequency {current_event_frequency}")
    days_repeated -= return_number_of_times_event_was_cancled(row["event"], start_time = max_start_time, end_time = data_end_time)
    days_repeated -= offset
    events.loc[index, "duration"] = ceil(days_repeated) * row["duration"]

# remove nan rows
events = events[pd.notna(events["name"])]

events.sort_values(by="duration", ascending=False).head(20)

# %% [markdown]
# ### Use chatpgt to classify my events

# %%
import pyperclip

#  Use chatgpt for some parsing of the event data into the following categories
categories = {
  "SCHOOL",
  "PRODUCTIVITY",
  "SOCIALIZING",
  "EATING",
  "UNKNOWN",
  "MISC",
  "CAREER",
  "RESEARCH",
  "CHORES",
  "EXERCISING",
  "HOMEWORK",
  "PERSONAL PROJECTS",
  "GIRLFRIEND",
  "SELF IMPROVEMENT",
  "STUDENT ORGANIZATIONS",
  "FREE TIME"
}

courses = {
  "CS 225: Data Structures",
  "CS 233: Computer Architecture",
  "CS 100: Computer Science Orientation",
  "ENG 100 Engineering Orientation",
  "Dance 100: Introduction to Dance",
  "PSYC417: Neuroscience of Eating and Drinking", 
"CS 361: Probability & Statistics in Computer Science",
"CS 357: Numerical Methods I",
"CS 446: Machine Learning",
"CS 374: Introduction to Algorithms & Models of Computation",
"CS 341: System Programming",
"CS 416: Data Visualization",
"CS 421: Programming Languages & Compilers",
"PSYC 250: Psychology of Personality",
"CS 222: Software Design Lab",
}

info = f'''
- I am a college student studying computer science taking the following courses {courses}
- I am in two research organizations, one of the projects is Jimmy Project (which deals with TCRs) and the other is Universum (which is a neuroscience project)
- If you do not know what something is then label it as unknown and I will manually label it
- Forager project is a personal project
- My girlfriend's name is Bella
- preflights are homework for my cs 233 class
- if i am studying anything then it is for school
- when the letters EX or XC are related to something with school then it means extra credit  
- labs are considered homework
- sometimes I will just say the name of a class if I am doing homework for that class (ex. eng 100) 
- any public talks by people are considered career or self-improvement depending on the topic
- networking is considered career or self-improvement depending on the topic
- anything that has mp is a machine problem which is homework for cs 225
- Do NOT fix any spelling mistakes in the event names
- DO NOT CHANGE THE EVENT NAMES, IF YOU CHANGE THE EVENT NAMES THIS WONT WORK
- SIGAIDA, Neurotech, Pacbots, AI@UIUC, and WCS (women in computer science) are student organizations
- Output your response as a JSON object that is a map from event name to the category it belongs to
- If you are unsure about where something belongs then ask me where it belongs and then re-output this information with the new information that you learned from me
- Typically I won't label what class my homework is for. You can use your understanding of the courses to know what homework assignments are for what classes. For example, MP Malloc is most likely for a Systems programming class. You can also label it as UNKNOWN and I can label it. Look at the previous event labels if you are stuck.
- sights and sounds is a socializing event
- vex stand is a student organization event
- E-day is a socialzing event
- flippin illini practice is a student organization
- learn refers to productivity, work is productivity, controverises and wicked problems is homework, lab is school, mp stickrs is cs225
- if something has a class name or number in it and then some words or work phrase then it is likely homework if it just the name of the class then it is school
- any planning is classified as productivity if it is vague, if it has something to do with a class then it is school, if it related to a project that is research then it is research
- sometimes I say just the class number instead of the class
- potd stands for problem of the day, gps stands for guided problem set, 
- your output should be syntactically correct json, nothing extra
- to prof out means to take a proficiency exam for a class
- sometimes the name of events are just reminders to do something or tasks to complete
'''

example = """
{
  "cold email profs": "CAREER",
  "11 mentoring with sailaja": "SELF IMPROVEMENT",
  "lunch": "EATING",
  "cs 225 potd": "HOMEWORK",
  "workout": "EXERCISING",
  "bella": "GIRLFRIEND",
  "interview for research": "CAREER",
  "233 lab": "HOMEWORK",
  "work on selfimprovement": "SELF IMPROVEMENT",
  "workout at cerce": "EXERCISING",
  "plan the following week": "PRODUCTIVITY",
  "cold email": "CAREER",
  "prepare for neurotech interview": "CAREER",
  "neurotech interview": "CAREER",
  "academic progress": "SCHOOL",
  "pacbots meeting": "STUDENT ORGANIZATIONS",
  "dinner": "EATING",
  "selfimprovement": "SELF IMPROVEMENT",
  "wakeup": "MISC",
  "do small tasks": "PRODUCTIVITY",
  "run": "EXERCISING",
  "showerget ready": "CHORES",
  "computer vision group ai under professor": "CAREER",
  "qamar lunch": "SOCIALIZING",
}
"""

events_to_classify = []
for event in events["name"]:
  if event not in previous_map and event not in events_to_classify:
    events_to_classify.append(event)
  if len(events_to_classify) >= 150:
    break

while events_to_classify != []:

    prompt = f'''
    Your instruction is to categorize the following events you will use the following information to help you classify. First, look through the events and see what patterns you see. Write in bold any questions you may have for me if you are unsure how to label a specific event. Here is an example of how to label the events.
    \'\'\'
    {info}
    \'\'\'
    The categories you must classify everything into are as follows:
    \'\'\'
    {categories}
    \'\'\' 

    Here is an example of 
    \'\'\'
    {example}
    \'\'\'

    The following are the events you must classify:
    \'\'\'
    {events_to_classify}
    \'\'\'
    '''
    print(prompt)

# use pyperclip to copy the prompt into clipboard
    time.sleep(0.2)
    pyperclip.copy(prompt)

    response = input("Input when you are ready!\n")
    response = pyperclip.paste()
    response = json.loads(response)
    print("We recieved the following response: ", response)

# concatenate the two maps
    previous_map.update(response)

# save the map
    with open(calendar_events_folder_dir + "/event_classification.json", "w") as f:
        f.write(json.dumps(previous_map, indent=2))
    events_to_classify = []
    for event in events["name"]:
      if event not in previous_map and event not in events_to_classify:
        events_to_classify.append(event)
      if len(events_to_classify) >= 150:
        break



# %%
events["categories"] = [previous_map[event] if event in previous_map else "UNKNOWN" for event in events["name"]]

# i dont go to cs 225 lecture lol
def drop_events_with_following_names(dataframe, names):
  return dataframe[~dataframe["name"].isin(names)]

events = drop_events_with_following_names(events, ["data structures", "black music", "night routine & falling asleep", "home", "home :)", "wakeup + morning routine", "wakeup"])

# filter out the events that are misc or unknown or eating or chores
# Create a pie chart for how much of each category i spend my time on using the duration

# %%
# add calendar data to daily_df (add a column for every category and then write down the total amount of time spent per day on each category)

for category in categories:
  daily_df[category] = 0

print(daily_df.columns)
print(daily_df.head(5))
for index, row in events.iterrows():
  if row["date"] not in daily_df.index:
    continue
  daily_df.loc[row["date"], row["categories"]] += row["duration"]

# plot the pie chart
daily_df

# %%
summation = sum_df_by_variable(events, "categories", "duration", "sum") 


events.sort_values(by="start_time", ascending=True, inplace=True)
print(events.head(5))
first_calendar_event_time = events["start_time"].iloc[0]
first_calendar_event_time = max(data_start_time, first_calendar_event_time)
last_calendar_event_time = min(events["start_time"].iloc[-1], data_end_time)
day_difference = round((last_calendar_event_time - first_calendar_event_time) / 60 / 60 / 24)

summation /= day_difference
summation *= 7


# get sleep information
# sum_df_by_variable(daily_sleep, "start", "duration", "sum")

# Get average amount of sleep for the date range chosen 
t = daily_sleep[daily_sleep["start_time"] > data_start_time]
t = t[t["start_time"] < data_end_time] 
sleep_average = t["sleep_duration"].mean() * 7
summation.loc["SLEEP"] = sleep_average
summation.loc["HOME"] = days_spent_at_home / day_difference * (168 - sleep_average) 

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = round((pct*total/10.0)) / 10
        return '{v:2}'.format(v=val) if pct > 0 else ''
    return my_format
summation.loc["UNKNOWN"] = 24 * 7 - summation.sum()
values = summation['duration']

summation.sort_values(by="duration", ascending=False, inplace=True)
colors = plt.cm.rainbow(np.linspace(0, 1, len(summation)) ** (1/1.5))
title = "How I Spend my Week Categorized (Hours)"
# Plot the pie chart with a legend
ax = summation.plot.pie(y='duration', figsize=(10, 9), autopct=autopct_format(values), startangle=180, colors=colors, legend=False, title=title, labeldistance=1.05)
# ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_ylabel('')
savefig(title, type_="pie")
# plot a stacked bargraph 
summation.plot.bar(stacked=True)

# %%
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ics')

events.sort_values(by="duration", ascending=False, inplace=True)

events[events["categories"] == "RESEARCH"].head(25)

# %%
columns_to_drop =["UNKNOWN", "start_time", "height", "sleep_day_of_the_week",  "skeletal_muscle_mass", "create_time", "basal_metabolic_rate", "fat_free",  "update_time", "water_mass" "create_time", "MISC", "pkg_name", "time_offset"] 
for column in columns_to_drop:
  try:
    daily_df.drop(column, axis=1, inplace=True)
  except:
    pass
print(daily_df.columns)

daily_df.head(20)
daily_df.to_csv("daily_df.csv")

# %%
import seaborn as sns
hm = daily_df.corr()
plt.figure(figsize=(32, 18))
sns.heatmap(hm, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")


# %% [markdown]
# ## Construct Readme.md

# %%


# %%
do_we_have_a_valid_start_time = data_start_time > time.time() / 100 
do_we_have_a_valid_end_time = data_end_time < time.time() * 100
slope_of_line_of_best_fit_for_weight_loss = np.polyfit(weight["start_time"], weight["weight"], 1)[0] * (3600 * 7 * 24)

the_data_timeline = "" if not do_we_have_a_valid_start_time and not do_we_have_a_valid_end_time else "This report was written with data from " + convert_time_since_last_epoch_to_date(data_start_time) + " to " + convert_time_since_last_epoch_to_date(data_end_time) +  f" (a total of {round((data_end_time - data_start_time) / 60 / 60 / 24)} days)."
insights = f'''\
One critical insight for me during this period is that I was going through a cut, so I wanted to make sure I was losing weight at a healthy rate so that I wouldn't lose too much muscle. \
I was able to do this by plotting my weight vs. time and then finding the slope of the line of best fit. Which was {round(slope_of_line_of_best_fit_for_weight_loss,2)} lbs/day (which is a healthy rate of weight loss).
'''
text = f'''\
# Description
In this project, I collected and analyzed data from a variety of sources (Samsung Galaxy Watch 5, Google Calendar, smart scale, computer activity, etc.), put it all in one place, and then did visualizations about different metrics to figure out trends, correlations, and improve my life. Examples of some things I've explored are how I spend my time, how my weight has been changing since college, how my one rep max has increased at the gym. 

## Equipment
This is a project that will analyze data from a variety of sources, from my smartwatch (for health-related metrics like sleep, steps walked, hr, etc.) from my phone (used for my location log and to log my weight-training sessions), from my google calendar (to get an idea of how I spend my time), and computer (to understand when and how frequently I use my computer)

## Data/Insights
{the_data_timeline}
{insights}


# Graphs

###### Weight (& average weight loss per day) vs. time
![alt text](./figs/plot_for_weight_vs._time.png)

###### How I spend my time
![alt text](./figs/pie_for_how_i_spend_my_week_categorized_(hours).png)

###### Sleep duration Vs. Time
![scatter plot for sleep duration vs. time](figs/line_of_best_fit_for_sleep_duration_(hours)_vs._time.png)

###### Sleep duration Vs. Day of the Week
![box plot for sleep duration vs. day of the week](figs/box_and_whisker_for_sleep_duration_vs_day_of_the_week.png)

###### Weight vs. Time
![scatter plot for weight vs. time](figs/plot_for_weight_vs._time.png)

###### Stress Vs. Day of the Week
![box plot for stress vs. day of the week](figs/box_and_whisker_for_stress_level_vs_day_of_the_week.png)

###### Stress Vs. Hour
![Alt text](figs/box_and_whisker_for_stress_level_vs_hour_of_the_day.png)

###### Stress. Vs. Day of the Week Correlation Matrix
![alt text](figs/matrix_for_stress_correlation_matrix_for_days_of_the_week_vs._hour_of_the_day.png)

##### Computer Usage vs. Day of the Week
![alt text](figs/box_and_whisker_for_computer_usage_vs_day_of_the_week.png)
'''
print(text)
with open("README.md", "w") as f:
  f.write(text)

# look at the last commit, if it was very recent, don't commit again

last_commit = os.popen("git log -1 --pretty=format:\"%ci\"").read()
parse_time = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S %z")

# should_we_commit = last_commit.split(" ")[0] != str(datetime.now().date())

last_commit_time = parse_time(last_commit)
daily_df.to_csv("./data/daily_df.csv")
print(last_commit_time)
print(current_date)
if current_date - last_commit_time < timedelta(days=1):
    print("We don't need to commit again")
    raise Exception("We don't need to commit again")


command = ("git add README.md && git commit -a -m \"Updated README.md\" && git push")
os.system(command)

# %% [markdown]
# 


