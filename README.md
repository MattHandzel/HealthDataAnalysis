# Description
In this project, I collected and analyzed data from a variety of sources (Samsung Galaxy Watch 5, Google Calendar, smart scale, computer activity, etc.), put it all in one place, and then did visualizations about different metrics to figure out trends, correlations, and improve my life. Examples of some things I've explored are how I spend my time, how my weight has been changing since college, how my one rep max has increased at the gym. 

![image](https://github.com/user-attachments/assets/467effc4-9929-4305-ab7b-92c3a0d860d2)
I also created an [interactive visualization](https://matthandzel.github.io/CS416-narrative-visualization-final-project/) for my Data Visualization class

## Equipment
This is a project that will analyze data from a variety of sources, from my smartwatch (for health-related metrics like sleep, steps walked, hr, etc.) from my phone (used for my location log and to log my weight-training sessions), from my google calendar (to get an idea of how I spend my time), and computer (to understand when and how frequently I use my computer)

## Data/Insights
This report was written with data from 2023/08/05 to 2024/05/17 (a total of 286 days).
One critical insight for me during this period is that I was going through a cut, so I wanted to make sure I was losing weight at a healthy rate so that I wouldn't lose too much muscle. I was able to do this by plotting my weight vs. time and then finding the slope of the line of best fit. Which was -0.1 lbs/day (which is a healthy rate of weight loss).



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
