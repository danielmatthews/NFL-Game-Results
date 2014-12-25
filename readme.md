# Course Project -- Predicting NFL game outcomes
####By Dan Matthews

### Null Hypothesis
There is no way to predict the outcome of NFL games using game-day data.

### Alternative Hypothesis
Using data from http://pro-football-reference.com, it is possible to predict the outcome of an NFL game.

## Background

### Reason for choosing this project
I am interested in sports data, predicting game outcomes, and discovering the statistics that matter.  For example, over the 2014 season, the Washington Nationals had ~80% chance of winning if they scored 4 runs or more in any game.  This was a result of their consistently well-performing pitching staff, but I believe knowing stats like that can significantly reduce the number of aspects of any game and result in a much more successful team overall.  The Nationals ended the 2014 season with the best record in the National League.  

The NFL and football overall is much more difficult to predict as there are significantly fewer games compared to the MLB (16 vs. 162).  Any sort of statistic driven through any single NFL season would be meaningless due to it's lack of significant data.  Thus, I have compiled data from every game from 2008 to 2013 to draw my conclusions.  I am curious if certain statistics are more likely to lead to wins than others.
### Data Dictionary
Beginning data aka `nfl_stats.csv`:
###### *asterisk denotes data was not used in model
| Name | definition |
| -------- | -------- | 
| Win | whether or not the team won (response variable) |
| Day* | day of the week the game took place (Monday - Sunday)| 
| Date* | date the game took places | 
| [link to box score]* | link to the extended box score |
| Overtime [y/n]* | whether or not the game went into overtime | 
| Points Scored* | number of points scored scored |
| Points Allowed* | number of points allowed |
| First downs gained | number of first downs the team gained throughout the game |
| Total yards | total yards gained by the offense |
| Total passing yards | total yards gained by the offense through passing |
| Total rushing yards | total yards gained by the offense through running the ball |
| Turnovers lost | total turnovers lost by the offense (interceptions or fumbles) |
| First downs given up | amount of first downs given up by the defense |
| Total yards given up | total yards given up to the opposing team's offense by the defense |
| Total passing yards given up | total yards given up to the opposing team's passing offense |
| Total rushing yards given up | total yards given up to the opposing team's running offense |
| Turnovers gained by defense | total turnovers recovered by the defense (through interceptions or fumbles) |
| Offensive rank* | +/- rank for the offense |
| Defensive rank* | +/- rank for the defense |
| Special teams rank* | +/- rank for the special teams |

### Data Structure (final variable name)
* Win (win)
* ~~Day~~
* ~~Date~~
* ~~[link to box score]~~
* ~~Overtime (Y/N)~~
* ~~Opponent~~
* ~~Points Scored~~
* ~~Points Allowed~~
* First downs gained (dn)
* Total yards (TY)
* Total passing yards (PassY)
* Total rushing yards (RushY)
* Turnovers lost (TO)
* First downs given up (dn_ald)
* Total yards given up (TY_ald)
* Total passing yards given up (PassY_ald)
* Total rushing yards given up (RushY_ald)
* Turnovers gained by defense (TO_df)
* ~~Offensive rank~~
* ~~Defensive rank~~
* ~~Special teams rank~~

### Final Data Structure
aka `nflResults.csv`:

| win | dn | TY | PassY | RushY | TO | dn_ald | TY_ald | PassY_ald | RushY_ald | TO_df
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 1 | 26 | 431 | 273 | 158 | 3 | 15 | 286 | 150 | 136 | 2 |
| 1 | 9	| 232| 178 | 54	| 0 | 15 | 318| 189| 129 | 4 |
| 1 | 21 | 358 | 202 | 156 | 1 | 18 | 323 | 226 | 97 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 0 | 18 | 342 | 211 | 131 | 2 | 16 | 333 | 208 | 125 | 2 |
| 0	| 21 | 343 | 214 | 129 | 1 | 15 | 273 | 190 |83 | 4 |
| 0 | 22 | 408 | 206 | 202 | 0 | 17 | 417 | 154 | 263 | 3 |

## Data Exploration and Analysis
I began by cleaning the data.  The data obtained from http://pro-football-reference.com was a little bit messy.  I removed a number of columns: Day, Date, [link to boxscore], Overtime [Y/N], Opponent, Points Scored, Points Allowed, Offensive Rank, Defensive Rank,  and Special Teams rank (columns 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 21, 22, 23 respectively).   Most of the data was insignificant to the analysis (day, date, etc), but some needed to be removed to make the analysis more than just "who won?."  The model would not be very helpful in determining a meaningful statistic if the points scored an points allowed were included as those can determine the result of any game on their own.  I was more interested in the slightly advanced statistics that may be overlooked by a number of teams.  

After removing the columns I deemed unnecessary, I cleaned the remaining data.  I turned wins and losses into 1s and 0s and converted all the NaNs to 0s.  Once clean, the data was exported to `nflResults.csv`.

I then read in the new `nflResults.csv` into python.  I split my data into train and test sets and set the `feature_cols` = `dn, TY, PassY, RushY, TO, dn_ald, TY_ald, PassY_ald, Rush_ald, TO_df`.  I set my `x=nfl_dat[feature_cols]`.  I ran a standard `AdaBoostClassifier` using `DecisionTreeClassifier` and the SAMME algorithm which is preferable for two response adaboosts.  I then calculated the CrossVal score for `n_estimators=200`, `n_estimators=100`, and `n_estimators=50` and determined `n_estimators=200` was the best for my model with an average result of .80.  

### Data Issues
There were two rows in the data that were completely NaNs.  Besides that, no major data issues.
### Results

| dn | TY | PassY | RushY | TO | dn_ald | TY_ald | PassY_ald | RushY_ald | TO_df
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 0.0349 | 0.1397 | 0.0067 | 0.0993 | 0.2288 | 0.0532 | 0.1527 | 0.0 | 0.09927 | 0.2062 | 

### Confusion Matrix
| ~~/~~  | Predicted Yes | Predicted No |  |
| ----- | ----- | ----- | ----- | ----- |
| Actual Yes | 349 | 92 | 441 | 
| Actual No | 79 | 397 | 476 | 
| Total: | 428 | 489 |  | 

### Conclusions and further study 
The most meaningful statistic in determining the result of an NFL game in my dataset is the number of turnovers given up by an offense followed closely by the number of turnovers gained by the defense.  When I started this project, I was convinced that the most important statistic in determining the outcome was the number of first downs gained by an offense. While I was not completely incorrect, turnovers outweighed first downs on every test run of this model.
If I could do this project again, I would spent more time gathering the data.  At the moment, each NFL game is counted twice, once from the winning team and once from the losing team.  I would attempt to find a separate data source or scan for duplicates over the offense/defense statistics to avoid overvaluing each game, which this model could potentially be doing.