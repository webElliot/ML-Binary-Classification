# Binary Classification

## Description

This project is looking at a snapshot of a player's stats to try and determine if the user is a bot or not.


The data used in this project has 2 flaws

1) The Imbalance of data leads the AI to have a Bias towards stating a player is a high-risk Induvidual
2) Too much variety
   1) This data does not isolate a specific category of users
   2) Due to the nature of the data the Model cannot find any generalization or figure out which features cause a player to be a bot or not.



## What to take from this project

- The next ML Task for Runescape needs to:

1) Isolate variables by Selecting One Activity.
   1) Then there will be less unnecessary features for the AI and it can look at the features that directly correlate with gold farmers and bots.
2) Focus on a smaller dataset.
3) More Information Rich features:
   1) Xp / Day , Skills / Day, Relative ratio in skills.