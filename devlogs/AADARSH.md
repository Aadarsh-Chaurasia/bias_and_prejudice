## Before hackathon
I have created basic directory.
/data - training sample data (1000 samples)
/notebooks - our own isolated notebooks for EDA and other testing.
/src - isolated helper files like load.py, train.py or whatever needed.
/submissions - Output csv (for both testing and leaderboard)

Things I noticed about data :
s1 se s2 aur s3 ye entities match krne hai 
ground truth wali file contains 2 columns 1st : s1 ki entity anc col 2 contains corresponding s2 s3 ki entites basically sql jesa hi data hai aur ye file apna main training ka source rhegi

Getting the entities that don't match is as important as finding that do match so I'm thinking of focusing on entities that don't match.


Just created 80/20 split on given data for training and validation. and sample of 2000 rows.

