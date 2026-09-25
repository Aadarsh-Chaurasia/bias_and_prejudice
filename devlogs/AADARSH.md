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

clean -> split by country -> vectorize -> find top 15 matches using KNN (candidate_pairs) -> LightGBM to refine it further.


## 1. Cleaning Data
1. Normalizing -> lowercase, punctuation.
2. Standardization -> pvt ltd, private limited -> ltd;
3. Postal_code_extraction -> 
4. Phonetic Hashing -> lakshmi and laxmi might be same.
5. Concatenate name + address

## 2. Candidate Pair
using multi-layer soft blocking, union of 3 index
1. for typos -> char N-gram
2. for similar sound by diff spelling -> BM25s
3. Geographic -> 4 layer spatial blocking -> PIN -> State -> region/city -> alphabet
