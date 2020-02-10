from sklearn.model_selection import KFold


email_data = [1,2,4,5,6,6,7,8,8,9,9,3,343,5,6,546,4,24,32,55,6,5,3432,5,42,4,32235]

kf = KFold(n_splits = 4)

for train_index , test_index in kf.split(email_data):
    print(train_index , test_index)
    
