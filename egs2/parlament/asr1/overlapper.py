with open('./downloads/parla/clean_test.tsv', "r") as f:
    test_data = f.readlines()

with open('./downloads/parla/clean_train.tsv', "r") as f:
    train_data = f.readlines()

train_data_lis = [i.split("\t")[2] for i in train_data]
count = 0

for i in test_data:
    if i.split("\t")[2] in train_data_lis:
        count += 1

over_lap_perc = count/len(train_data_lis)

print("count:", count)
print("overlap percentage", over_lap_perc)