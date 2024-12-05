import json
with open("data/dress/remaining_dress_asin.json", 'r') as f:
    data = json.load(f)
# with open("data/dress/variant_dress_asin.json", 'r') as f:
#     target_data = json.load(f)
    
# for i in data:
#     if i in target_data:
#         print(i)
print(len(data))
print(len(list(set(data))))
for i in range(0,len(data)):
    for j in range(i+1,len(data)):
        if data[i] == data[j]:
            print(data[i])