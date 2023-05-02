from typing import List
from submission import Solution

# read input file
n = "01"
data = {'train': [], 'test': []}
with open(f'PA-Week11/sample_test_cases/tree_structure/input{n}.txt') as f:
    for line in f:
        items = line.strip().split()
        label = int(items[0])
        features = {int(item.split(':')[0]): float(item.split(':')[1]) for item in items[1:]}
        if label == -1:
            data['test'].append(features)
        else:
            data['train'].append((features, label))

# instantiate solution class
clf = Solution()

# train the model
train_data = [x[0] for x in data['train']]
train_label = [x[1] for x in data['train']]
test_data = [x[0] for x in data['test']]

predictions = clf.fit(train_data, train_label)



# print(f"We predict {predictions}")


with open(f'PA-Week11/sample_test_cases/tree_structure/output{n}.txt') as f:
    preorder = f.readline().split('}')
    inorder = f.readline().split('}')

print(f"Preorder Expected")
for i in preorder:
    print(i)
print(f"Preorder Observed")
clf.root.pre_order(clf.root)

print(f"\nInorder Expected")
for i in inorder:
    print(i)
print(f"Inorder Observed")
clf.root.in_order(clf.root)

