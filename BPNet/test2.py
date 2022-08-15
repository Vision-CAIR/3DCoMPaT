from dataset.utils import unpickle_data,pickle_data
data_path = "/home/liy0r/3d-few-shot/3dcompat/3dcompat/baselines/material_classifier/"
splits = next(unpickle_data(data_path + 'dataset_v1.pickle'))
#
train_model_ids = splits['train']
test_model_ids = splits['test']
val_model_ids = splits['val']
l=[]
with open('dataset/indexs.txt','r') as f:

      for i in f:
          l.append(i[:-1])

print(len(train_model_ids))
print(len(test_model_ids))
print(len(val_model_ids))
train_ids=[]
test_ids=[]
val_ids=[]

errors=[]
for i in l:
    if not (i in train_model_ids or i in test_model_ids or i in val_model_ids):
        errors.append(i)
print(len(errors))
# for i in errors:
#     l.remove(i)

for i in train_model_ids:
    if i in l:
        train_ids.append(i)

for i in test_model_ids:
    if i in l:
        test_ids.append(i)

for i in val_model_ids:
    if i in l:
        val_ids.append(i)


print(len(train_ids),len(test_ids),len(val_ids))
new_splits={}
new_splits['train']=train_ids

new_splits['test']=test_ids

new_splits['val']=val_ids
pickle_data('dataset_nov.pkl',new_splits)