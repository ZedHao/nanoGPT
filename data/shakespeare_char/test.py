import pickle

data = {
    'name': 'Alice',
    'age': 30,
    'scores': [90, 85, 95]
}

# 将对象保存到文件
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)