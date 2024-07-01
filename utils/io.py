import os
import pickle


## 获取目录下的所有文件路径
def all_file(dirname):
    fl = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            fl.append(path)
    return fl


# 读取文件内容
def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return [line[:-1] for line in f]    #读取指定文件的所有行，并去除每行末尾的换行符

# 写入内容到文件
def write_file(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])    # 提取文件所在的目录路径
    try:
        os.makedirs(dirname, exist_ok=True)      # 递归创建目录，如果目录已存在则不报错
    except:
        pass         # 忽略任何可能的异常
    with open(filename, 'w', encoding='utf-8') as f:
        for line in obj:
            f.write(str(line) + '\n')          # 将对象转换为字符串，并写入文件，末尾添加换行符



# 将对象序列化并写入文件
def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:     # 以二进制写入模式打开文件
        pickle.dump(obj, f)


# 从文件中读取并反序列化对象
def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# 将数组分割成指定数量的子数组
def seg_array(inp, num=1):          #将输入列表分割成 num 个子列表
    length = len(inp) // num + 1
    return [inp[ids * length:(ids + 1) * length] for ids in range(num)]


def read_dialog(filename):          #读取对话文件，并将连续的非空行分组为子列表，空行分隔对话
    raw = read_file(filename)
    data = [[]]
    for line in raw:
        if line == '':
            data.append([])
        else:
            data[-1].append(line)
    data = [item for item in data if item != []]        # 去掉空的对话
    return data
