import json
import os

def load_jsonl(file_path):
    """
    加载 jsonl 文件，返回包含所有数据的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_jsonl_generator(file_path):
    """
    生成器版本：用于逐行加载大型 jsonl 文件以节省内存。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

if __name__ == "__main__":
    # file_path = "/mnt/hdfs/lyl1015/data/piaa/new/sft_rater_data.jsonl"
    # new_file_path = "/mnt/hdfs/lyl1015/data/piaa/new/sft_rater_data_processed.jsonl"

    file_path = "/mnt/hdfs/lyl1015/data/piaa/new/sft_profile_data.jsonl"
    new_file_path = "/mnt/hdfs/lyl1015/data/piaa/new/sft_profile_data_processed.jsonl"
    datas = load_jsonl(new_file_path)
    for d in datas:
        print(d)
        break

    # with open(new_file_path, 'w', encoding='utf-8') as f:
    #     for data in datas:
    #         data['images'] = [os.path.join("/mnt/hdfs/lyl1015/data/piaa/40K", os.path.basename(img)) for img in data['images']]
    #         json.dump(data, f)
    #         f.write('\n')
