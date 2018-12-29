import pandas as pd

def label_extract(in_path, out_path):
    label_nudle = pd.read_excel(in_path)
    # 首先根据['seriesuid','class']两列去重
    label_nudle_nodup = label_nudle.drop_duplicates(subset=['seriesuid','class'], keep='first', inplace=False)
    del label_nudle_nodup['coordX']
    del label_nudle_nodup['coordY']
    del label_nudle_nodup['coordZ']
    # 然后遍历'seriesuid'，把同时在'class'列出现'0'和'1'的病例标为'1'，只出现'0'或'1'的分别标为'0'或'1'
    case_nodup = []
    case_cls = []
    print(label_nudle_nodup)
    for i,case in enumerate(label_nudle_nodup['seriesuid']):
        if case not in case_nodup:
            case_nodup.append(case)
            case_cls.append(label_nudle_nodup['class'].iloc[i])
        else:
            case_cls[len(case_cls)-1] = 1
    # 两个list合并为dataframe
    label_dict = {"seriesuid": case_nodup,
                  "class": case_cls}
    labels = pd.DataFrame(label_dict)
    # 写入csv文件
    columns = ['seriesuid','class']
    labels.to_csv(out_path, index=False, columns=columns)
    return labels

if __name__ == "__main__":
    label_extract("C:\\Users\\Royce\\Desktop\\labels.xlsx", "C:\\Users\\Royce\\Desktop\\labels_nodop.csv")


