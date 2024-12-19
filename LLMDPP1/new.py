def evaluateGA(groundtruth, result):
    # load logs and templates
    compared_list = result['log'].tolist()


    '''
    # select groundtruth logs that have been parsed
    parsed_idx = []
    for idx, row in groundtruth.iterrows():

        if row['Content'] in compared_list:
            parsed_idx.append(idx)
            compared_list.remove(row['Content'])
        else:
            print(row['Content'])

    if not (len(parsed_idx) == 2000):
        print(len(parsed_idx))
        print("Wrong number of groundtruth logs!")
        return 0

    groundtruth = groundtruth.loc[parsed_idx]
    '''
    # grouping
    groundtruth_dict = {}
    for idx, row in groundtruth.iterrows():
        if row['EventTemplate'] not in groundtruth_dict:
            # create a new key
            groundtruth_dict[row['EventTemplate']] = [row['Content']]
        else:
            # add the log in an existing group
            groundtruth_dict[row['EventTemplate']].append(row['Content'])

    result_dict = {}
    for idx, row in result.iterrows():
        if row['template'] not in result_dict:
            # create a new key
            result_dict[row['template']] = [row['log']]
        else:
            # add the log in an existing group
            result_dict[row['template']].append(row['log'])

    # sorting for comparison
    for key in groundtruth_dict.keys():
        groundtruth_dict[key].sort()

    for key in result_dict.keys():
        result_dict[key].sort()

    # calculate grouping accuracy
    count = 0
    for parsed_group_list in result_dict.values():
        for gt_group_list in groundtruth_dict.values():
            if parsed_group_list == gt_group_list:
                count += len(parsed_group_list)
                break

    return count / 2000



import os
import pandas as pd
if not os.path.exists("DivLog_bechmark_result.csv"):
    df = pd.DataFrame(columns=['Dataset', 'Parsing Accuracy', 'Precision Template Accuracy', 'Recall Template Accuracy', 'Grouping Accuracy'])
else:
    df = pd.read_csv("DivLog_bechmark_result.csv")
df_groundtruth = pd.read_csv(r'C:\Users\wangk\Desktop\LLMParser-main\LLMParser-main\logs\Mac\Mac_2k.log_structured_corrected.csv')
df_parsedlog = pd.read_csv(r'C:\Users\wangk\Desktop\prediction.csv')
GA = evaluateGA(df_groundtruth, df_parsedlog)
print(GA)
