import os
import re
import pandas as pd


folder_path = "/scratch0/zx22/zijie/esm/eval_results_70/esm2/8m_lora"
all_items = os.listdir(folder_path)
# print(all_items)
files_only = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
name_lst = []

for file in files_only:
  file_name = file.split("_")[1]
  name_lst.append(file_name)

name_lst = [int(num) for num in list(set(name_lst))]
name_lst.sort()

patterns = {
    "precision": r"eval/precision (\d+\.\d+)",
    "recall": r"eval/recall (\d+\.\d+)",
    "em": r"eval/em (\d+\.\d+)",
    "f1": r"eval/f1 (\d+\.\d+)"
}
total_dict = {}
halogenase_lst = []
multi_lst = []
new_lst = []
price_lst = []
column_name = ['precision', 'recall', 'em', 'f1']

# print(files_only)
for name in name_lst:
   for file in files_only:
      file_name = file.split("_")[1]
      # print(file_name)
      if str(file_name) == str(name):
         type_name = file.split("_")[2]
         file_data = open(os.path.join(folder_path, file), "r").read()
         extracted_values = {metric: float(re.search(pattern, file_data).group(1)) for metric, pattern in patterns.items()}
         if type_name == "halogenase":
            halogenase_lst.append(list(extracted_values.values()))
         elif type_name == "multi":
            multi_lst.append(list(extracted_values.values()))
         elif type_name == "new":
            new_lst.append(list(extracted_values.values()))
         elif type_name == "price":
            price_lst.append(list(extracted_values.values()))
   #    break
   # break

# print(halogenase_lst)
# print(multi_lst)
# print(new_lst)
# print(price_lst)

halogenase_df = pd.DataFrame(halogenase_lst, columns=column_name)
print("halogenase")
print(halogenase_df)
multi_df = pd.DataFrame(multi_lst, columns=column_name)
print("multi")
print(multi_df)
new_df = pd.DataFrame(new_lst, columns=column_name)
print("new")
print(new_df)
price_df = pd.DataFrame(price_lst, columns=column_name)
print("price")
print(price_df)
