import os
import shutil


file_lst = []
FILE_PATH="/scratch0/zx22/zijie/esm/results_70/esm1b/650mlora_2024-02-27_23-12-28"
for filename in os.listdir(FILE_PATH):
    ids = int(filename.split("-")[-1])
    file_lst.append(ids)
file_lst.sort()

extracted_numbers = [num for i, num in enumerate(file_lst, start=1) if i % 5 == 0]

# print(len(file_lst))
print(file_lst)
# print(extracted_numbers)
# count = 0 
# for file_name in file_lst:
#     if file_name not in extracted_numbers:
#         folder_name = f"checkpoint-{file_name}"
#         folder_path = os.path.join(FILE_PATH, folder_name)
#         shutil.rmtree(folder_path)

