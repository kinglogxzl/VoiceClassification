import os
for dirpath, dirnames, filenames in os.walk(r"/data/voice/processed_merge"):
    file_count = 0
    for file in filenames:
        file_count = file_count + 1
    with open("file_num.txt", 'a') as f:
        f.write("{}\t{}\n".format(dirpath, file_count))
