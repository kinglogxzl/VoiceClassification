import shutil
if __name__ == "__main__":
    orign_path = '/data/voice/origin/'
    with open('./val_data.txt', encoding='utf-8') as f:
        x = f.readlines()
    x2 = [d.split()[0] for d in x]
    # print(x2)
    index = 1
    for path in x2:
        shutil.copyfile(orign_path + path, './test/%d.wav' % index)
        index = index + 1
