import os

def processCheckFiles(file=None, latency=None, data_dir=None):
    # создание папки, где будут храниться изображения кросс-корр
    if not os.path.isdir(f"{data_dir}/crosscorr"):
        os.mkdir(f"{data_dir}/crosscorr")
    metka=0
    for lat in latency:
        if os.path.isfile(f'{data_dir}/crosscorr/{file[:-5]}_crosscorr_{lat}.npy'):
            metka += 0
        else:
            metka += 1
    if metka>0:
        final_metka = 'yes'
    else:
        final_metka = 'no'
    
    return final_metka