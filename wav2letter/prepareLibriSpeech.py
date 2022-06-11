import os
import pandas as pd
import soundfile


def create_manifest(data_dir, url):
    #Reads through the directory structure of the 'extracted' LibriSpeech dataset
    #and creates manifest as Dataframe.
    #url: data to be read e.g. 'train-clean-100' or 'dev-clean'
    df = pd.DataFrame(columns=['audio','trans','len'])
    path = os.path.join(data_dir, 'LibriSpeech', url)
    speakers = os.listdir(path)
    for speaker in speakers:
        chapters = os.listdir(os.path.join(path,speaker))
        for chapter in chapters:
            filename = os.path.join(path, speaker, chapter, f"{speaker}-{chapter}.trans.txt")
            with open(filename) as file:
                lines = file.readlines()
            for line in lines:
                data = line.strip("\n").split(" ", 1)
                audio = os.path.join(path, speaker, chapter, f"{data[0]}.flac")
                df.loc[len(df.index)] = [audio,data[1], len(data[1])]
    return df
def save_manifest(data_dir, urls, filename):
    # Calls 'create_manifest(.)' and saves the returned manifest as a CSV file. 
    dataframes = []
    for url in urls:
        dataframes.append(create_manifest(data_dir, url))
    manifest = pd.concat(dataframes)
    # df = create_manifest(urls[0])
    manifest.sort_values(by=['len'], inplace=True)
    manifest.set_index(['audio','trans','len'], inplace=True)
    manifest.to_csv(os.path.join(data_dir,filename))



def prepare_dataset(config, train=True, val=False, test=False):
    # wrapper function to prepare manifests for train, val and test data.
    # Takes in the 'config' ('model_param' dict as e.g. configured in conf/lighnting.yaml)
    # Expects the desired urls for train, val and test to be already downloaded and extracted.
    data_dir = config["data_dir"]
    if train:
        print("Preparing training manifest...!")
        training_urls = config["training_url"]
        train_manifest = config["train_manifest"]
        save_manifest(data_dir,training_urls,train_manifest)
    if val:
        print("Preparing val manifest...!")
        val_urls = config["val_url"]
        val_manifest= config["val_manifest"]
        save_manifest(data_dir,val_urls,val_manifest)
    if test:        
        print("Preparing test manifest...!")
        test_urls = config["test_url"]
        test_manifest= config["test_manifest"]
        save_manifest(data_dir,test_urls,test_manifest)
    print("Done...!")
