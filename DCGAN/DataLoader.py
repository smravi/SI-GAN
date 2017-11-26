import glob
import os
import sys
dataset_path = "." # Directory where all data is placed
correct_image_sound_pair = "./Correct_Image_Sound_Pair.txt"
dir_category_map = {'0':'AirCondition','1':'CarHorn','2':'ChildPlaying',
                   '3':'DogBark','4':'Drilling','5':'EngineDrilling',
                   '6':'GunShot','7':'JackHammer',
                   '8':'Siren','9':'StreetMusic'}

num_classes = 5
data_path_label_dict = {}
for label,category_dir in enumerate(os.listdir(dataset_path)):
    cat_audios_path_list = glob.glob(dataset_path + category_dir + "/Audio/*.wav")
    cat_images_path_list = glob.glob(dataset_path + category_dir + "/Image/*.jpeg") + glob.glob(dataset_path + category_dir + "/Image/*.jpg")
    cat_labels = [label]*len(cat_audios_path_list)
    data_path_label_dict[label] = [cat_audios_path_list,cat_images_path_list,cat_labels]
print(data_path_label_dict.keys())
try:
    for category in range(len(data_path_label_dict)):
        sound_files,image_files,labels = data_path_label_dict[category]
        correct_image_indices = [random.randint(0,len(image_files)-1) for _ in range(len(sound_files))]
        category_list = list(range(num_classes+1))
        category_list.remove(category)
        with open(correct_image_sound_pair,"a") as f:
            for id in range(len(sound_files)):
                wrong_category=random.choice(category_list)
                #print(wrong_category)
                image_id = random.randint(0,len(data_path_label_dict[wrong_category][1])-1)
                #sound_id = random.randint(0,len(data_path_label_dict[wrong_category][0])-1)
                wrong_image = data_path_label_dict[wrong_category][1][image_id]
                #wrong_sound = data_path_label_dict[wrong_category][0][sound_id]
                f.write(','.join([sound_files[id],image_files[correct_image_indices[id]],wrong_image,str(labels[id])])) 
                f.write('\n')
except IndexError:
    print("Data Files might be missing")