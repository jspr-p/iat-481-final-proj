#splitting images into an 80/20 ratio
#total images 9000  = 9000 -> 80%: 7200 ; 20%: 1800 


classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
import os
import glob
import shutil
import random

src_folder = '/Users/jaspe/Documents/GitHub/iat-481-final-proj/incomingData'
train = src_folder +'/train'
test = src_folder +'/test'
val = src_folder +'/val'
src = train


# MOVE 600/3000 IMAGES TO TEST FOLDERS
def splitimg_80_20():
    for letter in classes:
        src_folder = os.path.join(src, letter)
        src_imgs = glob.glob(src_folder+'/*.jpg')
        count=1

        #RANDOMLY TAKE 600/3000 
        random.shuffle(src_imgs)
        while count <= 3:
            src_path = src_imgs[count]
            dst_path = test + '/' + letter +'/' + os.path.basename(src_path)
            #dst_path = test + '/' + letter +'/' + letter + str(count) + '.jpg'
            # print(dst_path);
            shutil.move(src_path, dst_path)
            count = count+1

        #TAKE EVERY 5th 
        # for src_path in train_imgs:
        #     img_num = os.path.splitext(os.path.basename(src_path))[0][1:]
        #     if (int(img_num) % 5)==0:
        #         dst_path = dst + '/' + letter +'/' + letter + str(count) + '.jpg'
        #         # print(dst_path)
        #         # shutil.move(src_path, dst_path)
        #         count = count+1

# FOR FIXING LABELS AND MOVING REMAINING IMAGES TO TRAIN FOLDERS
def renumber_imgs():
    for letter in classes:
        src_folder = os.path.join(src, letter)
        src_imgs = glob.glob(src_folder+'/*.jpg')
        iterator = 1
        for img in src_imgs:
            # print(train+'/'+letter+'/'+letter+str(iterator)+'.jpg')
            os.rename(img, train+'/'+letter+'/'+letter+str(iterator)+'.jpg')
            iterator = iterator+1

def split_test_val():
    for letter in classes:
        src_folder = test + '/' + letter
        src_imgs = glob.glob(src_folder+'/*.jpg')

        random.shuffle(src_imgs)
        count=1
        while count <= 1:
            src_path = src_imgs[count]
            dst_path = val + '/' + letter +'/' + os.path.basename(src_path)
            shutil.move(src_path, dst_path)
            count = count + 1

if __name__ == "__main__": #Ensure the code is only run when intended from this file.
    splitimg_80_20()
    split_test_val()
    print('Splitting complete.')