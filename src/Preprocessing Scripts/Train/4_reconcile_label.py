import os
import pandas as pd

'''
   Process Pipline:
       -Get the CSV
       
       -Get the file name in a folder
        --Return list
        
       -Frame the list of file name in sequence
        --Can thought of Dict-like container for series object
        --pd.DataFrame({'Column': Value})
        
       -Assign the framed list to image2 column
       --Return new_train_label['image2']
       
       -MAJOR PART:
           -The image2 label is of the form "100_right_mir.jpeg" OR "100_right_90.jpeg"
            --we have to make it '100_right.jpeg' i.e remove '_90.jpeg' OR '_mir.jpeg'
            
           -We can split the the string from '_'
            --there are two '_' in the 100_right_mir.jpeg" so we have to split from 
              second '_' therefore use [0:2]
              ---Try:
                  test = "100_right_mir.jpeg"
                  print(test.split('_')[0:2])
              ---Return:
                  ['100', 'right']
                
            -To combine the splited string(['100', 'right']) use join
             --join the two list with '_'
               ---Try:
                   test = "100_right_mir.jpeg"
                   print('_'.join(test.split('_')[0:2]))
               ---Return:
                   100_right
                   
            -Use the above concept to apply through all label and use 
             --"lambda argument: expression"
               ---EG: (Without Lambda)
                   def add(x, y): 
                       return x + y
                   add(2, 3)
               
               ---EG: (With Lambda)
                   add = lambda x, y : x + y 
                   print add(2, 3)
                  
            -Locate(.loc) the label 'image2' and apply the lambda function on it
            
            -After the label is split and joined again there are still original files in 
             list which don't have '_mir.jpeg' OR '_90.jpeg' labels in it 
             i.e 100_right.jpeg(original file)
             
            -So to make it like other file in the list we have to remove the '.jpeg'
             extension from the list of original files and again add the '.jpeg'
             to all the files in the list.
             ---lambda x: '_'.join(x.split('_')).strip('.jpeg')
                -This will remove .jpeg extension form original file
                
             ---lambda x: '_'.join(x.split('_')).strip('.jpeg') + '.jpeg')
                -This will again add the '.jpeg' extension.
                -TRY PRINTING THE LABELS BEFORE AND AFTER ADDING THE '.JPEG'
                 SO YOU WILL GET IDEA
                 TRY:- 
                   'print(new_train_label['image2'])'
                   
            -Replace the 'image' column with the 'train_image_name'
             --'new_train_label.columns = ['train_image_name', 'image']'
            
            -Merge the dataFrame with original CSV.
             --pd.merge(left, right, how='outer', on=None)
               ---left: A DataFrame object.
               ---right: Another DataFrame object.
               ---on: Column or index level names to join on. Must be found in 
                  both the left and right DataFrame objects
               ---how: One of 'left', 'right', 'outer', 'inner'.
                  -outer:- Use union of keys from both frames
                  
        -Save .to_csv

'''

def get_images_list(file_path):
    # .DS_Store is for Mac Users
    return [i for i in os.listdir(file_path) if i != '.DS_Store']


if __name__ == '__main__':

    #read CSV (INCLUDE trainLabels_find_black_images.csv)
    trainLabels = pd.read_csv('E:/Tirth/trainlabel_zip/trainLabels_csv/trainLabels_find_black_images.csv')
    
    #get file name
    list_images = get_images_list('E:/Tirth/train/crop_256/')
    
    #framing the list_of_filename in sequence
    new_trainLabels = pd.DataFrame({'image': list_images})     
    new_trainLabels['image2'] = new_trainLabels.image
    
    #Remove the suffix from the image names.
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    
    #Strip and add .jpeg back into file name
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(
            lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')
    
    new_trainLabels.columns = ['train_image_name', 'image']
    
    trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
    trainLabels.drop(['black'], axis = 1, inplace = True)
    print(trainLabels.head(8))
    trainLabels = trainLabels.dropna()
    print(trainLabels.shape)
    
    print('Writing CSV')
    trainLabels.to_csv('E:/Tirth/trainlabel_zip/trainLabels_csv/trainlabel_master_v2.csv', index = False, header = True)
    
    
