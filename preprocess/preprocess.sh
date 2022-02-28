#Copied From [Object-DeforNet](https://https://github.com/mentian/object-deformnet)
echo "python shape_data.py"
python shape_data.py

#Copied From [Object-DeforNet](https://https://github.com/mentian/object-deformnet)
echo "python pose_data.py"
python pose_data.py

#remove the outliners points for the training set
echo "python outliner_removal_train.py"
python outliner_removal_train.py

#remove the outliners points for the test set. Note that the processed data is not used durning training and test
#We keep this step only for the convenience of dataloder.
echo "python outliner_removal_test.py"
python outliner_removal_test.py

#flip the real training data 
echo "python flip_data.py"
python flip_data.py

#generate the label for the flipped data. 
#Note that the calculated objcet pose of the filpped data is wrong, but it is not used during our self-supervised training.
#We keep this step only for the convenience of dataloder.
echo "python pose_data_flipped.py"
python pose_data_flipped.py 