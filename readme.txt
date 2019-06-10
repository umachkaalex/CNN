******************Setup******************

1. Install requirements
2. Unzip numbers.rar in 'img_data' folder
3. Run CNN_model.ipynb

******************Adding your images******************

If you want to use your images:
1. Put them into 'img_data' folder as one folder (for example, "cats_dogs")
2. The structure of the subfolders have to be the following:
   - '...\cats_dogs\train\' path for train images,
   - '...\cats_dogs\test\' path for test images.
3. Inside test and train folder have to be the same number of folders (thet will be defined as classes):
   - '...\cats_dogs\train\dogs\' path for train images of dogs
   - '...\cats_dogs\train\cats\' path for train images of cats

******************Running your CNN configuration******************

In 'CNN_model.ipynb' yupyter notebook or 'CNN_model.py' file you can play with different combination of options for Conv2D layers to find best for your task.
