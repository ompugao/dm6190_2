mkdir -p ./data-science-bowl-2018/
cd ./data-science-bowl-2018/
kaggle competitions download -c data-science-bowl-2018
unzip data-science-bowl-2018.zip
unzip stage1_train.zip -d stage1_train
unzip stage1_test.zip -d stage1_test
unzip stage1_train_labels.csv.zip 
unzip stage2_test_final.zip -d stage2_test_final

git clone --depth 1 https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes
mv kaggle-dsbowl-2018-dataset-fixes/stage1_train ./
