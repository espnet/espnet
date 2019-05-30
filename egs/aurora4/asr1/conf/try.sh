train_config=train.yaml
t = basename ${train_config%.*}
echo ${t}
