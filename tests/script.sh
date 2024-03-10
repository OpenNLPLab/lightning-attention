date=$(date '+%Y-%m-%d-%H:%M:%S')

folder=ops
file=test_srmsnorm
file=test_lightning2
file=test_lightning2_no_decay

mkdir -p $folder/log

pytest $folder/${file}.py  2>&1 | tee -a $folder/log/${date}-${file}.log
