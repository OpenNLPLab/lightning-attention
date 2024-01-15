date=$(date '+%Y-%m-%d-%H:%M:%S')

file=benchmark_srmsnorm
file=benchmark_lightning2

mkdir -p log

python ${file}.py  2>&1 | tee -a log/${date}-${file}.log
