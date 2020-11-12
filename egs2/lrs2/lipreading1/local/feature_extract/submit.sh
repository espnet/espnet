source activate ml3

job_name=$1
split_id=$2

num_cpu=12
num_gpu=1
start_time=`date -Iminutes`



sbatch -J $job_name -c $num_cpu \
    -p gpu --gres=gpu:$num_gpu \
    -x gqxx-01-002 \
    -o ./log/$job_name.log \
    -e ./log/$job_name.log \
    ./run.sh $split_id
