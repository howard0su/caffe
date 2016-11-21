cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$HOME/data/Tiger
dst_file=test.txt
dataset=test

cd $root_dir

python $cur_dir/create_label.py
$cur_dir/../../build/tools/get_image_size $root_dir $dst_file $cur_dir/$dataset"_name_size.txt"
