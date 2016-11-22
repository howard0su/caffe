cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$HOME/data/Tiger
dst_file=test.txt
dataset=test

cd $root_dir

python $cur_dir/create_label.py
$cur_dir/../../build/tools/get_image_size $root_dir $dst_file $cur_dir/$dataset"_name_size.txt"
cp $dst_file $cur_dir/$dst_file

dst_file=trainval.txt
rand_file=$dst.random
cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
mv $rand_file $dst_file

cp $dst_file $cur_dir/$dst_file

