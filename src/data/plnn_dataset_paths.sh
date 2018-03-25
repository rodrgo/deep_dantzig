
#!/bin/bash

PLNN_DIR=$1
trap "exit" INT
for prop in $(find $PLNN_DIR/planet/benchmarks/ACAS/ -name "*.rlv" | sort);
do
    outfile=$(echo $prop | gawk 'match($0, /(property[0-9]+\/.+)\.rlv/, arr) {print "results/ACAS/BaB/" arr[1] ".txt"}')
    if [ ! -f $outfile ]; then
        echo "$prop $outfile"
    fi
done

coll_idx=1
for prop in $(find $PLNN_DIR/planet/benchmarks/collisionDetection/ -name "*.rlv"| sort);
do
    target_fname=$coll_idx-$(basename $prop .rlv)
    outfile="results/collisionDetection/BaB/$target_fname.txt"
    if [ ! -f $outfile ]; then
        echo "$prop $outfile"
    fi
    coll_idx=$(($coll_idx + 1))
done

for prop in $(find $PLNN_DIR/planet/benchmarks/twinLadder/ -name "*.rlv"| sort);
do
    target_fname=$(basename $prop .rlv)
    outfile="results/twinLadder/BaB/$target_fname.txt"
    if [ ! -f $outfile ]; then
        echo "$prop $outfile"
    fi
done
