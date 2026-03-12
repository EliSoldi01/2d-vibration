for file in utils/*.py data_io/*.py analysis/*.py visualization/*.py; do
    echo "Diff $file"
    diff PHASE2/src/$file PHASE3/src/$file
done