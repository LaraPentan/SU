#! /bin/bash

cd ../src

python main.py $@

cd ../output

cat greske.dat
