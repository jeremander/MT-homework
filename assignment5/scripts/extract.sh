#!/bin/bash

# Usage: two modes:
#
# 1. Just invoke the script. Make sure to define $LDC2006T01. All files will be processed
# and placed under plain/$file
#
# 2. Invoke it with the full path to a file under $LDC2006T01, which will then be processed.
# 
# Afterwards, aggregate all the individual ones with:
#
# mkdir data
# for dir in train dtest etest; do
#   for type in lemma form tag tree; do
#     cat plain/$LDC2006T01/data/full/mw/$dir*/*.$type > data/$dir.$type
#   done
# done

#$ -q text.q
#$ -l h_vmem=1G,mem_free=1G,h_rt=02:00
#$ -l num_proc=1
#$ -S /bin/bash
#$ -V -j y -o /dev/null
#$ -cwd

extract() {
  file=$1
  dir=$(dirname $file)
  [[ ! -d "plain/$dir" ]] && mkdir -p plain/$dir

  if [[ ! -s plain/$file.form ]]; then
    echo "building plain/$file.form"
    python2.7 scripts/pull.py $file -tag form > plain/$file.form
  fi
  if [[ ! -s plain/$file.lemma ]]; then 
    echo "building plain/$file.lemma"
    python2.7 scripts/pull.py $file -tag lemma > plain/$file.lemma
  fi
  if [[ ! -s plain/$file.tag ]]; then
    echo "building plain/$file.tag"
    python2.7 scripts/pull.py $file -tag tag | perl -ane '@F = map {/(..)/} @F; print join(" ", @F), $/' > plain/$file.tag
  fi
  if [[ ! -s plain/$file.tree]]; then
    echo "building plain/$file.tree"
    python2.7 scripts/pull_tree.py $file > plain/$file.tree
  fi
}

if [[ ! -z "$1" ]]; then
  extract $1
elif [[ -z $LDC2006T01 ]]; then
  echo "Please define LDC2006T01 to point to the root of that LDC release."
  exit
else
  for file in $LDC2006T01/data/full/amw/*/*.m.gz; 
    do extract $file
  done
fi
