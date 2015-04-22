#!/bin/bash

[[ ! -d "data" ]] && mkdir data

for prefix in train dtest etest; do
  for suffix in form tag lemma tree; do
    ln -s ~mpost/data/en600.468/inflect/data/$prefix.$suffix data/$prefix.$suffix
  done
done
