# Instructions for students

In this assignment, your task is to take a sequence of Czech lemmas and produce the correct
inflection for each of them. Since [the data used in the
assignment](https://ufal.mff.cuni.cz/pdt2.0/) is licensed under the LDC, the assignment is written
to require an account on the CLSP servers. Please contact Matt Post if you would like information on
how to extract the PDT information from your own copy of the [LDC
catalog](http://catalog.ldc.upenn.edu/LDC2006T01).

Originally developed for our [Machine Translation course](http://mt-class.org/jhu/hw5.html) at Johns
Hopkins University.

# Setup instructions (for instructors)

To setup the data, you need access to LDC2006T01. Set the environment variable
`$LDC2006T01` to point to the location of this data, then run

    ./scripts/extract.sh

You need Python 2.7. If you wish to parallelize the extraction (which takes a few hours),
see that script for more info. Once the extraction is done, gather all the individually
extracted files together:

    mkdir data
    for dir in train dtest etest; do 
      for type in lemma form tag tree; do 
        cat plain/$LDC2006T01/data/full/amw/$dir*/*.$type > data/$dir.$type
      done
    done

This will create files in the `data/` subdirectory corresponding to the files indicated
in the assignment instructions.

Make sure to change the permissions of the test data so prevent peeking at the test data.

    chmod 600 data/etest.form

