Version: Python 2

How to run: python hmm_wembeddings.py (-v)

Files:
- hmm_wembeddings.py (program)
- output.txt (transition probabilities, emission probabilities and purity table)
- README
- toy.txt (data)

Functions:

Almost all functions were taken from HMM - they were only modified to account for multiple states as the initial HMM program was only a 2-state implementation. 

* state_filter: creates a state by key b_prob dict (which is originally key by state). Also only looks at state emission probabilities that are > 0.001.

* print_em: prints emission probabilities.

* calc_purity: maps the parts of speech tags to the state emission probability dict with words. calculates the average probability for each state. 

* print_transitions: prints transition probabilities

* print_purity: prints average purity table for each of the 24 iterations.