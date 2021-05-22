This is a distribution of randomly-generated, realistic instances of the
ramp-constrained, hydro-thermal Unit Commitment problem. The distribution
comes with the following files:

- Format.pdf: a document describing the format of the instances, as well as
  giving references to articles where they are used and described;
  NOTE: there is an error in the document: right after the integer (which
  can be positive or negative) describing the initial value of the unit,
  the two subsequent integers denote the minimum up- and down-time of the
  unit. This is not correctly described in the document, which therefore
  only describes 14 of the 16 numerical parameters of a thermal unit
  (two of which, the last two ones, are actually unused). 

- T-Ramp: a directory containing pure thermal (i.e., without hydro units)
  instances.

- HT-Ramp: a directory containing hydro-thermal instances.

Each instance is named according to the scheme

     T-H-N.mod

where T is the number of thermal units (between 10 and 200), H is the number
of hydro units (between 10 and 100), and N is an integer (between 1 and 5 for
pure thermal instances, between 1 and 2 for hydro-thermal ones) distinguishing
different instances of the same size between them. Pure thermal instances all
have H = 0.

