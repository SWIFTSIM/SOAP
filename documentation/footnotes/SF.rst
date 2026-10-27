**When distinguishing between star-forming and non star-forming gas and
computing the total star formation rate,** we have to be careful about the interpretation of the
``StarFormationRates`` dataset in the snapshots, since negative values in that dataset are used to store
another quantity, the last scale factor when that particular gas particle was star-forming. Star-forming gas
is then gas for which ``StarFormationRates`` is strictly positive, and the total star formation rate is the
sum of only the strictly positive values.
