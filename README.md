# Clustering-DNA
Two algorithms were compared in their ability to accurately cluster copies of centroids with varying error rates.
The description of the Q-gram Algorithm comes from "Clustering Billions of Reads for DNA Data Storage" (Rashtchian et al., 2017).
The description of the LSH Algorithm comes from "Low cost DNA data storage using photolithographic synthesis and advanced information reconstruction and error correction" (Antkowiak et al., 2020). Using this code, the Q-gram Algorithm performs better in terms of time complexity and accuracy, with accuracy being measured as the proportion of true clusters which have a corresponding cluster produced by our clustering algorithms that is the same as the true cluster. Both algorithms were evaluated using sequences with low error-rates (deletion rate of 0.0005) common to some sequencing methods that produce short reads, medium error-rates (insertion rate = 0.017, deletion rate = 0.03, substitution rate = 0.022) common to sequencing methods which can produce longer reads with more efficiency. Both alrogithms were also evaluated on real sequenced data from Microsoft. 
