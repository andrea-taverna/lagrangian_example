# Lagrangian Example

Sample code to showcase:
* Mathematical Programming and Mixed-Integer Programming
* Lagrangian decomposition
* Stochastic programming

applied to "power plant scheduling problem" a.k.a. the Unit Commitment Problem (UCP).

The code has been used to produce content for some of my talks. See slides deck [ATaverna_DLI_2022.pdf](https://github.com/andrea-taverna/lagrangian_example/blob/master/ATaverna_DLI_2022.pdf) for the 
[talk](https://talks.codemotion.com/operations-research-the-scalable-ai-for-?view=true) at
DeepLearning Italia / Codemotion Italy in 2022 (audio in Italian).

# UCP Data
The example data for the UCP has been generated using the ["ucig" code](https://commalab.di.unipi.it/files/Data/UC/ucig.tgz) 
available at [prof. Frangioni's CommaLAB webpage](https://commalab.di.unipi.it/datasets/UC/)
under "Deterministic Unit Commitment".

# Usage

Install requirements via `requirements.txt`.

Content is in runnable notebooks:
* `Knapsack` for knapsack problem
* `UCP`, `UCP_Lagrangian` and `UCP_Stochastic` for UCP.
* `Non-smooth Dual Example` shows an example of non-smooth problem, written in Markdown/LaTeX


# License

Licensed according to MIT licence, see `LICENSE.txt`
