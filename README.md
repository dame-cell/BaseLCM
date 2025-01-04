# BaseLCM
This will be the implementation of LCM but only the transformers-base LCM

<p align="center">
  <img src="src/lcm.png" alt="lcm" width="400"/>
</p>


From my understanding the LCM  basically focuses more on sentences rather than tokens and since it focuses more on sentences it is able to bypass the transformers quadratic scaling issue  with the increase in sequence length, but also since it works only on sentences it has its limitations like  cannot really focus  on the entire context but this seems to be useful for meta social media app where users typically communicate through short text


So the baselcm has three components :
1) Pre-net : which normalizes the input SONAR embeddings and maps them to the model’s hidden dimension 
2) Transformer-decoder : which transduces a sequence of preceding concepts (read sentence embeddings) into a sequence of future
ones
1) Post-net : which maps the hidden representations produced by the Transformer-Decoder back to the original embedding space
  
The workflow is as follows:
1) First we download the dataset any dataset from hugginface 
2) We then split those chunks of texts into sentences using spacy 
3) Then we pass those sentences to the sonar encoder 
4) We then add noise to those embeddings provided by the sonar model 
5) We then train the model after this 

### Loss Function and Training Objective

The **Loss Function** in Base-LCM is the **Mean Squared Error (MSE)**, which measures the difference between the predicted next concept embedding and the true next concept embedding. The model is trained to minimize this loss, effectively learning to predict the next concept in a sequence.

The **Training Objective** is to optimize the model's parameters \(\theta\) so that it can accurately predict the next concept \(x_n\) from a sequence \(x_{<n}\).

### Getting started 


# Citations 

```bash
@article{lcm2024,
  author = {{LCM team}, Lo\"{i}c Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R. Costa-juss\`{a}, David Dale, Hady Elsahar, Kevin Heffernan, Jo\~{a}o Maria Janeiro, Tuan Tran, Christophe Ropers, Eduardo Sánchez, Robin San Roman, Alexandre Mourachko, Safiyyah Saleem, Holger Schwenk},
  title = {{Large Concept Models}: Language Modeling in a Sentence Representation Space},
  publisher = {arXiv},
  year = {2024},
  url = {https://arxiv.org/abs/2412.08821},
}
```
```bash
@misc{Duquenne:2023:sonar_arxiv,
  author = {Paul-Ambroise Duquenne and Holger Schwenk and Benoit Sagot},
  title = {{SONAR:} Sentence-Level Multimodal and Language-Agnostic Representations},
  publisher = {arXiv},
  year = {2023},
  url = {https://arxiv.org/abs/2308.11466},
}
```
