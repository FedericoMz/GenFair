# GenFair
_A Genetic Fairness-Enhancing Data Generation Framework_

**[Read the paper](https://github.com/FedericoMz/GenFair/blob/main/GenFair%20Additional%20Materials.pdf)**
(and the [additional materials](https://github.com/FedericoMz/GenFair/blob/main/GenFair%20Tutorial.ipynb))

Do you want to use our algorithm? It's pretty easy, [**follow the tutorial!**](https://github.com/FedericoMz/GenFair/blob/main/GenFair%20Tutorial.ipynb)

## Background
Many datasets used to train machine learning models are biased towards a particular group defined by a sensitive attribute, such as Women (for _Gender_), Black people (for _Race_), or people below a certain _Age_ threshold. This human bias is propagated in the trained model, and therefore in its decision. Various fairness-enhancing algorithms have been proposed, operating either on the training set, on the model, or on the model’s output.

The _[Preferential Sampling](https://dtai.cs.kuleuven.be/events/Benelearn2010/submissions/benelearn2010_submission_18.pdf)_ algorithm attempts to “balance” the training dataset, mitigating its bias. The algorithm identifies four groups of instances: **D**iscriminated instances with a **P**ositive label (e.g., _Women, Hired_), **P**rivileged instances with a **N**egative label (e.g., _Men, Not-Hired_), **D**iscriminated instances with a **N**egative label, and **P**rivileged instances with a **P**ositive label. DN and PP instances near a model’s decision boundary are removed, whereas DP and PN instances are duplicated. It has been proved that a model trained on the balanced dataset makes more fair predictons.

## GenFair
Preferential Sampling has four key limitations: it only works with **_one_** sensitive attribute, it only works with **_binary_** sensitive attribute (_Race_ should be binarized), it only **_duplicates_** existing data, and the user must know beforehand the discriminated value. 

GenFair addresses all these points. The **Discrimination Test** identifies which values of a (potentially non-binary) sensitive attributes are discriminated, without the user’s input. In the **Instance Removal** step, instances closer to a model’s decision boundary are then removed, as in Preferential Sampling. The **Combination Test** finds the best combination of target values and values from multiple sensitive attributes in order to balance the dataset (e.g., _Black Woman, Hired_). These combinations are called constraint. Finally, a genetic algorithm called **_GenSyn_** (_Genetic Synthesiser_) creates new instances following the constraints, resulting in a fair dataset to be used with machine learning models.

![image](https://github.com/FedericoMz/GenFair/assets/80719913/2d2c1672-95bb-4c29-a8c3-ada39887eb48)

