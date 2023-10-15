# GenFair

## The Fairness Problem
Many datasets used to train machine learning models are biased towards a particular group defined by a sensitive attribute, such as Women (for _Gender_), Black people (for _Race_), or people below a certain _Age_ threshold. 

This human bias is propagated in the trained model, and therefore in its decision. Various fairness-enhancing algorithms have been proposed, operating either on the training set, on the model, or on the model’s output.

The Preferential Sampling algorithm attempts to “balance” the training dataset, mitigating its bias. The algorithm identifies four groups of instances: **D**iscriminated instances with a **P**ositive label (e.g., _Women, Hired_), **P**rivileged instances with a **N**egative label (e.g., _Men, Not-Hired_), **D**iscriminated instances with a **N**egative label, and **P**rivileged instances with a **P**ositive label.

DN and PP instances near a model’s decision boundary are removed, whereas DP and PN instances are duplicated. It has been proved that a model trained on the “balanced” dataset makes more fair predictons.
