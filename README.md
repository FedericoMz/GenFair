# GenFair

## The Fairness Problem
Many datasets used to train machine learning models are biased towards a particular group defined by a sensitive attribute, such as Women (for Gender), Black people (for Race), or people below a certain Age threshold. 
This human bias is propagated in the trained model, and therefore in its decision. Various fairness-enhancing algorithms have been proposed, operating either on the training set, on the model, or on the model’s output.
The Preferential Sampling algorithm attempts to “balance” the training dataset, mitigating its bias. The algorithm identifies four groups of instances: Discriminated instances with a Positive label (e.g., Women, Hired), Privileged instances with a Negative label (e.g., Men, Not-Hired), Discriminated instances with a Negative label, and Privileged instances with a Positive label.
DN and PP instances near a model’s decision boundary are removed, whereas DP and PN instances are duplicated. It has been proved that a model trained on the “balanced” dataset makes more fair predictons.
