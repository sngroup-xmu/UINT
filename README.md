# UINT
A Unified Framework for Verifying and Interpreting Learning-Based Networking Systems


## **Overview:**
UINT is designed to allow a operator to verify common robustness properties and answer interpretability problems of networking systems.

This repository contains:
- log: The result files;
- model_file: The trained weights file of the verified and interpreted ML systems;
- result: Plotting functions for visualizing experimental data;
- src: The main part of the project
   - encoder: Complete model SMT encoding and properties encoding;
   - exp: Complete verification and interpretation of systems;
   - utils: Generic files;


## **Geting Started:**

### **1. SMT encoding**
- We use src/encoder/Encoder.py to define the input and output SMT variables as __in_vars_, __out_vars_,
the network encoding variables as __nn_vars_.

- Network model encodings are defined as __nn_constrs_:
input and output layers are valid ranges for input and output variables, then step-by-step layered coding to add intermediate variables.

- Properties encoding are defined as __in_out_constrs_: 
use the functions argmax_eq_constr(), argmax_neq_constr(), eq_constr(), approximate_eq_constr(), assign_constr(), boundary_constr(), discrete_constr(), etc. to complete the property encoding.

### **2. Verify and interpret the specific system**
Execute **python pensieve.py/aurora.py** file in src/exp directory to implement each property verification and interpretation problem.

### **3. Adding a new system**
If you want to add a new system, you need to add the post-training weight file for the new system in the model_file directory, add the parsing code in the utils/nn.py file according to the specific model structure, and add the code file for calling the corresponding property encoding and interpretable problems in the src/exp directory.


