# confirmation-optimal-classification-trees

This repo holds code for building optimal classification trees as described in my confirmation of candidature document.

Basic structure:
- BendOCT.py and FlowOCT.py hold code for BendOCT and FlowOCT models respectively. Implementation code is split into the following types of classes (parent classes all found in OptimisationModels.py)
  - Base model which add variables, constraints, objective, and implement warm starts
  - Callback class which creates the callback function. The update_model method is used to modify the gurobi model object as required based on the settings, for example to set the LazyConstraints parameter to 1 or to set up a cache. The gen_callback method returns a callback function which is provided to Gurobi, this callback function only has access to the Gurobi model object and as such most variables of interest (decision variables, data) are attached to the model to make them available in the callback
  - Initial cut classes which implement valid inequalities. Key method is add_cuts, which takes in the gurobi model and adds the valid inequalities as constraints. It also implements an optimal gen_CompleteSolution method which returns a function which can fill in any variables added for the valid inequality given a partial solution. This is useful for when the warmstart or callback primal heuristic is not aware of any variables introduced by the initial cut. if the initial cut introduces a large amount of variables, particularly integral variables, this prevents Gurobi from spending a significant amount of time attempting to complete the solution itself
- BendOCTWrapper/FlowOCTWrapper functions set up the BendOCT/FlowOCT model objects with the provided settings. It automatically stores results and gurobi logs in the Results folder.
- The wrapper functions are called from BendOCT_run and FlowOCT_run with a desired hyperparameter grid, list of datasets, optimisation parameters and gurobi parameters.
- Processing of results done by HPBenchmarkTable.py and HPBenchmarkFigure.py. The interface to generate result plots and tables is somewhat convoluted at the moment and will likely be rewritten in future. 
