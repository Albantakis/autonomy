import pyphi
from math import factorial

#Todo: The shapley values of the individual nodes should sum to the alpha of the purview as a whole.
# That is true, but only up to 10^-4 because of rounding in pyphi I assume.

def compute_shapley_values(CausalLink, transition, purview = None):

    if purview == None:
        purview = CausalLink.purview
    
    #Compute alpha values for all sets in the powerset of the purview nodes
    alpha_dict = {
        p: transition.find_mip(CausalLink.direction,CausalLink.mechanism, p).alpha 
        for p in pyphi.utils.powerset(purview)
    }
       
    shapley_values = []
    for node in purview:
        complement_set = tuple(sorted(set(purview) - set([node])))

        shapley = 0.
        for p in pyphi.utils.powerset(complement_set):

            # alpha(subset plus node) - alpha(subset without node)

            delta_alpha = alpha_dict[tuple(sorted(set(p).union(set([node]))))] - alpha_dict[p]

            shapley = shapley + shapley_factor(p,purview) * delta_alpha

        shapley_values.append(shapley)

    return shapley_values

def shapley_factor(S, F):
	# (|S|! (|F| - |S| - 1)!)/F!
	factor = factorial(len(S))*factorial(len(F)-len(S)-1)/factorial(len(F))
	return factor

	