import Scripts/Output/results as mod_results # import model results

def find_max_values(dict):
    """
    find_max_values: Returns highest value of dict and corresponding probablities
    """
    max_mvc = max(mod_res.NN_List['movement_probs']) #find highest likely hood movement class
    #max_mvc_prob = max(mod_res.NN_List['movement_probs']) #find highest likely hood movement class probablity
    max_sv = max(mod_res.NN_List['severity_probs']) #find highest likely hood severity
    #max_sv_prob = (mod_res.NN_List['severity_probs']) #find highest likely hood severity probablity
    movement = [max_mvc_prob]
