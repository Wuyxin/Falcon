def query_index(lst, query):
    try:
        return lst.index(query)
    except:
        return None


# Rule 1: if we don't use pooling, then pool_loop shall be -100
def check_rule1(stc: list, labels):
    pool_loop_idx = query_index(labels, 'pool_loop')
    pool_type_idx = query_index(labels, 'pool_type')
    check_flag = pool_loop_idx and pool_type_idx
    if check_flag and \
        ((stc[pool_type_idx] == 'none' and stc[pool_loop_idx] > 0) or \
        (stc[pool_type_idx] != 'none' and stc[pool_loop_idx] < 0)):
        return False 
    return True
    

# Rule 2: pool.loop < # message passing layers
def check_rule2(stc: list, labels):
    pool_loop_idx = query_index(labels, 'pool_loop')
    layers_mp_idx = query_index(labels, 'layers_mp')
    check_flag = pool_loop_idx and layers_mp_idx
    if check_flag and stc[pool_loop_idx] > stc[layers_mp_idx]:
        return False 
    return True
    
    
RULE_LIST = [check_rule1, check_rule2]
def check_all_rules(stc: list, labels):

    for rule in RULE_LIST:
        if rule(stc, labels) == False:
            return False
    return True

