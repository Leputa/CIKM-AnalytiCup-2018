import tensorflow as tf

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    '''
    type(model_params):dict
    '''
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

def LCS(str_a, str_b):
    len_a = len(str_a)
    len_b = len(str_b)

    dp = [[0]*(len_b + 1) for i in range(len_a + 1)]

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if str_a[i-1] == str_b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    common_string = []
    index_a = len_a
    index_b = len_b
    while index_a >= 1 and index_b >= 1:
        if str_a[index_a - 1] == str_b[index_b -1]:
            common_string.append(str_a[index_a - 1])
            index_a -= 1
            index_b -= 1
        elif dp[index_a][index_b - 1] > dp[index_a - 1][index_b]:   # go to left
            index_b -= 1
        else:                                                       # go to up
            index_a -= 1

    common_string.reverse()
    return common_string



