def generate_config_string(network, query_workload, selectivities):
    # Load data from files
    nw = network
    wl= query_workload
    
    # Create the configuration as a multi-line string
    config_string = "network\n"
    for i in range(len(nw)):
        config_string += f"Node {i} {nw[i]}\n"
    config_string += "\nqueries\n"
    for query in wl:
        query = query.strip_NSEQ()
        config_string += f"{query.stripKL_simple()}\n"
    config_string += "\nmuse graph\n"
    config_string += "SELECT SEQ(A, B, C, D, E) FROM AND(B, SEQ(A, E, F)); I ON {1, 2, 4, 6, 7, 8, 9}/n(I)\n"
    config_string += "\nselectivities\n"
    config_string += str(selectivities)
    
    return config_string
