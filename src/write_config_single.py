import io

def generate_config_buffer(network, query_workload, selectivities):
    # Create an in-memory file-like object
    config_buffer = io.StringIO()

    # Write the configuration to the buffer
    config_buffer.write("network\n")
    for i in range(len(network)):
        config_buffer.write(f"Node {i} {network[i]}\n")
    
    config_buffer.write("\nqueries\n")
    for query in query_workload:
        query = query.strip_NSEQ()
        config_buffer.write(f"{query.stripKL_simple()}\n")
    
    config_buffer.write("\nmuse graph\n")
    config_buffer.write("SELECT SEQ(A, B, C, D, E) FROM AND(B, SEQ(A, E, F)); I ON {1, 2, 4, 6, 7, 8, 9}/n(I)\n")
    
    config_buffer.write("\nselectivities\n")
    config_buffer.write(str(selectivities))

    # Reset the buffer's position to the beginning
    config_buffer.seek(0)

    return config_buffer
