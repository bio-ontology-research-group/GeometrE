import sys
import re

def extract_mrr_results(filename, transitive_subset=False):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Dictionary to store results for each step
    step_results = {}
    
    # Find all step numbers that have MRR results
    step_pattern = re.compile(r'Test 1p MRR at step (\d+):')
    steps = step_pattern.findall(content)
    
    # Add "average" to the query types
    query_types = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2u-DNF', 'up-DNF', '2in', '3in', 'inp', 'pin', 'pni', 'average']
    
    # For each step, extract all MRR values
    for step in steps:
        metrics = {query_type: None for query_type in query_types}
        
        # Extract each query type separately to ensure we get all of them
        for query_type in query_types:
            if transitive_subset:
                pattern = re.compile(f'TestTr {query_type} MRR at step {step}: ([0-9.]+)')
            else:
                pattern = re.compile(f'Test {query_type} MRR at step {step}: ([0-9.]+)')
            match = pattern.search(content)
            if match:
                metrics[query_type] = float(match.group(1))
        
        # Convert to percentages
        metrics = {k: v*100 if v is not None else None for k, v in metrics.items()}
        step_results[step] = metrics
    
    return step_results

def print_latex_table(step_results):
    # Add "average" to the query types
    query_types = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2u-DNF', 'up-DNF', '2in', '3in', 'inp', 'pin', 'pni', 'average']

    # Print LaTeX-friendly table format
    for step, metrics in sorted(step_results.items(), key=lambda x: int(x[0])):
        print(f"Step {step}:")
        header = []
        values = []

        for qt in query_types:
            if metrics[qt] is not None:
                header.append(qt)
                values.append(metrics[qt])
                 
        # header = "\t&\t".join(query_types)
        # values = "\t&\t".join([f"{metrics[qt]:.1f}" if metrics[qt] is not None else "N/A" for qt in query_types])
        header = "\t&\t".join(header)
        values = "\t&\t".join([f"{values[i]:.1f}" for i in range(len(values))])
        

        print(header)
        print(values)
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename> <transitive_subset (optional)>")
        sys.exit(1)
    
    filename = sys.argv[1]
    transitive_subset = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
    step_results = extract_mrr_results(filename, transitive_subset=transitive_subset)
    print_latex_table(step_results)


    
