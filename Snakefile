rule all:
  input:
    "intermediates/flows.csv",
    "intermediates/duplicate_flows.pkl",
    "intermediates/openml-datasets-CC18.csv",
    "intermediates/runs.csv",
    "intermediates/interaction_table.csv",
  conda: "mapping.yaml"
  script: "demo.py"

rule get_openml_flows:
  output: "intermediates/flows.csv"
  conda: "mapping.yaml"
  script: "scripts/task01-get-openml-flows.py"

rule get_openml_runs_list:
  input: "intermediates/flows.csv"
  output: "intermediates/runs-list.csv"
  conda: "mapping.yaml"
  script: "scripts/task02-get-openml-runs-list.py"

rule get_openml_runs:
  input: "intermediates/runs-list.csv", "intermediates/openml-datasets-CC18.csv"
  output: "intermediates/runs-CC18/*.pkl", "intermediates/runs.csv"
  conda: "mapping.yaml"
  script: "scripts/task03-get-openml-runs.py"

rule get_openml_datasets_list:
  output: "intermediates/openml-datasets-CC18.csv"
  conda: "mapping.yaml"
  script: "scripts/task04-get-openml-datasets-list.py"

rule get_openml_datasets:
  input: "intermediates/openml-datasets-CC18.csv"
  output: expand("intermediates/datasets/*/{f}", f=['features.csv', 'target.csv', 'name'])
  conda: "mapping.yaml"
  script: "scripts/task05-get-openml-datasets.py"

rule profile_datasets:
  input: expand("intermediates/datasets/*/{f}", f=['features.csv', 'target.csv', 'name'])
  output: "intermediates/datasets/*/metafeatures.pkl"
  conda: "mapping.yaml"
  script: "scripts/task06-profile-datasets.py"

rule flows_to_dags:
  input: "intermediates/flows.csv"
  output: "intermediates/flows-as-graphs/"
  conda: "mapping.yaml"
  script: "scripts/task07-flows-to-dags.py"

rule datasets_to_kdtree:
  input:
    "intermediates/flows.csv",
    "intermediates/flows-as-graphs/"
  output:
    "intermediates/kdtree.pkl",
    "intermediates/openml_id_mapping.pkl",
    "intermediates/norms.pkl"
  conda: "mapping.yaml"
  script: "scripts/task08-graphs-to-kdtree.py"

rule graphs_to_bktree:
  input: "intermediates/flows.csv"
  output: "intermediates/bktree.pkl"
  conda: "mapping.yaml"
  script: "scripts/task09-graphs-to-bktree.py"

rule analyze:
  input:
    "intermediates/flows.csv",
    "intermediates/flows-as-graphs/",
  output:
    "intermediates/duplicate_flows.pkl",
    "intermediates/interaction_table.csv"
  conda: "mapping.yaml"
  script: "scripts/task10-analysis.py"
