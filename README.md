# Assisted Design of DS Pipelines

This repository is structured as follows:

```
scripts/                    <-- supporting scripts to populate the DS experiments from OpenML
Snakefile                   <-- an automation script to populate the data
data_summarizer.py          <-- main code for lightweight data footprints
demo.py                     <-- prototype demonstration
evaluation.py               <-- evaluation setup
experiment_database.py      <-- main code for the experiment store
mapping.yaml   
recommendation_engine.py    <-- main code for the recommendation engine
requirements.txt
runs.csv                    <-- collected experiment runs for reproducibility purposes
search_strategies.py        <-- strategies to simulate user behaviour
task11-plots.ipynb          <-- data visualization
utils.py                    <-- supporting utilities
```

use ```snakemake --use-conda``` to generate the data, [check snakemake documentation](https://snakemake.readthedocs.io/en/stable/).
