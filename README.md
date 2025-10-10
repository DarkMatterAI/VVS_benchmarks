# VVS Benchmarks

Benchmark code for the paper [Efficient Search of Ultra-Large Synthesis On-Demand Libraries with Chemical Language Models](https://www.biorxiv.org/content/10.1101/2025.09.04.674350v1.full)

Code subject to change during peer review process

Order of operations:

1. Create artifacts
    * Downloads files and creates datasets. Note dataset creation is slow
    * `cd src/src_processing`
    * `./run_download_files.sh` (note slow)
2. Model training
    * `cd src/model_training`
    * Train erbb1 mlp model
        * `cd erbb1_mlp/`
        * `./process_dataset.sh`
        * `./train_erbb1_mlp.sh`
    * Train embedding compression
        * `cd embedding_compression/`
        * `./process_dataset.sh`
        * update `sweep_space.yaml` as desired
        * `./train_model.sh`
    * Train decomposer model
        * `cd enamine_decomposer`
        * `./process_dataset.sh`
        * update `sweep_space.yaml` as desired
        * `./train_model.sh`
3. Model Benchmarks/Analysis
    * `cd src/data_analysis`
    * Embedding compression analysis
        * `./src/model_evaluation/embedding_compression/run.sh generate` - generate analysis data
        * `./src/model_evaluation/embedding_compression/run.sh plot` - plot results
    * Enamine decomposer analysis
        * `./src/model_evaluation/enamine_decomposer/run.sh generate` - generate analysis data
        * `./src/model_evaluation/enamine_decomposer/run.sh plot` - plot results
4. Set up scoring endpoints
    * Set up OpenEye license file
        * Place OpenEye license at `./src/score_consumer/secrets/oe_license.txt`
    * If running with VVS:
        * Start VVS with Qdrant and Triton plugins enabled
        * `cd score_consumer/`
        * `./manage_records.sh create`
        * `docker compose up -d --build`
        * `docker compose scale score_consumer={SCORE_WORKERS}` for the desired number of parallel workers
    * If running without VVS:
        * Note the `vvs_platform` cannot be run without VVS
        * `cd score_consumer`
        * `docker compose -f docker-compose-self-contained.yml` up -d --build
        * `docker compose -f docker-compose-self-contained.yml scale score_consumer={SCORE_WORKERS}` for the desired number of parallel workers
5. BBKNN Benchmarks
    * Benchmarks
        * `cd src/bbknn`
        * `./run {bbknn|natprod|egfr}`
            * Note EGFR benchmark requires scoring functions to be set up
    * Analysis
        * `cd src/data_analysis`
        * `./src/bbknn/run.sh {cosine|sampling|egfr}`
6. VVS Local Benchmarks
    * Local benchmarks require scoring endpoints to be running
    * `cd src/vvs_local`
    * Build indices (required for later benchmarks)
        * `./run.sh generate_indices`
    * Embedding size latency benchmarks
        * `./run.sh compression_eval`
    * Learning rate sweeps
        * `./run.sh lr_sweep --cfg lr_sweep_bbknn.yaml`
        * `./run.sh lr_sweep --cfg lr_sweep_knn.yaml`
    * VVS Local Benchmarks
        * `./run.sh hyperparam_sweep --cfg bbknn_sweep.yaml --run_type sweep`
    * Analysis
        * `cd src/data_analysis`
        * `./src/vvs_local/run.sh {sweep|embed}`
7. Other Benchmarks
    * Run SyntheMol benchmarks
        * `cd benchmarks_synthemol/`
        * `./run.sh sweep` 
    * Run Thompson Sampling benchmarks
        * `cd benchmarks_ts/`
        * `./run.sh sweep` 
    * Run RxnFlow benchmarks
        * `cd benchmarks_rxnflow/`
        * `./run.sh sweep` 
    * Run RAD benchmarks
        * `cd benchmarks_rad/`
        * `./process_dataset.sh`
        * `./run.sh sweep`
    * Compile best hyperparameters
        * `cd data_analysis`
        * `./src/benchmarks/run.sh sweep`
    * Run Final benchmarks
        * Re-run the `./run.sh sweep` command as as `./run.sh final`
    * Plot results
        * `cd data_analysis`
        * `./src/benchmarks/run.sh final`

