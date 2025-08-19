# llm_case_study

Studiu de caz asupra răspunsurilor oferite de modelele lingvistice mari și a înclinațiilor acestora în cadrul anumitor întrebări controversate ce aparțin de istoria României

## Usage

### Running the bulk test

```bash
python main.py
```

### Checkpoint System

The script now supports resuming from where it left off:

-   **Automatic checkpointing**: Progress is saved every 5 tests automatically
-   **Graceful shutdown**: Press `Ctrl+C` to stop the script safely - it will save progress and allow you to resume later
-   **Resume on restart**: When you run the script again, it will automatically detect the checkpoint and continue from where it stopped

### Files created:

-   `bulk_test_results.json` - Final results (created when all tests complete)
-   `bulk_test_checkpoint.json` - Temporary checkpoint file (automatically removed when tests complete)

### Starting fresh:

If you want to start over from the beginning, run:

```bash
python cleanup_checkpoint.py
```

This will remove the checkpoint file so the next run starts fresh.
