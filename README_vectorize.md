# Transcription Vectorizer

This script uses Ollama to vectorize transcription files and saves the embeddings locally.

## Prerequisites

1. Ollama must be installed locally. You can download it from [https://ollama.ai/](https://ollama.ai/)
2. Python 3.7+ with the following packages:
   - numpy
   - tqdm
   - requests

## Installation

Make sure Ollama is running. You can start it with:

```bash
ollama serve
```

You also need to have a suitable embedding model. The script uses `mxbai-embed-large:335m` by default, but you can specify other models.
To pull a model with Ollama:

```bash
ollama pull mxbai-embed-large:335m
```

## Usage

### Basic Usage

```bash
python vectorize_transcriptions.py
```

This will:
1. Process all `.txt` files in the `transcriptions/` directory
2. Use the `mxbai-embed-large:335m` model to generate embeddings
3. Save the embeddings to the `embeddings/` directory

### Continuous Monitoring Mode

You can run the script in continuous mode to automatically detect and process new transcription files:

```bash
python vectorize_transcriptions.py --continuous
```

In continuous mode, the script will:
1. Process any existing transcription files that haven't been vectorized yet
2. Then sleep for a specified interval (default: 5 minutes)
3. Wake up and check for new transcription files
4. Process any new files and go back to sleep
5. Repeat until you stop the script with Ctrl+C

This is useful for automatically vectorizing new transcriptions as they are created by the TransFixer tool.

### Command-line Options

```
--model MODEL               Ollama model to use for embeddings (default: mxbai-embed-large:335m)
--transcriptions_dir DIR    Directory containing transcription files (default: transcriptions)
--output_dir DIR            Directory to save embeddings (default: embeddings)
--continuous                Run in continuous mode, periodically checking for new files
--interval SECONDS          Seconds to wait between checks in continuous mode (default: 300)
```

### Examples

Process all files once and exit:
```bash
python vectorize_transcriptions.py --model mxbai-embed-large:335m
```

Run in continuous mode, checking every 2 minutes:
```bash
python vectorize_transcriptions.py --continuous --interval 120
```

Use a different model and custom directories:
```bash
python vectorize_transcriptions.py --model nomic-embed-text --transcriptions_dir my_transcripts --output_dir my_vectors
```

## Output Format

For each transcription file, the script creates a `.pkl` file containing:

1. The embedding vector as a numpy array
2. Metadata including:
   - Source file path
   - Model used
   - Timestamp
   - File size
   - Vector dimension

The directory structure of the source files is preserved in the output directory.

A `processing_summary.json` file is also created in the output directory with overall statistics.

## Troubleshooting

1. If Ollama is not running, you'll see an error message. Start Ollama with `ollama serve`.
2. If the specified model is not available, you'll be prompted to pull it.
3. Failed file processing is reported in the console output.
4. To stop the script in continuous mode, press Ctrl+C. The script will finish processing the current batch before exiting. 