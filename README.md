# Missing Semester RAG Assistant

A RAG-based learning assistant for [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/). Ask questions about the course content and get answers grounded in the official materials.

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your Anthropic API key**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_api_key_here
```

**3. Build the index (first time only)**
```bash
python cli.py --setup
```
This clones the Missing Semester course materials and builds a local vector store. Takes a few minutes on first run.

## Usage

**Interactive mode** — ask as many questions as you like:
```
python cli.py
```
```
> How do I use git to undo a commit?
> What is the difference between grep and sed?
> exit
```

**Single question**
```bash
python cli.py "How do I use tmux?"
```

## Other Commands

Rebuild the index from scratch (e.g. after updating course materials):
```bash
python cli.py --rebuild
```
