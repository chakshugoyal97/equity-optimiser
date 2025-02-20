## Running this notebook in VS Code

You will need the Jupyter extension in VS Code for this.
From the root dir of this repo, after following steps in main README.md to run the tests, do -

```bash
uv pip install -e .
```

Preparing all the data, by running download_data script. 
Note that the data is cached in pickle file unless you delete it.
```bash
python ./workbooks/download_data.py
```

Then, play with the notebook!