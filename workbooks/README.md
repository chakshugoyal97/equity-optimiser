## Running this notebook in VS Code

- From the root dir of this repo, after following steps in main README.md so you are able to run the tests, do -

    ```bash
    uv pip install -e .
    python -m ipykernel install --user --name=my-env-name
    ```

- [Optional] Refresh all the data, by running download_data script. Note that the data is cached in pickle file unless you delete it.
    ```bash
    python ./workbooks/download_data.py
    ```

- Install an ipykernel in your venv and lunch jupyterlab
    ```bash
    python -m ipykernel install --user --name=equity_optimiser
    ```
    ```bash
    jupyter lab
    ```

- Select the kernel that you created in Jupyter from Kernel -> Change Kernel... 
Then, play with the notebook!