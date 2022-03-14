## Running the server


For the first time, run:
```bash
pip install -r requirments.txt
```
Each time you open a new terminal session, run:

```bash
set FLASK_APP=app.py;
```

To run the server, execute:

```bash
flask run --reload
```

The `--reload` flag will detect file changes and restart the server automatically.

## Use the server

http://127.0.0.1:5000/ this path returns general statistics 

http://127.0.0.1:5000/skill this path returns specific statistics about a skill


