import os

do_save = eval(os.getenv("do_save", "False"))
do_show = eval(os.getenv("do_show", "False"))
artifact_dir = os.getenv("artifact_dir", "artifact")
os.makedirs(artifact_dir, exist_ok=True)