import config
import tensorflow as tf
import os
from zipfile import ZipFile
from glob import glob
from termcolor import colored

if __name__=="__main__":
    summ_path = config.result_path
    if not os.path.exists(summ_path):
        os.makedirs(summ_path)
    with ZipFile(summ_path + "code.zip", "w") as zip_fn:
        for fn in glob(config.project_path + "**/*.py", recursive=True):
            zip_fn.write(filename=fn)
        log = ", ".join("%s: %s" % item for item in vars(config).items() if not item[0].startswith("__"))
        zip_fn.writestr("logs.txt", log)
    print(colored("Started at %s"%summ_path, "red"))
    import train_seg, train_stage
    if config.mode==0:
        trainer = train_seg.Trainer(summ_path)
    else:
        trainer = train_stage.Trainer(summ_path)
    train_summary_writer = tf.summary.create_file_writer(summ_path)
    with train_summary_writer.as_default():
        trainer.train()