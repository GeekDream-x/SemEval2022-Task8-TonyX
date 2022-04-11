import logging
import argparse
import transformers
from utils import load
from models import Reg_FT_Configer, Reg_Trainer

transformers.logging.set_verbosity_error()


def main():

    parser = argparse.ArgumentParser(description="Reg-Argparser")
    parser.add_argument('--params', type=str, required=True, help="JSON dict of parameters for model training.")
    args = parser.parse_args() 

    params = load(args.params)

    logging.basicConfig(filename=params['log_path'], filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.DEBUG)

    logging.info(f"Params:\n{params}")


    config = Reg_FT_Configer(params)
    trainer = Reg_Trainer(config)
    trainer.run_finetune()

if __name__ == "__main__":
    main()
