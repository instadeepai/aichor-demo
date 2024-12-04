import os
import logging
import traceback
from typing import Dict, Any

import xgboost as xgb
from tracker import RabitTracker

from train_data import read_train_data
from dump_model import dump_model


logger = logging.getLogger(__name__)




def extract_xgbooost_cluster_env():
    """
    Extract the cluster env from pod
    :return: the related cluster env to build rabit
    """

    logger.info("starting to extract system env")

    master_addr = os.environ.get("MASTER_ADDR", "{}")
    master_port = int(os.environ.get("MASTER_PORT", "{}"))
    rank = int(os.environ.get("RANK", "{}"))
    world_size = int(os.environ.get("WORLD_SIZE", "{}"))

    logger.info("extract the Rabit env from cluster :"
                " %s, port: %d, rank: %d, word_size: %d ",
                master_addr, master_port, rank, world_size)

    return master_addr, master_port, rank, world_size




def train_distributed(xgboost_kwargs: Dict[str, Any]):
    addr, port, rank, world_size = extract_xgbooost_cluster_env()
    rabit_tracker = None

    try:
        """start to build the network"""
        if world_size > 1:
            if rank == 0:
                logger.info("start the master node")

                rabit = RabitTracker(hostIP="0.0.0.0", nslave=world_size,
                                     port=port, port_end=port + 1)
                rabit.start(world_size)
                rabit_tracker = rabit
                logger.info('###### RabitTracker Setup Finished ######')
            # else:
            #     logger.info("I'm not the leader, sleeping for 10 second")
            #     time.sleep(10)
            init_args = {
                'DMLC_NUM_WORKER': world_size,
                'DMLC_TRACKER_URI': addr,
                'DMLC_TRACKER_PORT': port,
                'DMLC_TASK_ID': rank,
                'DMLC_WORKER_CONNECT_RETRY': 10,
            }
            logger.info('##### Rabit rank setup with below envs #####')
            logger.info(init_args)
            xgb.collective.init(**init_args)

            logger.info('##### Rabit rank = %d' % xgb.collective.get_rank())
            rank = xgb.collective.get_rank()

        else:
            world_size = 1
            logging.info("Start the train in a single node")

        xgboost_kwargs["dtrain"] = read_train_data(rank=rank, num_workers=world_size)

        logging.info("starting to train xgboost at node with rank %d", rank)
        bst = xgb.train(**xgboost_kwargs)

        if rank == 0:
            model = bst
        else:
            model = None

        logging.info("finish xgboost training at node with rank %d", rank)

    except Exception as e:
        logger.error("something wrong happen: %s", traceback.format_exc())
        raise e
    finally:
        logger.info("xgboost training job finished!")
        if world_size > 1:
            xgb.collective.finalize()
        if rabit_tracker:
            rabit_tracker.join()

    return model




def main():
    xgboost_kwargs = {}
    xgboost_kwargs["num_boost_round"] = 10
    xgboost_kwargs["params"] = {'max_depth': 2, 'eta': 1, 'silent': 1,
                                'objective': 'multi:softprob', 'num_class': 3}

    model = train_distributed(xgboost_kwargs)
    if model is not None:
        dump_model(model)

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s')
    logging.getLogger().setLevel(logging.INFO)
    main()
