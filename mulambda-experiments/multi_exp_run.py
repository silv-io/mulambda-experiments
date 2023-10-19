from dotenv import load_dotenv
from galileo.shell.shell import init
from galileo.worker.context import Context
from util import create_experiment_job

load_dotenv()
ctx = Context()
rds = ctx.create_redis()
g = init(rds)
exp = g['exp']
telemd = g['telemd']


def run_experiment(exp_id, target, usecase, amount, size, iterations):
    # telemd.start_telemd(NODE_HOSTS)
    exp.start(name=f"exp-{exp_id}-{target}-{usecase}-{amount}-{size}-{iterations}",
              creator="silvio",
              metadata={"exp_id": exp_id, "target": target, "usecase": usecase,
                        "amount": amount, "size": size, "iterations": iterations})

    exp_redis = ctx.create_redis()
    create_experiment_job(exp_id, target, usecase, amount, size, iterations)

    print("starting the wait...")
    pubsub = exp_redis.pubsub()
    pubsub.subscribe("exp/events")

    received_iterations = 0
    for msg in pubsub.listen():
        print(msg)
        if msg['type'] == 'message':
            if msg['data'] == f"{amount} {exp_id}":
                received_iterations += 1
                if received_iterations == iterations:
                    break
    print("wait over!")
    # telemd.stop_telemd()
    exp.stop()


if __name__ == '__main__':
    exp_id = "log-mult-rem"
    targets = ["mulambda-client", "plain-net-latency-client", "random-client",
               "round-robin-client"]
    usecases = ["env", "scp", "mda", "psa"]
    amounts = [100, 1000]
    sizes = [10, 1000]
    iterations = [5, 10]
    exp_amount = len(targets) * len(usecases) * len(amounts) * len(sizes) * len(
        iterations)
    current_exp = 1
    for iteration in iterations:
        for size in sizes:
            for amount in amounts:
                for usecase in usecases:
                    for target in targets:
                        print(f"running experiment {current_exp}/{exp_amount}")
                        run_experiment(exp_id, target, usecase, amount, size, iteration)
                        current_exp += 1
