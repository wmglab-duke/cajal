import argparse


class MutationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 2:
            parser.error(
                "Maximum number of arguments for {0} is 2".format(option_string)
            )

        if any(value >= 2 for value in values):
            parser.error("Maximum mutation bound for {0} is 2".format(option_string))

        if any(value < 0 for value in values):
            parser.error("Minimum mutation bound for {0} is 0".format(option_string))

        setattr(namespace, self.dest, values)


class StrategyAction(argparse.Action):
    options = {
        "best1bin",
        "best1exp",
        "rand1exp",
        "randtobest1exp",
        "currenttobest1exp",
        "best2exp",
        "rand2exp",
        "randtobest1bin",
        "currenttobest1bin",
        "best2bin",
        "rand2bin",
        "rand1bin",
    }

    def __call__(self, parser, namespace, values, option_string=None):
        if values not in self.options:
            parser.error("Select valid mutation strategy: {}".format(self.options))

        setattr(namespace, self.dest, values)


class UpdatingAction(argparse.Action):
    options = {"immediate", "deferred"}

    def __call__(self, parser, namespace, values, option_string=None):
        if values not in self.options:
            parser.error("Select valid updating strategy: {}".format(self.options))

        setattr(namespace, self.dest, values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--job",
        action="store",
        type=str,
        required=True,
        help="Job ID (for saving purposes).",
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        type=int,
        required=True,
        help="Target state for which to optimise (./targets/target_[TARGET]).",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        action="store",
        type=int,
        required=True,
        help="Number of axons to be distributed to each worker when "
        "fragmenting model for MPI execution.",
    )
    parser.add_argument(
        "-u",
        "--updating",
        type=str,
        default="immediate",
        action=UpdatingAction,
        help="Updating method for Differential Evolution. " '(default: "immediate")',
    )
    parser.add_argument(
        "-p",
        "--popsize",
        action="store",
        type=int,
        default=2,
        help="Factor by which to scale number of parameters to get number of "
        "candidates in population. (default: 2)",
    )
    parser.add_argument(
        "-m",
        "--maxiter",
        action="store",
        type=int,
        default=1,
        help="Maximum number of generations to run. (default: 1)",
    )
    parser.add_argument(
        "--mutation",
        type=float,
        nargs="+",
        default=[0.5, 1],
        action=MutationAction,
        help="Mutation rate - float in U[0, 2). If two arguments are "
        "provided, dithering is used in the range U[min_arg, max_arg). "
        "(default: [0.5, 1])",
    )
    parser.add_argument(
        "-r",
        "--recombination",
        type=float,
        nargs=1,
        default=0.8,
        action="store",
        help="Recombination rate in range [0, 1]. (default: 0.8)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="best1bin",
        action=StrategyAction,
        help='Mutation strategy, choice from: {}. (default: "best1bin")'.format(
            StrategyAction.options
        ),
    )
    args = parser.parse_args()

    return args
