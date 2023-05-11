import submitit
from utils.parser import get_args
from utils.experiments import launch_main
from utils.process import processDataset
# Ensure that all operations are deterministic on GPU (if used) for reproducibility

def main():
    args = get_args()
    experiments=launch_main(args)
    processDataset(experiments[0])

if __name__ == '__main__':
    main()