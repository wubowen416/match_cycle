import subprocess
import itertools
import threading


run_filename = "train_eval.py"
num_thread = 2


class MyThread (threading.Thread):
    """
    https://www.tutorialspoint.com/python/python_multithreading.htm#:~:text=Multiple%20threads%20within%20a%20process,if%20they%20were%20separate%20processes.
    """
    def __init__(self, threadID, name, combos, search_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.combos = combos
        self.search_name = search_name

    def run(self):
        print("Starting " + self.name)
        train_combos(self.search_name, self.combos, self.name)
        print("Exiting " + self.name)

def train_combos(search_name, combos, thread_name):
    counter = 0
    base_run_name = f"{search_name}_{thread_name}"
    for combo in combos:
        if combo == None:
            continue
        run_name = base_run_name + "_run_" + str(counter)
        command = ["python", run_filename, "--run_name", run_name]
        lambda_mse_is_0 = False
        lambda_l1_is_0 = False
        lambda_ratio_is_0 = False
        for arg_name, value in combo:
            if arg_name == "lambda_mse":
                if value == 0: lambda_mse_is_0 = True 
            if arg_name == "lambda_l1":
                if value == 0: lambda_l1_is_0 = True
            if arg_name == "lambda_ratio":
                if value == 0: lambda_ratio_is_0 = True
            command.append(f"--{arg_name}")
            command.append(str(value))
        # filter out 0 loss combos
        if lambda_mse_is_0 and lambda_l1_is_0 and lambda_ratio_is_0:
            continue
        counter += 1
        # execute
        print(command)
        subprocess.run(command)


def list_split(listA, n):
    """
    https://appdividend.com/2022/05/30/how-to-split-list-in-python/#:~:text=To%20split%20a%20list%20into%20n%20parts%20in%20Python%2C%20use,array%20into%20multiple%20sub%2Darrays.
    """
    for x in range(0, len(listA), n):
        every_chunk = listA[x: n+x]
        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [None for y in range(n-len(every_chunk))]
        yield every_chunk


def main():
    search_name = "grid_v1"
    grid = {
        "batch_size": [128, 256, 512],
        "lr": [1e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        "momentum": [0.9, 0.93, 0.95, 0.97, 0.99],
        "adagrad": [0.995, 0.997, 0.999],
        "lambda_mse": [0, 0.5, 1.0],
        "lambda_l1": [0, 0.5, 1.0],
        "lambda_ratio": [0, 0.5, 1.0],
        "weight_decay": [0, 1e-4, 1e-3],
        "dropout": [0, 0.1, 0.2]
    }
    # get all combinations
    name_value_pairs = []
    for name, values in grid.items():
        name_value_pair = [(name, v) for v in values]
        name_value_pairs.append(name_value_pair)
    combos = list(itertools.product(*name_value_pairs))
    # split combos
    splited_combos = list(list_split(combos, len(combos)//num_thread))
    print("runs for each thread:", len(splited_combos[0]))
    for i in range(num_thread):
        MyThread(i, f"Thread_{i}", splited_combos[i], search_name).start()


if __name__ == "__main__":
    main()