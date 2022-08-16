import subprocess
import threading


class MyThread(threading.Thread):
    """
    https://www.tutorialspoint.com/python/python_multithreading.htm#:~:text=Multiple%20threads%20within%20a%20process,if%20they%20were%20separate%20processes.
    """
    def __init__(self, threadID, name, num_runs_each, command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_runs_each = num_runs_each
        self.command = command

    def run(self):
        print("Starting " + self.name)
        command = self.command.split(" ")
        command.append("--count")
        command.append(str(self.num_runs_each))
        subprocess.run(command)
        print("Exiting " + self.name)


def main():
    num_threads = 2
    num_runs_each = 10
    command = "wandb agent wubowen/match_cycle/fvm7x82l"
    for i in range(num_threads):
        MyThread(i, f"thread_{i}", num_runs_each, command).start()


if __name__ == "__main__":
    main()