# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import getpass
import os
import re
import socket
from argparse import ArgumentParser
from contextlib import closing

import paramiko
import psutil
import torch
from tqdm import tqdm


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def connect_and_execute(ssh, host, commands):
    if not isinstance(commands, list):
        commands = [commands]
    ssh.connect(host)
    for command in commands:
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
        ssh_stdout.channel.recv_exit_status()
    ssh.close()


def parse_python_bash(ssh, host, folder, file):
    ssh.connect(host)
    with ssh.open_sftp() as sftp:
        # parse python command
        with sftp.open(os.path.join(folder, file)) as f:
            python_command = re.sub(
                " +",
                " ",
                "".join(f.readlines())
                .replace("\n", "")
                .replace("\\", "")
                .replace("python3", "")
                .strip(),
            )
            gpus_str = ",".join((str(i) for i in range(gpus_per_node)))
            # fix number of devices
            python_command = re.sub("--devices [^-]* -", f"--devices {gpus_str} -", python_command)
            # add arguments to command
            for i, arg in enumerate(args.extra_script_args, start=1):
                python_command = python_command.replace(f"${i}", arg)
    return python_command


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hostfile", type=str, default=None, required=True)
    parser.add_argument("--bash_file", type=str, required=True)
    parser.add_argument("--clone_repo", action="store_true")
    parser.add_argument("--github_user", type=str, default=None)
    parser.add_argument("--github_repo", type=str, default=None)
    parser.add_argument("--github_branch", type=str, default=None)
    parser.add_argument("--github_token", type=str, default=None)
    parser.add_argument("--extra_script_args", default=[], type=str, nargs="+")

    args = parser.parse_args()
    # gather hosts
    hosts = []
    with open(args.hostfile) as f:
        for line in f:
            host, *_ = line.split()
            hosts.append(host)

    num_nodes = len(hosts)
    gpus_per_node = torch.cuda.device_count()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # clone repo in all hosts
    if args.clone_repo is not None:
        assert args.github_repo is not None
        assert args.github_user is not None

        if args.github_token is None:
            commands = [
                f"rm -rf {args.github_repo}",
                f"git clone https://github.com/{args.github_user}/{args.github_repo}.git",
                f"cd {args.github_repo};git checkout {args.github_branch}",
            ]
        else:
            commands = [
                f"rm -rf {args.github_repo}",
                f"git clone https://{args.github_user}:{args.github_token}@github.com/{args.github_user}/{args.github_repo}.git",
                f"cd {args.github_repo};git checkout {args.github_branch}",
            ]

        for host in tqdm(hosts, desc="Cloning repository into nodes"):
            connect_and_execute(ssh, host, commands)

    # parse python command
    python_command = parse_python_bash(ssh, hosts[0], args.github_repo, args.bash_file)

    # find a free port for distributed
    PORT = find_free_port()

    # execute command
    channels = []
    for i, host in enumerate(tqdm(hosts, desc="Calling distributed_training")):
        # to run with torch.distributed.run (just as backup)
        # command = re.sub(
        #     " +",
        #     " ",
        #     f"""cd {args.github_repo};python3 -m torch.distributed.run \
        #     --nnodes={num_nodes} \
        #     --nproc_per_node={gpus_per_node} \
        #     --master_addr {hosts[0]} \
        #     --master_port {PORT} \
        #     --node_rank {i} \
        #     {python_command} \
        #     --num_nodes {num_nodes}""",
        # )

        command = re.sub(
            " +",
            " ",
            f"""
            cd {args.github_repo}; \
            MASTER_ADDR={hosts[0]} \
            MASTER_PORT={PORT} \
            WORLD_SIZE={num_nodes} \
            NODE_RANK={i} \
            python3 {python_command} \
            --num_nodes {num_nodes}""",
        )

        ssh.connect(host)
        channel = ssh.get_transport().open_session()
        channels.append(channel)
        channel.exec_command(command)

    print("Awaiting for processes to end")
    for channel in channels:
        channel.recv_exit_status()
        channel.close()

    print("Killing all leftovers")
    user_name = getpass.getuser()
    pids = {
        proc.pid
        for proc in psutil.process_iter()
        if proc.username() == user_name and "python3" in proc.name()
    }
    cur_pid = os.getpid()
    for pid in pids:
        if pid != cur_pid:
            psutil.Process(pid).terminate()
